"""
GraphRAG — Ingest + Chat + Memory + Communities
"""

from __future__ import annotations
import uuid
import logging
import os
import streamlit as st

# Pipeline imports
from pipeline.preprocessing import chunk_tokens
from pipeline.entity_extraction import extract_graph
from pipeline.relation_extractor import extract_relations_from_text
from pipeline.graph_builder import build_and_store_graph

from pipeline.neo4j_client import init_indexes, check_apoc
from pipeline.retrieval import gather_evidence_for_query, synthesize_answer

from pipeline.memory import (
    ensure_user_exists,
    store_query_and_answer,
    list_users,
    get_user_longterm_memory_text,
    list_user_memories,
    clear_user_memory,
)

from pipeline.clustering import run_leiden, summarize_communities
from pipeline.llm_client_gemini import gemini_complete

# NEW: PDF Processing Import
from pipeline.pdf_utils import process_pdf_upload


# ============================================================================
# STREAMLIT CONFIG
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

st.set_page_config(page_title="GraphRAG Chat", layout="wide")
st.title("GraphRAG — Contextual Chat")

log = logging.getLogger("app")


# ============================================================================
# USER SELECTION (SIDEBAR)
# ============================================================================
try:
    default_user = st.secrets.get("user_id", None)
except Exception:
    default_user = None

default_user = default_user or os.environ.get("USER_ID", "user_default")

existing_users = list_users(limit=100)
if default_user not in existing_users:
    existing_users = [default_user] + existing_users

selected = st.sidebar.selectbox("User ID", existing_users)
new_user = st.sidebar.text_input("Create new user")

if new_user.strip():
    selected = new_user.strip()

ensure_user_exists(selected)
st.session_state["user_id"] = selected
USER_ID = selected


# ============================================================================
# DATABASE INIT
# ============================================================================
if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = False

if not st.session_state["db_ready"]:
    st.info("Connecting to Neo4j...")
    if check_apoc():
        init_indexes()
        ensure_user_exists(USER_ID)
        st.session_state["db_ready"] = True
        st.success("Neo4j connected.")
    else:
        st.error("APOC not detected — enable APOC and restart.")
        st.stop()


# ============================================================================
# CHAT MEMORY (session-only working memory)
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ============================================================================
# INGESTION PANEL
# ============================================================================
with st.expander("Ingest Data Into Graph", expanded=False):
    
    # We use tabs to switch between raw text paste and PDF upload
    tab1, tab2 = st.tabs(["Text Input", "PDF Upload"])

    # ------------------------------------------------------------------------
    # TAB 1: RAW TEXT INPUT
    # ------------------------------------------------------------------------
    with tab1:
        st.markdown("Paste text → chunk → extract entities & relations → Neo4j")

        text_input = st.text_area("Input text", height=200)
        use_rel_text = st.checkbox("Enable relation extraction (Text)", value=True)

        if st.button("Ingest Text"):
            if not text_input.strip():
                st.warning("Please enter text")
            else:
                chunks = chunk_tokens(text_input)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                ent_total = 0
                rel_total = 0
                failed_chunks = 0

                for i, c in enumerate(chunks):
                    status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
                    chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"

                    try:
                        # 1. Extract Entities
                        data = extract_graph(c)
                        entities = data.get("entities", [])
                        base_rel = data.get("relations", [])

                        # 2. Refine Relations (Protected against crash)
                        refined = []
                        if use_rel_text:
                            try:
                                refined = extract_relations_from_text(c)
                            except Exception as e:
                                log.error(f"Relation extraction skipped for {chunk_id}: {e}")
                        
                        relations = base_rel + refined

                        # 3. Build & Store (Protected against crash)
                        build_and_store_graph(chunk_id, c, entities, relations, user_id=USER_ID)

                        ent_total += len(entities)
                        rel_total += len(relations)

                    except Exception as e:
                        failed_chunks += 1
                        log.error(f"CRITICAL: Failed to ingest chunk {chunk_id}: {e}", exc_info=True)
                        st.error(f"Chunk {i+1} failed: {e}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(chunks))

                status_text.text("Done!")
                
                if failed_chunks > 0:
                    st.warning(f"Ingestion finished with {failed_chunks} failures.")
                else:
                    st.success("Ingestion complete!")
                
                st.write(f"Entities added: {ent_total}")
                st.write(f"Relations added: {rel_total}")

    # ------------------------------------------------------------------------
    # TAB 2: PDF UPLOAD
    # ------------------------------------------------------------------------
    with tab2:
        st.markdown("""
        **PDF Processing Pipeline:**
        1. Extract Text
        2. **Summarize** via Gemini (to reduce noise & cost)
        3. Extract Graph from Summary
        """)
        
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        use_rel_pdf = st.checkbox("Enable relation extraction (PDF)", value=True)
        
        if st.button("Process PDF"):
            if uploaded_file:
                with st.spinner("Analyzing PDF... this uses Gemini for summarization..."):
                    try:
                        # process_pdf_upload handles the full cycle: 
                        # read -> summarize -> chunk -> extract -> store
                        result = process_pdf_upload(uploaded_file, run_relations=use_rel_pdf,user_id=USER_ID)
                        
                        st.success(f"PDF Processed! Saved reference text to: `{result['txt_path']}`")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Entities Added", result['total_entities'])
                        col2.metric("Relations Added", result['total_relations'])
                        
                        with st.expander("View Generated Summary (Used for Graph)"):
                            st.markdown(result['summary'])
                            
                    except Exception as e:
                        st.error(f"PDF Processing failed: {e}")
                        log.error(f"PDF Processing failed: {e}", exc_info=True)
            else:
                st.warning("Please upload a PDF file first.")


# ============================================================================
# CHAT PANEL
# ============================================================================
st.header("Chat")

# Show history
for role, msg in st.session_state["chat_history"]:
    st.chat_message(role).write(msg)

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state["chat_history"].append(("user", prompt))

    # Retrieve evidence for query automatically
    _, evidence = gather_evidence_for_query(
        prompt,
        k_hop=1,
        per_entity=3,
        top_entities=4,
        user_id=USER_ID
    )

    # Generate final answer
    answer = synthesize_answer(
        prompt,
        evidence,
        user_id=USER_ID,
        chat_history=st.session_state["chat_history"],
        use_plan=False
    )

    st.session_state["chat_history"].append(("assistant", answer))

    # Save memory
    store_query_and_answer(USER_ID, prompt, answer)

    st.rerun()


# ============================================================================
# MEMORY PANEL
# ============================================================================
with st.expander("User Memory", expanded=False):

    if st.button("Show long-term memory summary"):
        txt = get_user_longterm_memory_text(USER_ID, limit=40)
        st.text_area("Memory", txt or "(empty)", height=300)

    if st.button("List memory items"):
        rows = list_user_memories(USER_ID, limit=100)
        if not rows:
            st.info("No memory found.")
        else:
            for r in rows:
                st.write(f"• {r.get('value')}  (ts={r.get('created')})")

    if st.button("Clear memory (danger)"):
        clear_user_memory(USER_ID)
        st.success("Memory cleared.")


# ============================================================================
# COMMUNITY / LEIDEN PANEL
# ============================================================================
with st.expander("Communities & Leiden", expanded=False):

    if st.button("Run Leiden clustering (User-Scoped)"):
        # UPDATED: Pass user_id
        n = run_leiden(user_id=USER_ID)
        st.success(f"Leiden complete — {n} communities (for user {USER_ID})")

    if st.button("View community summaries (cached)"):
        # UPDATED: Pass user_id to only fetch relevant communities
        summaries = summarize_communities(force_refresh=False, user_id=USER_ID)
        if not summaries:
            st.info("No summaries exist yet.")
        else:
            for cid, txt in summaries:
                st.markdown(f"### Community {cid}")
                st.write(txt)

    if st.button("Regenerate all community summaries (Gemini)"):
        # UPDATED: Pass user_id
        summaries = summarize_communities(force_refresh=True, user_id=USER_ID)
        st.success(f"Regenerated {len(summaries)} summaries.")