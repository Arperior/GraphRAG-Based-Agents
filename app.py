"""
GraphRAG — Ingest + Chat + Memory + Communities
"""

from __future__ import annotations
import uuid
import logging
import logging.handlers
import os
import streamlit as st
from pathlib import Path

# Config import for safe paths
from config.config import load_config

# Pipeline imports
from pipeline.preprocessing import chunk_tokens
from pipeline.entity_extraction import extract_graph
from pipeline.relation_extractor import extract_relations_from_text

# Graph Builder Imports
from pipeline.graph_builder import build_and_store_graph, build_and_store_image

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
from pipeline.pdf_utils import process_pdf_upload

from pipeline.enrichment_graph import suggest_and_create_links
from pipeline.llm_client_vision import process_image_pipeline


# ============================================================================
# CONFIG & LOGGING
# ============================================================================
def setup_logging(logs_dir: Path):
    """
    Configures logging to BOTH console and rotating files.
    """
    # Create logs folder if missing
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Define file path: project/logs/app.log
    log_file_path = logs_dir / "app.log"

    # Define format: Time | Level | Module Name | Message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )

    # Get the Root Logger (The "Parent" of all other loggers)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clean up existing handlers (prevents duplicate logs on Streamlit reload)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # A. File Handler (Saves to app.log)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, 
        maxBytes=5*1024*1024, # 5 MB
        backupCount=5,        # Keep 5 old files
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # B. Console Handler (Prints to Terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Saving to: {log_file_path}")

# Load config
_cfg = load_config()

# Run Setup IMMEDIATELY
setup_logging(_cfg.logs_dir)

# Now create the logger for this specific file
log = logging.getLogger("app")

st.set_page_config(page_title="GraphRAG Chat", layout="wide")
st.title("GraphRAG — Contextual Chat")


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
# CHAT MEMORY
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ============================================================================
# INGESTION PANEL
# ============================================================================
with st.expander("Ingest Data Into Graph", expanded=False):
    
    tab1, tab2, tab3 = st.tabs(["Text Input", "PDF Upload", "Image Upload"])

    # --- TAB 1: TEXT ---
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
                ent_total, rel_total, failed_chunks = 0, 0, 0

                for i, c in enumerate(chunks):
                    status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
                    chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
                    try:
                        data = extract_graph(c)
                        entities = data.get("entities", [])
                        base_rel = data.get("relations", [])
                        refined = []
                        if use_rel_text:
                            try:
                                refined = extract_relations_from_text(c)
                            except Exception as e:
                                log.error(f"Relation extraction skipped for {chunk_id}: {e}")
                        relations = base_rel + refined
                        
                        build_and_store_graph(chunk_id, c, entities, relations, user_id=USER_ID)
                        ent_total += len(entities)
                        rel_total += len(relations)

                    except Exception as e:
                        failed_chunks += 1
                        log.error(f"CRITICAL: Failed to ingest chunk {chunk_id}: {e}", exc_info=True)
                        st.error(f"Chunk {i+1} failed: {e}")
                    
                    progress_bar.progress((i + 1) / len(chunks))

                status_text.text("Done!")
                if failed_chunks > 0:
                    st.warning(f"Ingestion finished with {failed_chunks} failures.")
                else:
                    st.success("Ingestion complete!")
                st.write(f"Entities added: {ent_total}")
                st.write(f"Relations added: {rel_total}")

    # --- TAB 2: PDF ---
    with tab2:
        st.markdown("**PDF Processing:** Extract Text → Summarize (Gemini) → Graph")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        use_rel_pdf = st.checkbox("Enable relation extraction (PDF)", value=True)
        
        if st.button("Process PDF"):
            if uploaded_file:
                with st.spinner("Analyzing PDF..."):
                    try:
                        result = process_pdf_upload(uploaded_file, run_relations=use_rel_pdf, user_id=USER_ID)
                        st.success(f"PDF Processed! Saved to: `{result['txt_path']}`")
                        col1, col2 = st.columns(2)
                        col1.metric("Entities", result['total_entities'])
                        col2.metric("Relations", result['total_relations'])
                        with st.expander("Summary"):
                            st.markdown(result['summary'])
                    except Exception as e:
                        st.error(f"PDF failed: {e}")
            else:
                st.warning("Upload a PDF first.")

    # --- TAB 3: IMAGE (N-MMKG) ---
    with tab3:
        st.markdown("""
        **Image Pipeline (MMGraphRAG Paper Implementation):**
        1. **Segmentation (YOLO)** - Splits image into semantic blocks.
        2. **Description (LLaVA)** - Describes each block in detail.
        3. **Scene Graph Extraction (LLaVA)** - Extracts global entities.
        4. **Alignment (Mistral)** - Maps blocks to entities for rich data.
        5. **Cross-Modal Fusion** - Links visual entities to text graph.
        """)
        
        img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        # User Context Input
        user_context_text = st.text_area(
            "Add Context (Optional)", 
            placeholder="e.g., 'This is the architecture diagram for the Q3 project report.'"
        )
        
        if st.button("Analyze & Ingest Image"):
            if img_file:
                with st.spinner("Running 5-Step Vision Pipeline (YOLO -> LLaVA -> Mistral)..."):
                    try:
                        # 1. Save locally
                        save_dir = _cfg.uploads_dir
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        file_ext = Path(img_file.name).suffix
                        unique_name = f"img_{uuid.uuid4().hex[:8]}{file_ext}"
                        temp_path = save_dir / unique_name
                        
                        with open(temp_path, "wb") as f:
                            f.write(img_file.getbuffer())
                        
                        # 2. Run Pipeline (Now uses process_image_pipeline)
                        scene_graph = process_image_pipeline(
                            str(temp_path), 
                            user_context=user_context_text
                        )
                        
                        num_ents = len(scene_graph.get('entities', []))
                        num_rels = len(scene_graph.get('relations', []))
                        
                        # Check for failure
                        if scene_graph.get("summary") == "Processing Failed":
                            st.error("Vision pipeline failed. Check logs.")
                        else:
                            st.success(f"Analysis Complete! Found {num_ents} entities & {num_rels} relations.")
                            
                            with st.expander("View Enriched Scene Graph"):
                                st.json(scene_graph)
                            
                            # 3. Store & Fusion
                            build_and_store_image(
                                str(temp_path), 
                                scene_graph, 
                                user_id=USER_ID, 
                                user_context=user_context_text
                            )
                            
                            st.success("Ingestion Complete! Image Node created & visual entities fused.")
                        
                    except Exception as e:
                        st.error(f"Image failed: {e}")
                        log.error(f"Image failed: {e}", exc_info=True)
            else:
                st.warning("Upload an image first.")


# ============================================================================
# CHAT PANEL
# ============================================================================
st.header("Chat")

for role, msg in st.session_state["chat_history"]:
    st.chat_message(role).write(msg)

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state["chat_history"].append(("user", prompt))
    _, evidence = gather_evidence_for_query(
        prompt, k_hop=1, per_entity=3, top_entities=4, user_id=USER_ID
    )
    answer = synthesize_answer(
        prompt, evidence, user_id=USER_ID, chat_history=st.session_state["chat_history"]
    )
    st.session_state["chat_history"].append(("assistant", answer))
    store_query_and_answer(USER_ID, prompt, answer)
    st.rerun()


# ============================================================================
# MEMORY & COMMUNITY PANELS
# ============================================================================
with st.expander("User Memory", expanded=False):
    if st.button("Show Memory"):
        txt = get_user_longterm_memory_text(USER_ID, limit=40)
        st.text_area("Memory", txt or "(empty)", height=300)
    if st.button("Clear Memory"):
        clear_user_memory(USER_ID)
        st.success("Cleared.")

with st.expander("Communities & Leiden", expanded=False):
    if st.button("Run Leiden (User-Scoped)"):
        n = run_leiden(user_id=USER_ID)
        st.success(f"Leiden complete — {n} communities")
    if st.button("View Summaries"):
        summaries = summarize_communities(force_refresh=False, user_id=USER_ID)
        for cid, txt in summaries:
            st.markdown(f"### Community {cid}")
            st.write(txt)
    if st.button("Regenerate Summaries"):
        summarize_communities(force_refresh=True, user_id=USER_ID)
        st.success("Regenerated.")
# ============================================================================
# GRAPH ENRICHMENT PANEL
# ============================================================================
with st.expander("Graph Enrichment (Connect the Dots)", expanded=False):
    st.markdown("""
    **Semantic Linking:**
    This tool scans for "Orphaned" entities (isolated nodes from recent uploads) 
    and uses the LLM to link them to "Anchor" entities (well-connected nodes) 
    already in your graph.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        # Increase defaults here too
        orphan_limit = st.number_input("New Nodes to Link", min_value=10, max_value=100, value=40)
    with col2:
        anchor_limit = st.number_input("Existing Anchors Context", min_value=10, max_value=100, value=20)

    if st.button("Run Graph Enrichment"):
        with st.spinner(f"Analyzing graph for isolated nodes ({USER_ID})..."):
            try:
                count = suggest_and_create_links(
                    USER_ID, 
                    limit_orphans=orphan_limit, 
                    limit_anchors=anchor_limit
                )
                
                if count > 0:
                    st.success(f"Success! Created {count} new semantic connections.")
                    st.balloons()
                else:
                    st.info("No high-confidence connections found (or graph is already well-connected).")
                    
            except Exception as e:
                st.error(f"Enrichment failed: {e}")
                log.error(f"Enrichment failed: {e}", exc_info=True)