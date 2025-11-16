# app.py
import streamlit as st
import uuid
import logging

from pipeline.preprocessing import chunk_text
from pipeline.entity_extraction import extract_graph
from pipeline.relation_extractor import extract_relations
from pipeline.graph_builder import build_and_store_graph
from pipeline.neo4j_client import init_indexes, check_apoc

# -------------------------------------------------------------------
# Configure logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("app")

# -------------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="GraphRAG Knowledge Graph Builder", layout="wide")
st.title("GraphRAG Knowledge Graph Builder")

st.write("Enter or paste your text below to extract entities and relationships and build a Neo4j knowledge graph.")

# -------------------------------------------------------------------
# Initialize Neo4j Connection
# -------------------------------------------------------------------
if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = False

if not st.session_state["db_ready"]:
    st.info("Checking Neo4j connection...")
    apoc_available = check_apoc()
    if apoc_available:
        init_indexes()
        st.session_state["db_ready"] = True
        st.success("Connected to Neo4j and initialized indexes.")
    else:
        st.error("APOC not detected. Please enable APOC in Neo4j plugins before proceeding.")
        st.stop()

# -------------------------------------------------------------------
# User Input
# -------------------------------------------------------------------
input_text = st.text_area("Enter text to process:", height=250, placeholder="Paste or type your text here...")

run_relations = st.checkbox("Run additional relation refinement (slower, more detailed)", value=True)

if st.button("Process Text"):
    if not input_text.strip():
        st.warning("Please enter some text before processing.")
    else:
        st.info("Processing input text...")
        chunks = chunk_text(input_text)

        all_entities, all_relations = [], []

        for i, chunk in enumerate(chunks):
            chunk_id = f"user_chunk_{uuid.uuid4().hex[:8]}"
            log.info(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} characters")

            # Step 1: Extract entities and base relations
            graph_data = extract_graph(chunk)
            entities = graph_data.get("entities", [])
            base_relations = graph_data.get("relations", [])
            log.info(f"Extracted {len(entities)} entities and {len(base_relations)} base relations from chunk {chunk_id}")

            # Step 2: Optionally extract refined relations
            refined_relations = []
            if run_relations:
                refined_relations = extract_relations(chunk)
                log.info(f"Extracted {len(refined_relations)} refined relations from chunk {chunk_id}")

            all_relations_chunk = base_relations + refined_relations
            all_entities.extend(entities)
            all_relations.extend(all_relations_chunk)

            # Step 3: Build and store the combined graph in Neo4j
            try:
                build_and_store_graph(chunk_id, chunk, entities, all_relations_chunk)
                log.info(f"Stored chunk {chunk_id} in Neo4j with {len(entities)} entities and {len(all_relations_chunk)} relations.")
            except Exception as e:
                log.error(f"Failed to store chunk {chunk_id}: {e}")

        st.success("Processing complete. Knowledge graph has been built successfully.")
        st.write(f"Total Entities: {len(all_entities)}")
        st.write(f"Total Relations: {len(all_relations)}")

        log.info(f"Total Entities: {len(all_entities)} | Total Relations: {len(all_relations)}")
