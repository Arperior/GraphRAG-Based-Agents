from __future__ import annotations
import logging
from typing import Dict, List
from pipeline.neo4j_client import store_chunk_with_graph

log = logging.getLogger("graph_builder")

def build_and_store_graph(chunk_id: str, chunk_text: str, entities: List[Dict], relations: List[Dict], source: str = "user_text"):
    """Merge entities and relations into a single graph chunk and push to Neo4j."""
    try:
        chunk_obj = {
            "id": chunk_id,
            "text": chunk_text,
            "source": source
        }
        log.info(f"Building graph chunk {chunk_id}: {len(entities)} entities, {len(relations)} relations")
        store_chunk_with_graph(chunk_obj, entities, relations)
        log.info(f"Successfully stored graph chunk {chunk_id}")
    except Exception as e:
        log.error(f"Failed to store chunk {chunk_id}: {e}")
        raise
