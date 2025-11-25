# pipeline/graph_builder.py
from __future__ import annotations
from typing import Dict, List, Iterable
import logging

from pipeline.neo4j_client import store_chunk_with_graph
from pipeline.memory import record_user_chunk_interest

log = logging.getLogger("graph_builder")


def _normalize_entity(e) -> Dict:
    if isinstance(e, dict):
        name = (e.get("name") or "").strip()
        return {
            "name": name,
            "type": e.get("type", "UNKNOWN"),
            "description": e.get("description", "") or ""
        } if name else None
    elif isinstance(e, str):
        n = e.strip()
        return {"name": n, "type": "UNKNOWN", "description": ""} if n else None
    return None


def _normalize_relation(r) -> Dict:
    """
    Produce a relation dict with keys:
      - src
      - tgt
      - relation
      - evidence
      - confidence (float)
    Accepts relation input from either stage (entity_extraction or relation_extractor).
    """
    if not r or not isinstance(r, dict):
        return None

    src = (r.get("source") or r.get("src") or r.get("from") or "").strip()
    tgt = (r.get("target") or r.get("tgt") or r.get("to") or "").strip()
    if not src or not tgt:
        return None

    relation = r.get("relation") or r.get("rel") or "RELATED_TO"
    evidence = r.get("evidence") or r.get("ev") or ""
    try:
        conf = float(r.get("confidence", 1.0) or 1.0)
    except Exception:
        conf = 1.0

    return {"src": src, "tgt": tgt, "relation": relation, "evidence": evidence, "confidence": conf}


def build_and_store_graph(
    chunk_id: str,
    chunk_text: str,
    entities: List[Dict] | None,
    relations: List[Dict] | None,
    user_id: str | None = None,
    source: str = "user_text",
    query: str | None = None,
    tokens: List[str] | None = None,
    extra_relations: Iterable[Dict] | None = None,
):
    """
    Build the normalized entities + relations payload and store to Neo4j.
    - entities: list from entity_extraction
    - relations: list from entity_extraction (may be empty)
    - extra_relations: optional list from relation_extractor (will be merged)
    If user_id is provided, we also record (User)-[:INTERESTED_IN]->(Chunk) via memory helper.
    """
    try:
        # Normalize entities
        ent_list: List[Dict] = []
        for e in (entities or []):
            n = _normalize_entity(e)
            if n:
                ent_list.append(n)

        # Normalize relations coming from primary extractor
        rels: List[Dict] = []
        for r in (relations or []):
            nr = _normalize_relation(r)
            if nr:
                rels.append(nr)

        # Append/merge any extra relations (e.g., relation_extractor output)
        for r in (extra_relations or []):
            nr = _normalize_relation(r)
            if nr:
                # avoid duplicates (simple check)
                key = (nr["src"], nr["tgt"], nr["relation"])
                if not any((x["src"], x["tgt"], x["relation"]) == key for x in rels):
                    rels.append(nr)

        chunk_obj = {"id": chunk_id, "text": chunk_text, "source": source}

        log.info(f"Building graph chunk {chunk_id}: {len(ent_list)} entities, {len(rels)} relations")

        # Store into Neo4j via the central client
        store_chunk_with_graph(chunk_obj, user_id, ent_list, rels)

        # If user context given, record the interest with tokens/query (memory)
        if user_id:
            try:
                # record_user_chunk_interest will MERGE and set r.count, r.tokens, r.last_query
                record_user_chunk_interest(user_id, chunk_id, query or "", tokens or [])
            except Exception as e:
                log.warning(f"Failed to record user-chunk interest for {chunk_id}: {e}")

        log.info(f"Successfully stored graph chunk {chunk_id}")

    except Exception as e:
        log.error(f"Failed to build/store graph chunk {chunk_id}: {e}", exc_info=True)
        raise
