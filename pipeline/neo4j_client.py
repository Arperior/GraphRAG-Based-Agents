# pipeline/neo4j_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import logging
from neo4j import GraphDatabase
from config.config import load_config

_cfg = load_config()
log = logging.getLogger("neo4j")

# Initialize Neo4j driver
_driver = GraphDatabase.driver(
    _cfg.neo4j.uri,
    auth=(_cfg.neo4j.user, _cfg.neo4j.password)
)


@dataclass
class Chunk:
    """Represents a text chunk node in the graph."""
    id: str
    text: str
    source: str = "user_text"


def init_indexes():
    """
    Ensure core indexes and constraints exist for performance and data integrity.
    """
    cyphers = [
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
        "CREATE INDEX community_idx IF NOT EXISTS FOR (c:Community) ON (c.id)",
        "CREATE INDEX user_id_idx IF NOT EXISTS FOR (u:User) ON (u.id)"
    ]
    with _driver.session() as s:
        for c in cyphers:
            try:
                s.run(c)
                log.info(f"Executed index/constraint: {c}")
            except Exception as e:
                log.warning(f"Failed to execute: {c} — {e}")


def check_apoc() -> bool:
    """
    Checks if APOC plugin is installed and callable in Neo4j.
    Returns True if available, False otherwise.
    """
    try:
        with _driver.session() as s:
            result = s.run("RETURN apoc.version() AS version").single()
            if result and result["version"]:
                log.info(f"[OK] APOC detected: {result['version']}")
                return True
            else:
                log.warning("[WARN] APOC returned no version; may be partially installed.")
                return False
    except Exception as e:
        log.warning(f"[WARN] APOC check failed: {e}")
        return False


def store_chunk_with_graph(
    chunk: Chunk | dict,
    user_id: str | None,
    entities: List[Dict] | List[str],
    relations: List[Dict]
):
    """
    Insert one Chunk + Entities + Relations into Neo4j.
    Optionally attach (User)-[:INTERESTED_IN]->(Chunk) when user_id provided.

    Notes:
      - relations: list of dicts with keys: src, tgt, relation, evidence, confidence
      - we MERGE relation edges on the relation property too so distinct relation types
        between the same two entities become separate edges.
    """
    if isinstance(chunk, dict):
        chunk = Chunk(**chunk)

    # normalize entities
    ent_dicts = []
    for e in entities or []:
        if not e:
            continue
        if isinstance(e, dict):
            name = e.get("name", "").strip()
            if not name:
                continue
            ent_dicts.append({
                "name": name,
                "type": e.get("type", "unknown"),
                "description": e.get("description", "")
            })
        else:
            ent_dicts.append({
                "name": str(e).strip(),
                "type": "unknown",
                "description": ""
            })

    # normalize relations
    rel_dicts = []
    for r in relations or []:
        src = str(r.get("src") or r.get("source") or "").strip()
        tgt = str(r.get("tgt") or r.get("target") or "").strip()
        if not src or not tgt:
            continue

        rel_dicts.append({
            "src": src,
            "tgt": tgt,
            "relation": r.get("relation", "RELATED_TO"),
            "evidence": r.get("evidence", ""),
            "confidence": float(r.get("confidence", 1.0) or 1.0),
        })

    log.info(f"Storing chunk {chunk.id}: {len(ent_dicts)} entities, {len(rel_dicts)} relations")

    ###########################################################################
    # CYPHER: create chunk, entities, provenance (MENTIONED_IN) and RELATION edges
    ###########################################################################
    q_main = """
    MERGE (c:Chunk {id:$cid})
      SET c.text=$text,
          c.source=$source,
          c.created_at=timestamp()
    WITH c
    UNWIND $entities AS e
      MERGE (n:Entity {name:e.name})
        ON CREATE SET n.type=e.type,
                      n.description=e.description,
                      n.first_seen=timestamp()
      MERGE (n)-[:MENTIONED_IN]->(c)
    WITH c
    UNWIND $relations AS r
      MERGE (a:Entity {name:r.src})
      MERGE (b:Entity {name:r.tgt})
      MERGE (a)-[rel:RELATION {relation:r.relation}]->(b)
      SET rel.evidence = r.evidence,
          rel.confidence = r.confidence,
          rel.last_seen = timestamp()
    """

    q_interest = """
    MERGE (u:User {id:$uid})
    MERGE (c:Chunk {id:$cid})
    MERGE (u)-[r:INTERESTED_IN]->(c)
      ON CREATE SET r.count = 1, r.last_seen = timestamp()
      ON MATCH SET  r.count = coalesce(r.count,0) + 1,
                    r.last_seen = timestamp()
    """

    try:
        with _driver.session() as s:
            s.run(
                q_main,
                cid=chunk.id,
                text=chunk.text,
                source=chunk.source,
                entities=ent_dicts,
                relations=rel_dicts
            )

            if user_id:
                s.run(q_interest, uid=str(user_id).strip(), cid=chunk.id)

        log.info(f"Chunk {chunk.id} stored successfully in Neo4j.")
    except Exception as e:
        log.error(f"Failed to store chunk {chunk.id}: {e}", exc_info=True)
        raise


def search_entities_contains(q: str, limit: int | None = None) -> List[Dict]:
    limit = limit or _cfg.retrieval_search_limit
    log.info(f"Searching entities containing '{q}' (limit={limit})")

    try:
        with _driver.session() as s:
            res = s.run(
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS toLower($q) "
                "RETURN e.name as name, id(e) as id, e.community as community "
                "LIMIT $limit",
                q=q, limit=limit
            )
            data = res.data()
            return data
    except Exception as e:
        log.error(f"Entity search failed for query '{q}': {e}")
        return []


def k_hop_chunks(entity_name: str, k: int = 1, limit: int | None = None) -> List[Dict]:
    """
    Returns chunk evidence k hops away from an entity.
    Tries APOC path expansion, but falls back to a pure-Cypher variable-length traversal
    if APOC is not available.
    Case-insensitive entity match (toLower compare).
    """
    limit = limit or _cfg.neo4j_query_limit
    log.info(f"Fetching {k}-hop neighborhood for '{entity_name}' (limit={limit})")

    q_apoc = """
    MATCH (e:Entity)
    WHERE toLower(e.name) = toLower($name)
    CALL apoc.path.subgraphNodes(e, {relationshipFilter:'RELATION>', maxLevel:$k})
    YIELD node
    WITH DISTINCT node WHERE node:Chunk
    RETURN node.id as cid, node.text as text
    LIMIT $limit
    """

    q_fallback = """
    MATCH (e:Entity)
    WHERE toLower(e.name) = toLower($name)
    MATCH (e)-[:RELATION*1..$k]->(x:Entity)
    MATCH (x)-[:MENTIONED_IN]->(c:Chunk)
    RETURN DISTINCT c.id as cid, c.text as text
    LIMIT $limit
    """

    try:
        with _driver.session() as s:
            try:
                res = s.run(q_apoc, name=entity_name, k=k, limit=limit)
                data = res.data()
                log.info(f"APOC: Retrieved {len(data)} chunks for '{entity_name}' (k={k})")
                return data
            except Exception as inner:
                log.warning(f"APOC query failed, falling back to pure-cypher: {inner}")
                res = s.run(q_fallback, name=entity_name, k=k, limit=limit)
                data = res.data()
                log.info(f"Fallback: Retrieved {len(data)} chunks for '{entity_name}' (k={k})")
                return data
    except Exception as e:
        log.error(f"Failed k-hop retrieval for '{entity_name}': {e}", exc_info=True)
        return []
