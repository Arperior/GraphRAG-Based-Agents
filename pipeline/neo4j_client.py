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
        "CREATE INDEX community_idx IF NOT EXISTS FOR (c:Community) ON (c.id)"
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


def store_chunk_with_graph(chunk: Chunk | dict, entities: List[Dict] | List[str], relations: List[Dict]):
    """
    Efficiently insert one Chunk, its Entities, and Relations in a single transaction using UNWIND.
    """
    if isinstance(chunk, dict):
        chunk = Chunk(**chunk)

    ent_dicts = [
        e if isinstance(e, dict) else {"name": e, "type": "UNKNOWN", "description": ""}
        for e in entities if e
    ]

    rel_dicts = [
        {
            "src": r.get("source"),
            "tgt": r.get("target"),
            "rel": r.get("relation", "RELATED_TO"),
            "ev": r.get("evidence", ""),
            "conf": float(r.get("confidence", 1.0) or 1.0),
        }
        for r in relations if r.get("source") and r.get("target")
    ]

    log.info(f"Storing chunk {chunk.id}: {len(ent_dicts)} entities, {len(rel_dicts)} relations")

    q = """
    MERGE (c:Chunk {id:$cid})
      SET c.text=$text, c.source=$source, c.created_at=timestamp()
    WITH c
    UNWIND $entities AS e
      MERGE (n:Entity {name:e.name})
        ON CREATE SET n.type=e.type, n.description=e.description, n.first_seen=timestamp()
      MERGE (n)-[:MENTIONED_IN]->(c)
    WITH c
    UNWIND $relations AS r
      MERGE (a:Entity {name:r.src})
      MERGE (b:Entity {name:r.tgt})
      MERGE (a)-[rel:RELATION {type:r.rel, chunk_id:$cid}]->(b)
        SET rel.confidence=r.conf, rel.evidence=r.ev
    """

    try:
        with _driver.session() as s:
            s.run(
                q,
                cid=chunk.id,
                text=chunk.text,
                source=chunk.source,
                entities=ent_dicts,
                relations=rel_dicts
            )
        log.info(f"Chunk {chunk.id} stored successfully in Neo4j.")
    except Exception as e:
        log.error(f"Failed to store chunk {chunk.id}: {e}")
        raise


def search_entities_contains(q: str, limit: int | None = None) -> List[Dict]:
    """
    Search for entities whose names partially match a given string.
    This is user-facing, so it uses retrieval_search_limit from config.
    """
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
            log.info(f"Found {len(data)} matching entities for query '{q}'.")
            return data
    except Exception as e:
        log.error(f"Entity search failed for query '{q}': {e}")
        return []


def k_hop_chunks(entity_name: str, k: int = 1, limit: int | None = None) -> List[Dict]:
    """
    Returns chunk evidence k hops away from an entity.
    Uses APOC for subgraph expansion.
    This is an internal traversal — uses neo4j_query_limit from config.
    """
    limit = limit or _cfg.neo4j_query_limit
    log.info(f"Fetching {k}-hop neighborhood for '{entity_name}' (limit={limit})")

    q = """
    MATCH (e:Entity {name:$name})
    CALL apoc.path.subgraphNodes(e, {relationshipFilter:'RELATION>', maxLevel:$k})
    YIELD node
    WITH DISTINCT node WHERE node:Chunk
    RETURN node.id as cid, node.text as text
    LIMIT $limit
    """

    try:
        with _driver.session() as s:
            res = s.run(q, name=entity_name, k=k, limit=limit)
            data = res.data()
            log.info(f"Retrieved {len(data)} chunks for '{entity_name}' (k={k})")
            return data
    except Exception as e:
        log.error(f"Failed k-hop retrieval for '{entity_name}': {e}")
        return []
