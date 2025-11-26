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
    Ensure core indexes and constraints exist.
    UPDATED: Added index for :Image nodes.
    """
    cyphers = [
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
        # NEW: Index for Image lookup
        "CREATE INDEX image_id_idx IF NOT EXISTS FOR (i:Image) ON (i.id)",
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
    """Checks if APOC plugin is installed."""
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


# =========================================================
# 1. TEXT STORAGE (Standard GraphRAG)
# =========================================================
def store_chunk_with_graph(
    chunk: Chunk | dict,
    user_id: str | None,
    entities: List[Dict] | List[str],
    relations: List[Dict]
):
    """
    Insert one Chunk + Entities + Relations into Neo4j.
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
                relations=rel_dicts,
            )
            if user_id:
                s.run(q_interest, uid=str(user_id).strip(), cid=chunk.id)
        log.info(f"Chunk {chunk.id} stored successfully.")
    except Exception as e:
        log.error(f"Failed to store chunk {chunk.id}: {e}", exc_info=True)
        raise


# =========================================================
# 2. IMAGE STORAGE (N-MMKG Implementation)
# =========================================================
def store_image_scene_graph(
    image_id: str,
    image_path: str,
    summary: str,
    entities: List[Dict],
    relations: List[Dict],
    user_id: str | None,
    user_context: str | None = None  # <--- NEW: Accept user context
):
    """
    Stores Image -> VisualEntities with optional User Context.
    Structure: (User)-[:INTERESTED_IN]->(Image {user_context:...})-[:DEPICTS]->(Entity)
    """
    log.info(
        f"Storing Scene Graph for Image {image_id} ({len(entities)} ents). "
        f"Context len: {len(user_context or '')}"
    )

    # Prepare entity data
    ent_dicts = []
    for e in entities:
        name = e.get("name", "").strip()
        if name:
            ent_dicts.append({
                "name": name,
                "type": e.get("type", "VISUAL"),
                "description": e.get("description", "Visual object")
            })

    # Prepare relation data
    rel_dicts = []
    for r in relations:
        src = r.get("source", "").strip()
        tgt = r.get("target", "").strip()
        if src and tgt:
            rel_dicts.append({
                "src": src,
                "tgt": tgt,
                "relation": r.get("relation", "RELATED_TO").upper().replace(" ", "_")
            })

    # Query: Create Image Node (with Summary & User Context) & Link to Entities
    q_image = """
    MERGE (i:Image {id:$iid})
    SET i.path = $path, 
        i.summary = $summary,
        i.user_context = $context,
        i.created_at = timestamp()
    
    WITH i
    UNWIND $entities AS e
      MERGE (n:Entity {name:e.name})
      ON CREATE SET n.type = e.type, 
                    n.description = e.description,
                    n.modality = 'visual',
                    n.first_seen = timestamp()
      ON MATCH SET n.modality = coalesce(n.modality, 'visual')
      
      // The Paper's N-MMKG link: Image DEPICTS Entity
      MERGE (i)-[:DEPICTS]->(n)

    WITH i
    UNWIND $relations AS r
      MERGE (a:Entity {name:r.src})
      MERGE (b:Entity {name:r.tgt})
      MERGE (a)-[rel:RELATION {relation:r.relation}]->(b)
      ON CREATE SET rel.type = 'visual'
    """

    # Link User to Image
    q_user_image = """
    MERGE (u:User {id:$uid})
    MERGE (i:Image {id:$iid})
    MERGE (u)-[:INTERESTED_IN]->(i)
    """

    try:
        with _driver.session() as s:
            s.run(
                q_image,
                iid=image_id,
                path=image_path,
                summary=summary,
                context=user_context or "",
                entities=ent_dicts,
                relations=rel_dicts,
            )

            if user_id:
                s.run(q_user_image, uid=str(user_id).strip(), iid=image_id)

        log.info(f"Image {image_id} stored successfully.")
    except Exception as e:
        log.error(f"Failed to store image graph: {e}", exc_info=True)


# =========================================================
# 3. SEARCH & FUSION HELPERS
# =========================================================
def search_potential_matches(entity_name: str) -> List[str]:
    """Find existing entities with similar names (Case-insensitive substring)."""
    q = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($name) 
       OR toLower($name) CONTAINS toLower(e.name)
    RETURN DISTINCT e.name AS name
    LIMIT 5
    """
    try:
        with _driver.session() as s:
            rows = s.run(q, name=entity_name).data()
            return [r["name"] for r in rows if r["name"] != entity_name]
    except Exception as e:
        log.warning(f"Search match failed: {e}")
        return []


def merge_entities(target_name: str, visual_name: str):
    """
    Merges the new 'visual_name' node INTO the existing 'target_name' node.
    Requires APOC.
    """
    q = """
    MATCH (target:Entity {name:$t_name})
    MATCH (visual:Entity {name:$v_name})
    WHERE id(target) <> id(visual)
    CALL apoc.refactor.mergeNodes([target, visual], {
        properties: {
            description:'combine', 
            modality:'combine'
        },
        mergeRels: true
    })
    YIELD node
    RETURN count(*)
    """
    try:
        with _driver.session() as s:
            s.run(q, t_name=target_name, v_name=visual_name)
        log.info(f"Cross-Modal Fusion: Merged '{visual_name}' -> '{target_name}'")
    except Exception as e:
        log.error(f"Merge failed for '{visual_name}' -> '{target_name}': {e}")


# =========================================================
# 4. RETRIEVAL
# =========================================================
def search_entities_contains(q: str, limit: int | None = None) -> List[Dict]:
    limit = limit or _cfg.retrieval_search_limit
    try:
        with _driver.session() as s:
            res = s.run(
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS toLower($q) "
                "RETURN e.name as name, id(e) as id, e.community as community "
                "LIMIT $limit",
                q=q,
                limit=limit,
            )
            return res.data()
    except Exception as e:
        log.error(f"Entity search failed: {e}")
        return []


def k_hop_chunks(
    entity_name: str,
    k: int = 1,
    limit: int | None = None,
    user_id: str | None = None,
) -> List[Dict]:
    """
    Expands *only within the user's personal subgraph*.
    Ensures no entities leak from other users.
    Returns only Chunks / Images / Entities.
    """
    if not user_id:
        log.warning("k_hop_chunks called without user_id, returning empty.")
        return []

    limit = limit or _cfg.neo4j_query_limit
    uid = str(user_id).strip()

    q = f"""
    MATCH (u:User {{id:$uid}})-[:INTERESTED_IN]->(src)
    MATCH (src)-[:MENTIONED_IN|DEPICTS]->(root:Entity)

    // Find base or enriched neighbour node
    MATCH (root)-[:RELATION|RELATED*0..2]-(e1:Entity {{name:$name}})

    // Expand only along permitted relations
    OPTIONAL MATCH p=(e1)-[:RELATION|RELATED*1..{k}]-(nbr)

    WITH DISTINCT nbr
    WHERE nbr IS NOT NULL AND (nbr:Chunk OR nbr:Image OR nbr:Entity)

    RETURN DISTINCT
        labels(nbr)[0] AS type,
        CASE WHEN nbr:Chunk OR nbr:Image THEN nbr.id ELSE nbr.name END AS cid,
        nbr.text AS text,
        nbr.summary AS summary,
        nbr.path AS path,
        nbr.name AS name,
        nbr.description AS description,
        nbr.modality AS modality
    LIMIT $limit
    """

    try:
        with _driver.session() as s:
            rows = s.run(q, uid=uid, name=entity_name, limit=limit).data()

        res = []
        for r in rows:
            ntype = r["type"]
            cid = r["cid"]
            text = (
                r.get("text")
                or r.get("summary")
                or r.get("description")
                or r.get("name")
                or ""
            )
            res.append(
                {
                    "type": ntype,
                    "cid": str(cid),
                    "text": text,
                    "name": r.get("name"),
                    "modality": r.get("modality"),
                }
            )
        return res
    except Exception as e:
        log.error(f"k_hop_chunks failed for entity={entity_name}: {e}", exc_info=True)
        return []
