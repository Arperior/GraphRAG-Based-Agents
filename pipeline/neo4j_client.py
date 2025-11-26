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
        if not e: continue
        if isinstance(e, dict):
            name = e.get("name", "").strip()
            if not name: continue
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
        if not src or not tgt: continue

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
            s.run(q_main, cid=chunk.id, text=chunk.text, source=chunk.source, entities=ent_dicts, relations=rel_dicts)
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
    log.info(f"Storing Scene Graph for Image {image_id} ({len(entities)} ents). Context len: {len(user_context or '')}")

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
        i.user_context = $context,  // <--- Store User Context here
        i.created_at = timestamp()
    
    WITH i
    UNWIND $entities AS e
      MERGE (n:Entity {name:e.name})
      ON CREATE SET n.type = e.type, 
                    n.description = e.description,
                    n.modality = 'visual',  // Mark as visual origin
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
            # Pass 'context' parameter to the query
            s.run(q_image, 
                  iid=image_id, 
                  path=image_path, 
                  summary=summary, 
                  context=user_context or "",  # Handle None safely
                  entities=ent_dicts, 
                  relations=rel_dicts)
            
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
                q=q, limit=limit
            )
            return res.data()
    except Exception as e:
        log.error(f"Entity search failed: {e}")
        return []


def k_hop_chunks(entity_name: str, k: int = 1, limit: int | None = None) -> List[Dict]:
    """
    Returns evidence (Text Chunks OR Images) k hops away from an entity.
    """
    limit = limit or _cfg.neo4j_query_limit
    
    # Updated to fetch BOTH Chunks and Images linked to the entity
    q = """
    MATCH (e:Entity)
    WHERE toLower(e.name) = toLower($name)
    CALL apoc.path.subgraphNodes(e, {relationshipFilter:'RELATION>|DEPICTS|MENTIONED_IN', maxLevel:$k})
    YIELD node
    WITH DISTINCT node
    WHERE node:Chunk OR node:Image
    RETURN labels(node)[0] as type, node.id as cid, node.text as text, node.summary as summary, node.path as path
    LIMIT $limit
    """
    try:
        with _driver.session() as s:
            res = s.run(q, name=entity_name, k=k, limit=limit)
            data = res.data()
            # Normalize output so retrieval.py can use it
            results = []
            for r in data:
                if r["type"] == "Image":
                    # Treat image summary as text for retrieval context
                    results.append({"cid": r["cid"], "text": f"[IMAGE SUMMARY] {r.get('summary','')} (File: {r.get('path','')})"})
                else:
                    results.append({"cid": r["cid"], "text": r.get("text","")})
            return results
    except Exception as e:
        log.error(f"k-hop retrieval failed: {e}")
        return []