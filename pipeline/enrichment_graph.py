from __future__ import annotations
import logging
import json
import re
from typing import List, Dict, Any
from neo4j import GraphDatabase
from config.config import load_config
from pipeline.llm_client_local import generate_json as generate_text_json

_cfg = load_config()
log = logging.getLogger("enrichment")
_driver = GraphDatabase.driver(_cfg.neo4j.uri, auth=(_cfg.neo4j.user, _cfg.neo4j.password))


def _load_prompt(filename: str) -> str:
    path = _cfg.prompts_dir / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _clean_json(raw_text: str) -> Dict:
    if not raw_text: return {}
    try: return json.loads(raw_text)
    except: pass

    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    return {}


def get_degree_thresholds(user_id: str) -> tuple[float, float]:
    """
    Calculates thresholds.
    UPDATED: Relaxed to p25 (Orphans) and p75 (Anchors) to catch more candidates.
    """
    #  QUERY: include CREATED edges and ANY path from user's sources
    q = """
    MATCH (u:User {id: $uid})-[:INTERESTED_IN|CREATED]->()--(e:Entity)
    OPTIONAL MATCH (e)-[r:RELATION|RELATED]-(:Entity)
    WITH e, count(r) as degree
    RETURN 
        percentileDisc(degree, 0.25) as p_orphan, 
        percentileDisc(degree, 0.75) as p_anchor
    """
    try:
        with _driver.session() as s:
            res = s.run(q, uid=user_id).single()
            if res:
                p_orphan = max(res['p_orphan'], 1)
                p_anchor = max(res['p_anchor'], 2)
                return p_orphan, p_anchor
    except Exception as e:
        log.error(f"Failed to calc thresholds: {e}")
    
    return 1, 3


def suggest_and_create_links(user_id: str, confidence_threshold: float = 0.6,
                             limit_orphans: int = 40, limit_anchors: int = 20):
    log.info(f"Running Graph Enrichment for User {user_id}...")
    
    p_orphan, p_anchor = get_degree_thresholds(user_id)
    log.info(f"Thresholds: Orphan <= {p_orphan} | Anchor >= {p_anchor}")

    # 🚀 NEW: Unified entity selection query
    q_entities = """
    MATCH (u:User {id: $uid})-[:INTERESTED_IN|CREATED]->()--(e:Entity)
    OPTIONAL MATCH (e)-[r:RELATION|RELATED]-(:Entity)
    WITH e, count(r) as degree
    RETURN DISTINCT e.name as name, e.type as type, e.description as desc, degree
    """

    try:
        with _driver.session() as s:
            entities = s.run(q_entities, uid=user_id).data()
    except Exception as e:
        log.error(f"DB Fetch Error: {e}")
        return 0

    # Classify orphans and anchors via thresholds
    orphans = [e for e in entities if e["degree"] <= p_orphan][:limit_orphans]
    anchors = [e for e in entities if e["degree"] >= p_anchor][:limit_anchors]

    log.info(f"--- DEBUG: Selected Entities ---")
    log.info(f"Orphans ({len(orphans)}): {[o['name'] for o in orphans]}")
    log.info(f"Anchors ({len(anchors)}): {[a['name'] for a in anchors]}")
    log.info(f"--------------------------------")

    if not orphans:
        log.info("No orphans — enrichment not needed.")
        return 0

    context_nodes = []
    seen = set()

    for group, role in [(orphans, "ORPHAN"), (anchors, "ANCHOR")]:
        for e in group:
            if e['name'] not in seen:
                seen.add(e['name'])
                context_nodes.append({
                    "name": e['name'],
                    "type": e['type'],
                    "description": e['desc'],
                    "role": role,
                    "connections": e['degree']
                })

    log.info(f"Prompt Context Size: {len(context_nodes)} Entities")

    template = _load_prompt("graph_enrichment.txt")
    if not template:
        return 0
    prompt = template.replace("{entities_json}", json.dumps(context_nodes, indent=2))

    try:
        raw_output = generate_text_json(prompt, max_tokens=3000)
        if isinstance(raw_output, dict):
            data = raw_output
        elif isinstance(raw_output, str):
            data = _clean_json(raw_output)
        else:
            data = {}
    except Exception as e:
        log.error(f"LLM inference failed: {e}")
        return 0

    suggested_rels = data.get("relations", [])
    log.info(f"LLM suggested {len(suggested_rels)} raw relations.")
    for r in suggested_rels:
        log.info(f"RAW: {r.get('source')} -> {r.get('relation')} -> {r.get('target')} | conf={r.get('confidence', 'MISSING')}")

    
    suggested_rels = [
        r for r in suggested_rels
        if r.get("confidence", 0) >= confidence_threshold
    ]

    log.info(f"{len(suggested_rels)} relations accepted (confidence >= {confidence_threshold})")

    if not suggested_rels:
        log.warning("No high-confidence relations to create.")
        return 0

    for r in suggested_rels:
        log.info(f"  ACCEPT: {r.get('source')} --[{r.get('relation')}]--> {r.get('target')} "
                 f"(conf={r.get('confidence')})")

    # WRITE TO GRAPH — unchanged except using r.confidence
    write_q = """
    UNWIND $rels as r
    MATCH (a:Entity {name: r.source})
    MATCH (b:Entity {name: r.target})
    WHERE NOT (a)-[]-(b)
    MERGE (a)-[rel:RELATED {type: r.relation}]->(b)
    ON CREATE SET
        rel.source = 'llm_enrichment',
        rel.confidence = r.confidence,
        rel.created_at = timestamp()
    RETURN count(rel) as created
    """

    with _driver.session() as s:
        created_count = s.run(write_q, rels=suggested_rels).single()['created']

    log.info(f"Enrichment Success: Created {created_count} new high-confidence links.")
    return created_count
