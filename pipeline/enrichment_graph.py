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
    if not raw_text:
        return {}
    try:
        return json.loads(raw_text)
    except:
        pass

    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    return {}


def get_degree_thresholds(user_id: str) -> tuple[int, int]:
    """
    Compute orphan/anchor thresholds only within user's graph.
    """
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
                return max(res['p_orphan'], 1), max(res['p_anchor'], 2)
    except Exception as e:
        log.error(f"Failed degree thresholds: {e}")

    return 1, 3


def _fetch_user_entities(user_id: str) -> List[Dict[str, Any]]:
    """
    Fetch entire user-owned subgraph entities with degree.
    """
    q = """
    MATCH (u:User {id: $uid})-[:INTERESTED_IN|CREATED]->()--(e:Entity)
    OPTIONAL MATCH (e)-[r:RELATION|RELATED]-(:Entity)
    WITH e, count(r) as degree
    RETURN DISTINCT e.name as name, e.type as type, e.description as desc, degree
    """
    try:
        with _driver.session() as s:
            return s.run(q, uid=user_id).data()
    except Exception as e:
        log.error(f"Entity fetch failed: {e}")
        return []


def suggest_and_create_links(
    user_id: str,
    confidence_threshold: float = 0.6,
    limit_orphans: int = 40,
    limit_anchors: int = 20,
):
    log.info(f"🔍 Enrichment run for User {user_id}")

    p_orphan, p_anchor = get_degree_thresholds(user_id)
    log.info(f"Thresholds → Orphan <= {p_orphan} | Anchor >= {p_anchor}")

    # Fetch entire user graph entities
    entities = _fetch_user_entities(user_id)
    if not entities:
        log.info("No user entities found.")
        return 0

    BLACKLIST = {"icon", "jpg", "png", "image", "button", "logo", "building icon", "vector"}

    def is_valid(e: Dict[str, Any]) -> bool:
        name = (e.get("name") or "").lower()
        return (
            len(name) >= 3
            and not name.isdigit()
            and all(b not in name for b in BLACKLIST)
        )

    entities = [e for e in entities if is_valid(e)]
    if not entities:
        log.info("Filtered out all entities.")
        return 0

    # Isolation: newly introduced nodes = very low degree
    orphans = [e for e in entities if e["degree"] <= p_orphan][:limit_orphans]

    # Anchors: well-connected nodes already in user's knowledge
    anchors = [e for e in entities if e["degree"] >= p_anchor][:limit_anchors]

    log.info(f"New entities (orphans): {len(orphans)}")
    log.info(f"Existing user knowledge (anchors): {len(anchors)}")

    if not orphans:
        log.info("No new information — skipping enrichment.")
        return 0

    # Build prompt strings
    orphan_names = [e["name"] for e in orphans]
    anchor_names = [e["name"] for e in anchors]

    orphans_str = ", ".join(orphan_names)
    anchors_str = ", ".join(anchor_names)

    template = _load_prompt("graph_enrichment.txt")
    if not template:
        log.error("Missing prompt template 'graph_enrichment.txt'")
        return 0

    prompt = (
        template.replace("{orphans_list}", orphans_str)
        .replace("{anchors_list}", anchors_str)
    )

    # Query LLM
    raw_output = generate_text_json(prompt, max_tokens=2000)
    data = raw_output if isinstance(raw_output, dict) else _clean_json(raw_output)

    suggested_rels = data.get("relations", []) or []
    log.info(f"LLM Proposed: {len(suggested_rels)} candidate links")

    # Validate & dedup
    valid_rels = []
    signatures = set()

    for r in suggested_rels:
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        rel = str(r.get("relation", "")).strip().upper()
        conf = float(r.get("confidence", 0) or 0)

        if conf < confidence_threshold:
            continue
        if not src or not tgt or src == tgt:
            continue

        sig = (src, tgt, rel)
        if sig in signatures:
            continue

        signatures.add(sig)
        valid_rels.append({"source": src, "target": tgt, "relation": rel, "confidence": conf})

    log.info(f"Accepted Relations: {len(valid_rels)} above confidence {confidence_threshold}")

    if not valid_rels:
        return 0

    # Write relationships — scoped only to this user's nodes
    write_q = """
    UNWIND $rels as r
    MATCH (u:User {id: $uid})
    MATCH (a:Entity {name:r.source})<-[:INTERESTED_IN|CREATED]-(u)
    MATCH (b:Entity {name:r.target})<-[:INTERESTED_IN|CREATED]-(u)
    MERGE (a)-[rel:RELATED {type:r.relation}]->(b)
    ON CREATE SET rel.source = 'llm_enrichment',
                  rel.confidence = r.confidence,
                  rel.created_at = timestamp()
    RETURN count(rel) as created
    """

    with _driver.session() as s:
        created_count = s.run(write_q, rels=valid_rels, uid=user_id).single()['created']

    log.info(f"✨ Created {created_count} new internal user-graph links.")
    return created_count
