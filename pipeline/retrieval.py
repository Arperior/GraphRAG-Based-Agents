from __future__ import annotations
from typing import List, Dict
import logging
import time

from pipeline.neo4j_client import _driver
from pipeline.utils import truncate
from config.config import load_config

log = logging.getLogger("retrieval")
_cfg = load_config()


def get_contextual_subgraph(entity_name: str, k: int = 1, limit: int | None = None) -> Dict:
    limit = limit or _cfg.neo4j_query_limit
    q = """
    MATCH (e:Entity {name:$name})
    CALL apoc.path.subgraphAll(e, {maxLevel:$k}) YIELD nodes, relationships
    WITH nodes, relationships
    RETURN
        [n IN nodes WHERE n:Entity | {name:n.name, community:n.community}] AS entities,
        [r IN relationships | {
            src:startNode(r).name,
            rel:type(r),
            tgt:endNode(r).name,
            evidence:r.evidence,
            confidence:r.confidence
        }] AS rels
    LIMIT $limit
    """
    log.info(f"Fetching contextual subgraph for entity='{entity_name}', k={k}, limit={limit}")
    start = time.time()
    try:
        with _driver.session() as s:
            res = s.run(q, name=entity_name, k=k, limit=limit).data()
        duration = time.time() - start
        log.info(f"Subgraph query completed in {duration:.3f}s — found {len(res)} records.")
        if res:
            return res[0]
        else:
            log.warning(f"No subgraph results found for entity='{entity_name}'.")
            return {"entities": [], "rels": []}
    except Exception as e:
        log.error(f"Error retrieving subgraph for entity='{entity_name}': {e}")
        return {"entities": [], "rels": []}


def gather_evidence(query: str, k_hop: int = 1, per_entity: int | None = None) -> tuple[list[str], str]:
    per_entity = per_entity or 3
    ents: List[str] = []
    evidences: List[str] = []
    log.info(f"Gathering evidence for query='{query}', k_hop={k_hop}, per_entity={per_entity}")

    try:
        with _driver.session() as s:
            res = s.run(
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS toLower($q) "
                "RETURN e.name as name "
                "LIMIT $limit",
                q=query,
                limit=_cfg.retrieval_search_limit,
            )
            ents = [r["name"] for r in res]
        log.info(f"Found {len(ents)} matching entities for query='{query}'.")
    except Exception as e:
        log.error(f"Error while searching entities for query='{query}': {e}")
        return [], ""

    for e in ents:
        try:
            sg = get_contextual_subgraph(e, k=k_hop)
            rel_count = len(sg.get("rels", []))
            if rel_count == 0:
                log.debug(f"No relations found for entity='{e}'.")
                continue
            for rel in sg["rels"][:per_entity]:
                evidence_text = (
                    f"({rel['src']}) -[{rel['rel']}]-> ({rel['tgt']}) "
                    f"[Conf:{rel.get('confidence', 1)}] : "
                    f"{truncate(rel.get('evidence', ''), 300)}"
                )
                evidences.append(evidence_text)
            log.info(f"Collected {min(per_entity, rel_count)} relations for entity='{e}'.")
        except Exception as e_sub:
            log.error(f"Error gathering subgraph for entity='{e}': {e_sub}")
            continue

    total_rels = len(evidences)
    log.info(f"Evidence collection complete — {len(ents)} entities, {total_rels} relations total.")
    return ents, "\n".join(evidences)
