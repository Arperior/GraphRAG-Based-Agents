# pipeline/clustering.py
from __future__ import annotations
from typing import List, Tuple
import logging
import igraph as ig
import leidenalg as la

from pipeline.neo4j_client import _driver
from config.config import load_config
from pipeline.llm_client_gemini import gemini_complete

_cfg = load_config()
log = logging.getLogger("clustering")


def _export_entities_and_edges(user_id: str | None = None):
    """
    Export nodes and edges from Neo4j.
    If user_id is provided, export ONLY the subgraph of entities that the user
    has interacted with via chunks (User)-[:INTERESTED_IN]->(Chunk).
    """
    nodes, edges = [], []

    with _driver.session() as s:
        if user_id:
            log.info(f"Exporting user-specific graph for user_id={user_id}")

            # 1) Export entities linked to chunks the user interacted with
            q_nodes = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e:Entity)
            RETURN id(e) AS id, e.name AS name
            """
            node_rows = s.run(q_nodes, uid=user_id)
            for r in node_rows:
                nodes.append((r["id"], r["name"]))

            # 2) Export edges among those same entities
            q_edges = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)
            MATCH (c)<-[:MENTIONED_IN]-(a:Entity)-[rel:RELATION]->(b:Entity)-[:MENTIONED_IN]->(c)
            RETURN id(a) AS a, id(b) AS b, coalesce(rel.confidence,1.0) AS w
            """
            edge_rows = s.run(q_edges, uid=user_id)
            for r in edge_rows:
                edges.append((r["a"], r["b"], float(r["w"] or 1.0)))

        else:
            # ORIGINAL GLOBAL CLUSTERING
            for r in s.run("MATCH (e:Entity) RETURN id(e) as id, e.name as name"):
                nodes.append((r["id"], r["name"]))
            for r in s.run(
                "MATCH (a:Entity)-[rel:RELATION]->(b:Entity) "
                "RETURN id(a) as a, id(b) as b, coalesce(rel.confidence,1.0) as w"
            ):
                edges.append((r["a"], r["b"], float(r["w"] or 1.0)))

    log.info(f"Exported {len(nodes)} nodes and {len(edges)} edges (user_id={user_id}).")
    return nodes, edges



def run_leiden(resolution: float | None = None, user_id: str | None = None) -> int:
    """
    Compute Leiden communities and write community id back to Entity nodes.
    Uses config-based resolution if not provided.
    Returns the number of unique communities.
    """
    resolution = resolution or _cfg.leiden_resolution
    log.info(f"Running Leiden clustering with resolution={resolution}")

    nodes, edges = _export_entities_and_edges(user_id)
    if not nodes:
        log.warning("No nodes found in database — skipping clustering.")
        return 0

    id2idx = {neo_id: i for i, (neo_id, _) in enumerate(nodes)}
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["neo_id"] = [nid for nid, _ in nodes]
    g.vs["name"] = [name for _, name in nodes]

    if edges:
        edge_tuples = []
        weights = []
        for a, b, w in edges:
            ia, ib = id2idx.get(a), id2idx.get(b)
            if ia is None or ib is None:
                continue
            if ia == ib:
                continue
            u, v = (ia, ib) if ia < ib else (ib, ia)
            edge_tuples.append((u, v))
            weights.append(w)
        if edge_tuples:
            unique = {}
            for (u, v), w in zip(edge_tuples, weights):
                unique[(u, v)] = unique.get((u, v), 0.0) + w
            edges_final = list(unique.keys())
            weights_final = list(unique.values())
            g.add_edges(edges_final)
            g.es["weight"] = weights_final
    else:
        log.warning("No edges found — clustering will operate on isolated vertices.")

    try:
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=g.es["weight"] if g.ecount() else None,
            resolution_parameter=float(resolution),
        )
        membership = partition.membership
    except Exception as e:
        log.error(f"Leiden clustering failed: {e}", exc_info=True)
        return 0

    idx2id = {idx: neo_id for neo_id, idx in id2idx.items()}
    with _driver.session() as s:
        tx = s.begin_transaction()
        try:
            for idx, comm in enumerate(membership):
                neo_id = idx2id.get(idx)
                if neo_id is None:
                    continue
                tx.run(
                    "MATCH (e) WHERE id(e)=$id "
                    "SET e.community=$c "
                    "WITH e "
                    "MERGE (com:Community {id:$c}) "
                    "SET com.last_updated=timestamp() "
                    "MERGE (e)-[:IN_COMMUNITY]->(com)",
                    id=int(neo_id),
                    c=int(comm),
                )
            tx.commit()
        except Exception as e:
            log.error(f"Error writing communities to Neo4j: {e}", exc_info=True)
            tx.rollback()
            return 0

    n_comms = len(set(membership))
    log.info(f"Leiden clustering complete — {n_comms} communities detected.")
    return n_comms


def summarize_communities(force_refresh: bool = False, max_relations: int = 250) -> List[Tuple[int, str]]:
    """
    For each community, assemble intra-community relations and ask Gemini
    to produce a short summary. Writes/updates (:Community {id, summary}).
    If force_refresh=False, skip summaries that already exist.
    """
    log.info("Generating community summaries via Gemini (force_refresh=%s).", force_refresh)
    # List communities and their existing summaries
    q_list = "MATCH (c:Community) RETURN c.id AS id, c.summary AS summary ORDER BY c.id"
    outputs = []
    with _driver.session() as s:
        rows = s.run(q_list).data()

    # If no communities exist, return empty
    if not rows:
        log.info("No communities present.")
        return []

    for row in rows:
        comm = row["id"]
        existing = row.get("summary")
        if existing and not force_refresh:
            outputs.append((int(comm), existing))
            log.info(f"Community {comm} summary loaded from cache.")
            continue

        # assemble relations for this community
        q = """
        MATCH (a)-[r:RELATION]->(b)
        WHERE (a)-[:IN_COMMUNITY]->(:Community {id:$cid}) AND (b)-[:IN_COMMUNITY]->(:Community {id:$cid})
        RETURN collect({src:a.name, rel:type(r), tgt:b.name}) AS rels
        """
        with _driver.session() as s:
            data = s.run(q, cid=int(comm)).single()
        rels = data["rels"] or []
        lines = "\n".join(f"{x['src']} -[{x['rel']}]-> {x['tgt']}" for x in rels[:max_relations]) or "(no edges)"

        prompt_path = _cfg.prompts_dir / "community_report.txt"
        try:
            prompt_template = prompt_path.read_text(encoding="utf-8")
        except:
            prompt_template = "Summarize the following relations:\n\n{community_reports}"

        prompt = prompt_template.replace("{community_reports}", lines)


        try:
            summary = gemini_complete(prompt, max_tokens=400)
            with _driver.session() as s:
                s.run("MERGE (c:Community {id:$id}) SET c.summary=$s, c.summary_updated=timestamp()", id=int(comm), s=summary)
            outputs.append((int(comm), summary))
            log.info(f"Community {comm} summarized ({len(rels)} relations).")
        except Exception as e:
            log.error(f"Failed to summarize community {comm}: {e}", exc_info=True)
            continue

    log.info(f"Summarized {len(outputs)} communities.")
    return outputs
