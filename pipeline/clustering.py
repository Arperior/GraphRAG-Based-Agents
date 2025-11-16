# pipeline/clustering.py
from __future__ import annotations
from typing import List, Tuple
import igraph as ig
import leidenalg as la
import logging
from neo4j import GraphDatabase

from config.config import load_config
from pipeline.llm_client_gemini import gemini_complete

_cfg = load_config()
_drv = GraphDatabase.driver(_cfg.neo4j.uri, auth=(_cfg.neo4j.user, _cfg.neo4j.password))
log = logging.getLogger("neo4j")


def _export_entities_and_edges():
    """Export (id, name) nodes and (a, b, weight) edges from Neo4j."""
    nodes, edges = [], []
    with _drv.session() as s:
        for r in s.run("MATCH (e:Entity) RETURN id(e) as id, e.name as name"):
            nodes.append((r["id"], r["name"]))
        for r in s.run(
            "MATCH (a:Entity)-[rel:RELATION]->(b:Entity) "
            "RETURN id(a) as a, id(b) as b, coalesce(rel.confidence,1.0) as w"
        ):
            edges.append((r["a"], r["b"], r["w"]))
    log.info(f"Exported {len(nodes)} nodes and {len(edges)} edges from Neo4j.")
    return nodes, edges


def run_leiden(resolution: float | None = None) -> int:
    """
    Compute Leiden communities and write community id back to Entity nodes.
    Uses config-based resolution if not provided.
    Returns the number of unique communities.
    """
    resolution = resolution or _cfg.leiden_resolution
    log.info(f"Running Leiden clustering with resolution={resolution}")

    nodes, edges = _export_entities_and_edges()
    if not nodes:
        log.warning("No nodes found in database — skipping clustering.")
        return 0

    id2idx = {neo_id: i for i, (neo_id, _) in enumerate(nodes)}
    g = ig.Graph(directed=True)
    g.add_vertices(len(nodes))
    g.vs["neo_id"] = [nid for nid, _ in nodes]
    g.vs["name"] = [name for _, name in nodes]

    if edges:
        g.add_edges([(id2idx[a], id2idx[b]) for a, b, _ in edges])
        g.es["weight"] = [w for _, _, w in edges]
    else:
        log.warning("No edges found — clustering may be meaningless.")

    try:
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=g.es["weight"] if g.ecount() else None,
            resolution_parameter=resolution
        )
        membership = part.membership
    except Exception as e:
        log.error(f"Leiden clustering failed: {e}")
        return 0

    idx2id = {v: k for k, v in id2idx.items()}
    with _drv.session() as s:
        for i, comm in enumerate(membership):
            s.run("MATCH (e) WHERE id(e)=$id SET e.community=$c", id=idx2id[i], c=int(comm))

    n_comms = len(set(membership))
    log.info(f"Leiden clustering complete — {n_comms} communities detected.")
    return n_comms


def summarize_communities() -> List[Tuple[int, str]]:
    """
    For each community, assemble intra-community relations and ask Gemini
    to produce a short summary. Writes/updates (:Community {id, summary}).
    """
    log.info("Generating community summaries via Gemini.")
    q = """
    MATCH (e:Entity)
    WITH DISTINCT e.community AS comm
    MATCH (a:Entity {community:comm})-[r:RELATION]->(b:Entity {community:comm})
    WITH comm, collect({src:a.name, rel:type(r), tgt:b.name}) AS rels
    RETURN comm, rels
    """
    outputs: List[Tuple[int, str]] = []
    with _drv.session() as s:
        data = s.run(q).data()

    for row in data:
        comm = row["comm"]
        rels = row.get("rels") or []
        lines = "\n".join(f"{x['src']} -[{x['rel']}]-> {x['tgt']}" for x in rels[:250]) or "(no edges)"
        prompt_path = _cfg.prompts_dir / "community_report_graph.txt"
        prompt = prompt_path.read_text(encoding="utf-8").replace("{community_data}", lines)

        try:
            summary = gemini_complete(prompt, max_tokens=400)
            with _drv.session() as s:
                s.run("MERGE (c:Community {id:$id}) SET c.summary=$s", id=int(comm), s=summary)
            outputs.append((int(comm), summary))
            log.info(f"Community {comm} summarized ({len(rels)} relations).")
        except Exception as e:
            log.error(f"Failed to summarize community {comm}: {e}")
            continue

    log.info(f"Summarized {len(outputs)} communities.")
    return outputs
