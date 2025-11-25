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
    Export nodes and edges for clustering.
    
    Weights:
    - Explicit RELATION: 1.0 (or confidence)
    - Co-occurrence (same chunk): 0.5
    
    FILTERS: If user_id is provided, ONLY exports entities/relations 
    connected to chunks the user has interacted with.
    """
    nodes = []
    # We use a dict to accumulate weights: (id_a, id_b) -> weight
    edge_weights = {}

    with _driver.session() as s:
        # 1. Fetch Nodes
        if user_id:
            q_nodes = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e:Entity)
            RETURN distinct id(e) AS id, e.name AS name
            """
            node_rows = s.run(q_nodes, uid=user_id)
        else:
            q_nodes = "MATCH (e:Entity) RETURN id(e) as id, e.name as name"
            node_rows = s.run(q_nodes)
        
        for r in node_rows:
            nodes.append((r["id"], r["name"]))

        # 2. Fetch Explicit Relations (Strong connections)
        if user_id:
            # Only relations where both entities appear in a chunk the user owns/saw
            q_explicit = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)
            MATCH (c)<-[:MENTIONED_IN]-(a:Entity)-[rel:RELATION]->(b:Entity)-[:MENTIONED_IN]->(c)
            RETURN id(a) as a, id(b) as b, coalesce(rel.confidence, 1.0) as w
            """
            rels_rows = s.run(q_explicit, uid=user_id)
        else:
            q_explicit = """
            MATCH (a:Entity)-[rel:RELATION]->(b:Entity)
            RETURN id(a) as a, id(b) as b, coalesce(rel.confidence, 1.0) as w
            """
            rels_rows = s.run(q_explicit)

        for r in rels_rows:
            u, v = r["a"], r["b"]
            if u > v: u, v = v, u
            edge_weights[(u, v)] = edge_weights.get((u, v), 0.0) + float(r["w"])

        # 3. Fetch Co-occurrence Relations (Implicit connections)
        # If user_id is set, the chunk 'c' MUST be one the user is INTERESTED_IN
        if user_id:
            q_cooc = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)
            MATCH (a:Entity)-[:MENTIONED_IN]->(c)<-[:MENTIONED_IN]-(b:Entity)
            WHERE id(a) < id(b)
            RETURN id(a) as a, id(b) as b, count(c) as freq
            """
            cooc_rows = s.run(q_cooc, uid=user_id)
        else:
            q_cooc = """
            MATCH (a:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(b:Entity)
            WHERE id(a) < id(b)
            RETURN id(a) as a, id(b) as b, count(c) as freq
            """
            cooc_rows = s.run(q_cooc)

        for r in cooc_rows:
            u, v = r["a"], r["b"]
            w = 0.5 * r["freq"] 
            edge_weights[(u, v)] = edge_weights.get((u, v), 0.0) + w

    # Flatten edges for igraph
    edges = []
    for (u, v), w in edge_weights.items():
        edges.append((u, v, w))

    log.info(f"Exported {len(nodes)} nodes and {len(edges)} weighted edges (User={user_id}).")
    return nodes, edges


def run_leiden(resolution: float | None = None, user_id: str | None = None) -> int:
    """
    Compute Leiden communities using explicit + implicit edges.
    """
    resolution = resolution or _cfg.leiden_resolution
    log.info(f"Running Leiden clustering with resolution={resolution} (User={user_id})")

    nodes, edges = _export_entities_and_edges(user_id)
    if not nodes:
        log.warning("No nodes found — skipping clustering.")
        return 0

    id2idx = {neo_id: i for i, (neo_id, _) in enumerate(nodes)}
    g = ig.Graph()
    g.add_vertices(len(nodes))
    g.vs["neo_id"] = [nid for nid, _ in nodes]
    g.vs["name"] = [name for _, name in nodes]

    if edges:
        edge_list = []
        weights = []
        for a, b, w in edges:
            ia, ib = id2idx.get(a), id2idx.get(b)
            if ia is not None and ib is not None:
                edge_list.append((ia, ib))
                weights.append(w)
        
        g.add_edges(edge_list)
        g.es["weight"] = weights
    else:
        log.warning("No edges found even after co-occurrence check.")

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
    
    # Write back to Neo4j
    with _driver.session() as s:
        tx = s.begin_transaction()
        try:
            for idx, comm in enumerate(membership):
                neo_id = idx2id.get(idx)
                if neo_id is None: continue
                # We overwrite the community ID for these entities
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


def summarize_communities(
    force_refresh: bool = False, 
    max_items: int = 150, 
    user_id: str | None = None
) -> List[Tuple[int, str]]:
    """
    Summarize communities.
    If user_id is provided, only summarizes communities that contain 
    entities linked to the user's chunks.
    """
    log.info("Generating community summaries (force_refresh=%s, User=%s).", force_refresh, user_id)
    
    with _driver.session() as s:
        # 1. Identify which communities to summarize
        if user_id:
            # Only communities reachable from the user's chunks
            q_list = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e:Entity)-[:IN_COMMUNITY]->(com:Community)
            RETURN distinct com.id AS id, com.summary AS summary 
            ORDER BY com.id
            """
            rows = s.run(q_list, uid=user_id).data()
        else:
            rows = s.run("MATCH (c:Community) RETURN c.id AS id, c.summary AS summary ORDER BY c.id").data()

    outputs = []
    if not rows:
        return []

    for row in rows:
        comm_id = row["id"]
        existing = row.get("summary")
        if existing and not force_refresh:
            outputs.append((int(comm_id), existing))
            continue

        # 2. Fetch Evidence (Relations)
        # If user_id is set, we only pull relations verified by user's chunks
        if user_id:
            q_rels = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(k:Chunk)
            MATCH (k)<-[:MENTIONED_IN]-(a)-[r:RELATION]->(b)-[:MENTIONED_IN]->(k)
            WHERE (a)-[:IN_COMMUNITY]->(:Community {id:$cid}) 
              AND (b)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN distinct a.name as src, type(r) as rel, b.name as tgt
            LIMIT $limit
            """
            with _driver.session() as s:
                rel_data = s.run(q_rels, uid=user_id, cid=int(comm_id), limit=max_items).data()
        else:
            q_rels = """
            MATCH (a)-[r:RELATION]->(b)
            WHERE (a)-[:IN_COMMUNITY]->(:Community {id:$cid}) 
              AND (b)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN distinct a.name as src, type(r) as rel, b.name as tgt
            LIMIT $limit
            """
            with _driver.session() as s:
                rel_data = s.run(q_rels, cid=int(comm_id), limit=max_items).data()

        # 3. Fetch Evidence (Entities fallback)
        # If relation query returns nothing, just list the entities
        if user_id:
            q_ents = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e:Entity)
            WHERE (e)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN distinct e.name as name, e.description as desc
            LIMIT $limit
            """
            with _driver.session() as s:
                ent_data = s.run(q_ents, uid=user_id, cid=int(comm_id), limit=max_items).data()
        else:
            q_ents = """
            MATCH (e:Entity)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN distinct e.name as name, e.description as desc
            LIMIT $limit
            """
            with _driver.session() as s:
                ent_data = s.run(q_ents, cid=int(comm_id), limit=max_items).data()

        # Build Prompt
        context_text = ""
        prompt_instruction = ""

        if rel_data:
            lines = [f"{x['src']} -[{x['rel']}]-> {x['tgt']}" for x in rel_data]
            context_text = "Relationships:\n" + "\n".join(lines)
            prompt_instruction = "Summarize the following relationships within this community."
        elif ent_data:
            lines = [f"{x['name']} ({x.get('desc') or 'no desc'})" for x in ent_data]
            context_text = "Members:\n" + ", ".join(lines)
            prompt_instruction = (
                "This community has no direct internal links, but contains these entities. "
                "Describe the common theme or topic unifying these entities."
            )
        else:
            # Community might exist but user has no data in it? Skip.
            continue

        prompt = f"{prompt_instruction}\n\n{context_text}\n\nSummary:"

        try:
            summary = gemini_complete(prompt, max_tokens=400)
            with _driver.session() as s:
                s.run("MERGE (c:Community {id:$id}) SET c.summary=$s, c.summary_updated=timestamp()", 
                      id=int(comm_id), s=summary)
            outputs.append((int(comm_id), summary))
            log.info(f"Community {comm_id} summarized.")
        except Exception as e:
            log.error(f"Failed to summarize community {comm_id}: {e}")

    return outputs