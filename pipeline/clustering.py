# pipeline/clustering.py
from __future__ import annotations
from typing import List, Tuple, Dict
import logging

import igraph as ig
import leidenalg as la

from config.config import load_config
from pipeline.neo4j_client import _driver
from pipeline.llm_client_gemini import gemini_complete

_cfg = load_config()
log = logging.getLogger("clustering")

# ---------------------------------------------------------------------------
# GRAPH EXPORT (ENTITIES + IMAGES, EXPLICIT + CO-OCCURRENCE EDGES)
# ---------------------------------------------------------------------------
def _export_entities_and_edges(
    user_id: str | None = None,
    min_cooccurrence: int = 1,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, int, float]]]:
    """
    Export nodes + edges for Leiden clustering.

    Nodes:
      - Entities reachable from the user's subgraph via Text OR Image.
        (User)-[:INTERESTED_IN]->(Chunk)<-[:MENTIONED_IN]-(Entity)
        (User)-[:INTERESTED_IN]->(Image)-[:DEPICTS]->(Entity)
      - Image nodes themselves are now included as clusterable nodes.

    Edges (undirected weights):
      1. Explicit semantic edges:
         - (a:Entity)-[:RELATION|RELATED]->(b:Entity)
         - (img:Image)-[:DEPICTS]-(e:Entity)  (Strong visual link)
      2. Text co-occurrence edges:
         - Mentioned together in the same Chunk.

    If user_id is None, we fall back to global graph.
    """

    log.info(
        "Exporting entities/images + edges for Leiden (user=%s, min_co=%s)",
        user_id,
        min_cooccurrence,
    )

    nodes: Dict[int, str] = {}  # neo4j id -> label name
    edges: Dict[Tuple[int, int], float] = {}  # (id1,id2) sorted -> weight

    with _driver.session() as s:
        # -----------------------------
        # 1) NODES (Entities & Images)
        # -----------------------------
        if user_id:
            # Fetch Root Entities from both Chunks (MENTIONED_IN) and Images (DEPICTS)
            q_entities = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(src)
            // Traverse from Chunk<-Entity OR Image->Entity
            OPTIONAL MATCH (src)<-[:MENTIONED_IN]-(root1:Entity)
            OPTIONAL MATCH (src)-[:DEPICTS]->(root2:Entity)
            WITH COLLECT(DISTINCT root1) + COLLECT(DISTINCT root2) AS roots
            UNWIND roots AS r
            WITH DISTINCT r
            WHERE r IS NOT NULL
            // Enriched neighbours via semantic edges
            OPTIONAL MATCH (r)-[:RELATION|RELATED*0..2]-(e:Entity)
            WITH COLLECT(DISTINCT r) + COLLECT(DISTINCT e) AS allE
            UNWIND allE AS e
            WITH DISTINCT e
            RETURN id(e) AS id, e.name AS name
            """
            rows_e = s.run(q_entities, uid=user_id).data()

            # Fetch Images directly connected to User
            # (User)-[:INTERESTED_IN]->(Image)
            q_images = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(img:Image)
            RETURN DISTINCT id(img) AS id,
                   coalesce(img.id, 'image_' + toString(id(img))) AS name
            """
            rows_i = s.run(q_images, uid=user_id).data()
        else:
            q_entities = """
            MATCH (e:Entity)
            RETURN id(e) AS id, e.name AS name
            """
            rows_e = s.run(q_entities).data()

            q_images = """
            MATCH (img:Image)
            RETURN DISTINCT id(img) AS id,
                   coalesce(img.id, 'image_' + toString(id(img))) AS name
            """
            rows_i = s.run(q_images).data()

    for r in rows_e:
        nodes[r["id"]] = r.get("name") or f"entity_{r['id']}"
    for r in rows_i:
        nodes[r["id"]] = r.get("name") or f"image_{r['id']}"

    log.info("Export: %d nodes (entities + images)", len(nodes))

    if not nodes:
        return [], []

    with _driver.session() as s:
        # -----------------------------
        # 2) EXPLICIT SEMANTIC EDGES (Entity-Entity)
        # -----------------------------
        # Links entities regardless of whether they came from text or image
        if user_id:
            q_rel = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(src)
            MATCH (src)-[:MENTIONED_IN|DEPICTS*1]-(a:Entity)
            MATCH (a)-[r:RELATION|RELATED]->(b:Entity)
            // Ensure b is also relevant/reachable (optimization)
            RETURN DISTINCT id(a) AS a, id(b) AS b,
                   coalesce(r.confidence, 1.0) AS w
            """
            rel_rows = s.run(q_rel, uid=user_id).data()
        else:
            q_rel = """
            MATCH (a:Entity)-[r:RELATION|RELATED]->(b:Entity)
            RETURN id(a) AS a, id(b) AS b,
                   coalesce(r.confidence, 1.0) AS w
            """
            rel_rows = s.run(q_rel).data()

        for r in rel_rows:
            a, b = r["a"], r["b"]
            if a not in nodes or b not in nodes:
                continue
            key = tuple(sorted((a, b)))
            w = float(r.get("w") or 1.0)
            edges[key] = edges.get(key, 0.0) + w

        # -----------------------------
        # 3) VISUAL EDGES (Image DEPICTS Entity)
        # -----------------------------
        # Strong link to bind images to the entities they represent
        if user_id:
            q_dep = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(img:Image)
            MATCH (img)-[:DEPICTS]->(e:Entity)
            RETURN DISTINCT id(e) AS e, id(img) AS img
            """
            dep_rows = s.run(q_dep, uid=user_id).data()
        else:
            q_dep = """
            MATCH (img:Image)-[:DEPICTS]->(e:Entity)
            RETURN DISTINCT id(e) AS e, id(img) AS img
            """
            dep_rows = s.run(q_dep).data()

        for r in dep_rows:
            e_id, i_id = r["e"], r["img"]
            if e_id not in nodes or i_id not in nodes:
                continue
            key = tuple(sorted((e_id, i_id)))
            # Strong weight: Images define the cluster topic significantly
            edges[key] = edges.get(key, 0.0) + 1.2

        # -----------------------------
        # 4) CO-OCCURRENCE EDGES (Text)
        # -----------------------------
        if user_id:
            q_co = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(c:Chunk)
            MATCH (c)<-[:MENTIONED_IN]-(a:Entity)
            MATCH (c)<-[:MENTIONED_IN]-(b:Entity)
            WHERE id(a) < id(b)
            RETURN id(a) AS a, id(b) AS b, count(*) AS co
            """
            co_rows = s.run(q_co, uid=user_id).data()
        else:
            q_co = """
            MATCH (c:Chunk)
            MATCH (c)<-[:MENTIONED_IN]-(a:Entity)
            MATCH (c)<-[:MENTIONED_IN]-(b:Entity)
            WHERE id(a) < id(b)
            RETURN id(a) AS a, id(b) AS b, count(*) AS co
            """
            co_rows = s.run(q_co).data()

        for r in co_rows:
            if r["co"] < min_cooccurrence:
                continue
            a, b = r["a"], r["b"]
            if a not in nodes or b not in nodes:
                continue
            key = tuple(sorted((a, b)))
            # smaller weight; scaled by co-occurrence count
            edges[key] = edges.get(key, 0.0) + float(r["co"]) * 0.2

    node_list = [(nid, name) for nid, name in nodes.items()]
    edge_list = [(a, b, w) for (a, b), w in edges.items()]

    log.info("Export: %d weighted edges", len(edge_list))
    return node_list, edge_list


# ---------------------------------------------------------------------------
# LEIDEN CLUSTERING
# ---------------------------------------------------------------------------
def run_leiden(
    resolution: float | None = None,
    user_id: str | None = None,
) -> int:
    """
    Run Leiden clustering.
    1. Clears OLD user-specific community links.
    2. Calculates NEW communities.
    3. Deletes EMPTY community nodes (garbage collection).
    """
    resolution = resolution or _cfg.leiden_resolution or 0.5

    # 1. Export Graph
    nodes, edges = _export_entities_and_edges(user_id=user_id)
    if not nodes or not edges:
        log.warning("Leiden: nothing to cluster.")
        return 0

    # 2. Prepare Graph
    id2idx = {neo_id: idx for idx, (neo_id, _) in enumerate(nodes)}
    g = ig.Graph()
    g.add_vertices(len(nodes))
    weights = [float(w) for _, _, w in edges]
    for a, b, _ in edges:
        g.add_edge(id2idx[a], id2idx[b])
    
    if g.ecount():
        g.es["weight"] = weights

    # 3. Run Algorithm
    try:
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=g.es["weight"] if g.ecount() else None,
            resolution_parameter=float(resolution),
        )
        membership = partition.membership
    except Exception as e:
        log.error("Leiden clustering failed: %s", e)
        return 0

    idx2id = {idx: neo_id for neo_id, idx in id2idx.items()}

    # =========================================================
    # 4. WRITE BACK (The Safe Way)
    # =========================================================
    with _driver.session() as s:
        tx = s.begin_transaction()
        try:
            # A) Clear OLD 'IN_COMMUNITY' relationships for these specific nodes only
            #    This prevents nodes from belonging to old + new communities simultaneously.
            node_ids = list(id2idx.keys())
            tx.run(
                """
                UNWIND $ids as nid
                MATCH (n) WHERE id(n) = nid
                MATCH (n)-[r:IN_COMMUNITY]->(:Community)
                DELETE r
                """,
                ids=node_ids
            )

            # B) Write NEW relationships
            for idx, comm in enumerate(membership):
                neo_id = idx2id.get(idx)
                if neo_id is None: continue
                
                tx.run(
                    """
                    MATCH (n) WHERE id(n)=$id
                    SET n.community = $c
                    WITH n
                    MERGE (com:Community {id:$c})
                    SET com.last_updated = timestamp()
                    MERGE (n)-[:IN_COMMUNITY]->(com)
                    """,
                    id=int(neo_id),
                    c=int(comm),
                )
            
            # C) Garbage Collection: Delete Communities that are now empty
            #    (i.e., No nodes point to them anymore)
            tx.run(
                """
                MATCH (c:Community)
                WHERE NOT (c)<-[:IN_COMMUNITY]-()
                DETACH DELETE c
                """
            )
            
            tx.commit()
        except Exception as e:
            log.error("Error writing communities: %s", e)
            tx.rollback()
            return 0

    n_comms = len(set(membership))
    log.info("Leiden complete. %d communities. Empty clusters removed.", n_comms)
    return n_comms


# ---------------------------------------------------------------------------
# COMMUNITY SUMMARIES (TEXT + IMAGE SUPPORT)
# ---------------------------------------------------------------------------
def summarize_communities(
    force_refresh: bool = False,
    max_items: int = 150,
    user_id: str | None = None,
) -> List[Tuple[int, str]]:
    """
    Summarize communities with Gemini.
    """
    log.info("Generating community summaries (force_refresh=%s, user=%s)", force_refresh, user_id)

    with _driver.session() as s:
        if user_id:
            # Match community via Entities OR Images linked to User
            q_list = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(src)
            MATCH (src)-[:MENTIONED_IN|DEPICTS]-(node)
            MATCH (node)-[:IN_COMMUNITY]->(com:Community)
            RETURN DISTINCT com.id AS id, com.summary AS summary
            ORDER BY com.id
            """
            rows = s.run(q_list, uid=user_id).data()
        else:
            rows = s.run(
                "MATCH (c:Community) RETURN c.id AS id, c.summary AS summary ORDER BY c.id"
            ).data()

    if not rows:
        return []

    outputs: List[Tuple[int, str]] = []

    for row in rows:
        comm_id = row["id"]
        existing = row.get("summary")
        if existing and not force_refresh:
            outputs.append((int(comm_id), existing))
            continue

        # ------------------------------------------------------------------
        # 1) Relation evidence (inside this community)
        # ------------------------------------------------------------------
        if user_id:
            q_rels = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(src)
            MATCH (src)-[:MENTIONED_IN|DEPICTS]-(a)
            MATCH (a)-[r:RELATION|RELATED]->(b)
            WHERE (a)-[:IN_COMMUNITY]->(:Community {id:$cid})
              AND (b)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN DISTINCT a.name AS src, type(r) AS rel, b.name AS tgt
            LIMIT $limit
            """
            with _driver.session() as s:
                rel_data = s.run(
                    q_rels, uid=user_id, cid=int(comm_id), limit=max_items
                ).data()
        else:
            q_rels = """
            MATCH (a)-[r:RELATION|RELATED]->(b)
            WHERE (a)-[:IN_COMMUNITY]->(:Community {id:$cid})
              AND (b)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN DISTINCT a.name AS src, type(r) AS rel, b.name AS tgt
            LIMIT $limit
            """
            with _driver.session() as s:
                rel_data = s.run(q_rels, cid=int(comm_id), limit=max_items).data()

        # ------------------------------------------------------------------
        # 2) Entity evidence (Text descriptions)
        # ------------------------------------------------------------------
        if user_id:
            q_ents = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(src)
            MATCH (src)-[:MENTIONED_IN|DEPICTS]-(e:Entity)
            WHERE (e)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN DISTINCT e.name AS name, e.description AS desc
            LIMIT $limit
            """
            with _driver.session() as s:
                ent_data = s.run(
                    q_ents, uid=user_id, cid=int(comm_id), limit=max_items
                ).data()
        else:
            q_ents = """
            MATCH (e:Entity)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN DISTINCT e.name AS name, e.description AS desc
            LIMIT $limit
            """
            with _driver.session() as s:
                ent_data = s.run(q_ents, cid=int(comm_id), limit=max_items).data()

        # ------------------------------------------------------------------
        # 3) Image summaries (Visual evidence)
        # ------------------------------------------------------------------
        if user_id:
            q_imgs = """
            MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(img:Image)
            WHERE (img)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN DISTINCT coalesce(img.summary, img.user_context) AS summary
            LIMIT $limit
            """
            with _driver.session() as s:
                img_data = s.run(
                    q_imgs, uid=user_id, cid=int(comm_id), limit=max_items
                ).data()
        else:
            q_imgs = """
            MATCH (img:Image)-[:IN_COMMUNITY]->(:Community {id:$cid})
            RETURN DISTINCT coalesce(img.summary, img.user_context) AS summary
            LIMIT $limit
            """
            with _driver.session() as s:
                img_data = s.run(q_imgs, cid=int(comm_id), limit=max_items).data()

        # ------------------------------------------------------------------
        # 4) Build prompt for Gemini (Dynamic Visuals)
        # ------------------------------------------------------------------
        rel_lines = [
            f"- {r['src']} --{r['rel']}--> {r['tgt']}" for r in (rel_data or [])
        ]
        ent_lines = [
            f"- {e['name']}: {e.get('desc') or ''}".strip()
            for e in (ent_data or [])
        ]
        img_lines = [
            f"- [Visual Content] {i.get('summary')}"
            for i in (img_data or [])
            if i.get("summary")
        ]

        if not rel_lines and not ent_lines and not img_lines:
            log.info("Community %s has no evidence; skipping.", comm_id)
            continue
        
        # Base Prompt
        prompt_parts = [
            "You are a domain expert summarizing a *personal multimodal knowledge graph* community.",
            "",
            f"Community ID: {comm_id}",
            "",
            "1) Key semantic relations between concepts:",
            *(rel_lines or ["(no explicit relations captured)"]),
            "",
            "2) Important entities in this community:",
            *(ent_lines or ["(no entity descriptions found)"]),
        ]
        
        # Conditionally add visual context
        visual_instructions = ""
        if img_lines:
            prompt_parts.extend([
                "",
                "3) Integrated Visual Context (Images grouped in this cluster):",
                *img_lines,
            ])
            visual_instructions = "- And how the visual details (from diagrams/photos) reinforce these concepts."
        else:
            visual_instructions = "(No visual context is available; focus solely on the text relationships. Do NOT mention the lack of images.)"

        prompt_parts.extend([
            "",
            "Write a concise paragraph (4-6 sentences) explaining:",
            "- The central topic of this community,",
            "- How the entities and concepts relate,",
            visual_instructions
        ])

        prompt = "\n".join(prompt_parts)

        summary = gemini_complete(prompt, max_tokens=400)

        # Persist summary
        with _driver.session() as s:
            s.run(
                """
                MERGE (c:Community {id:$cid})
                SET c.summary = $summary,
                    c.last_summarized = timestamp()
                """,
                cid=int(comm_id),
                summary=summary,
            )

        outputs.append((int(comm_id), summary))

    log.info("Community summarization done for %d communities.", len(outputs))
    return outputs