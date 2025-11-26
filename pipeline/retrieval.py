# pipeline/retrieval.py
"""
Retrieval module: broad search → re-ranking → k-hop evidence → synthesis
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import logging
import Levenshtein
import re

from config.config import load_config
from pipeline.neo4j_client import _driver, k_hop_chunks
from pipeline.utils import truncate
from pipeline.llm_client_gemini import gemini_complete
from pipeline.memory import (
    get_user_recent_queries,
    get_user_longterm_memory_text,
    record_user_chunk_interest,
    get_user_interest_count,
    get_user_interests,
    record_entity_success,
)

log = logging.getLogger("retrieval")
_cfg = load_config()


# ================================================================
# ENTITY SCORING — exact / prefix / substring + fuzzy fallback
# ================================================================
def _score_entity_name(query: str, name: str) -> float:
    """Heuristic string-matching score for entity names."""
    q = query.strip().lower()
    n = name.strip().lower()

    if not q or not n:
        return 0.0

    if q == n:
        return 1.0
    if n.startswith(q):
        return 0.9
    if q in n:
        score = 0.6 - (len(n) - len(q)) * 0.001
        return max(0.3, score)

    # fuzzy fallback
    try:
        ratio = Levenshtein.ratio(q, n)
    except Exception:
        ratio = 0.0
    if ratio > 0.85:
        return 0.8

    return 0.0


def _overlap_score(query: str, text: str) -> float:
    """
    Token overlap between query and text (used for descriptions).
    Returns 0.0–0.5 (balanced influence).
    """
    if not query or not text:
        return 0.0

    q_tokens = {t for t in re.split(r"\W+", query.lower()) if len(t) > 2}
    t_tokens = {t for t in re.split(r"\W+", text.lower()) if len(t) > 2}
    if not q_tokens or not t_tokens:
        return 0.0

    inter = q_tokens.intersection(t_tokens)
    if not inter:
        return 0.0

    base = len(inter) / max(len(q_tokens), 1)
    return min(0.5, 0.1 + 0.4 * base)


def _query_mentions_visual(q: str) -> bool:
    terms = ["image", "images", "picture", "diagram", "photo", "vision", "multimodal"]
    q_low = q.lower()
    return any(t in q_low for t in terms)


def _modality_bias(query: str, modality: str | None) -> float:
    """
    Small boost for visual / multimodal entities when query hints at that.
    """
    if not modality:
        return 0.0

    if _query_mentions_visual(query) and modality.lower() == "visual":
        return 0.25
    return 0.0


# ================================================================
# MEMORY BOOST — additive (more stable than multiplicative)
# ================================================================
def _boost_by_user_memory(entity_name: str, user_id: str | None) -> float:
    """Small additive boost using user query history and entity interest count."""
    if not user_id:
        return 0.0

    boost = 0.0
    name = entity_name

    try:
        recent = get_user_recent_queries(user_id, limit=20)
        recent_text = " ".join(r["q"] for r in recent) if recent else ""

        longterm = get_user_longterm_memory_text(user_id, limit=20) or ""

        if name.lower() in recent_text.lower():
            boost += 0.15
        if name.lower() in longterm.lower():
            boost += 0.25

        # legacy entity interest count (if present)
        try:
            cnt = get_user_interest_count(user_id, entity_name)
            if cnt > 0:
                boost += min(0.5, 0.08 * cnt)
        except Exception:
            pass

        return boost
    except Exception as e:
        log.warning(f"User memory boost failed: {e}")
        return 0.0


# ================================================================
# ENTITY RETRIEVAL (STRICT PERSONAL SCOPE + ENRICHED NEIGHBOURS)
# ================================================================
def find_entities(query: str, top_k: int | None = None, user_id: str | None = None) -> List[Dict]:
    """
    Entity retrieval with strict personal scope:
      - Only entities reachable from the current user's subgraph:
          (User)-[:INTERESTED_IN]->(src)
          (src)-[:MENTIONED_IN|DEPICTS]->(root:Entity)
          root-[:RELATION|RELATED*0..2]-(e:Entity)   (enriched neighbours)
      - Falls back to global search ONLY if user_id is None.
    """
    top_k = top_k or _cfg.retrieval_search_limit

    if not query or len(query.strip()) < 2:
        return []

    q_clean = query.strip()
    tokens = [t.lower() for t in re.split(r"\W+", q_clean) if t]

    # --------------------------------------------------------
    # Step 1: Strict personal-scope retrieval if user_id given
    # --------------------------------------------------------
    candidates: List[Dict] = []

    if user_id:
        if tokens:
            where_clauses = []
            for i in range(len(tokens)):
                where_clauses.append(
                    f"(toLower(e.name) CONTAINS toLower($t{i}) "
                    f"OR toLower(coalesce(e.description,'')) CONTAINS toLower($t{i}))"
                )
            cypher = (
                "MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(src)\n"
                "MATCH (src)-[:MENTIONED_IN|DEPICTS]->(root:Entity)\n"
                "OPTIONAL MATCH (root)-[:RELATION|RELATED*0..2]-(e:Entity)\n"
                "WITH DISTINCT e\n"
                "WHERE e IS NOT NULL AND (" + " OR ".join(where_clauses) + ")\n"
                "RETURN e.name AS name,\n"
                "       e.community AS community,\n"
                "       e.description AS description,\n"
                "       e.modality AS modality\n"
                "LIMIT 200"
            )
            params = {f"t{i}": tokens[i] for i in range(len(tokens))}
            params["uid"] = user_id
        else:
            cypher = """
                MATCH (u:User {id:$uid})-[:INTERESTED_IN]->(src)
                MATCH (src)-[:MENTIONED_IN|DEPICTS]->(root:Entity)
                OPTIONAL MATCH (root)-[:RELATION|RELATED*0..2]-(e:Entity)
                WITH DISTINCT e
                WHERE e IS NOT NULL
                RETURN e.name AS name,
                       e.community AS community,
                       e.description AS description,
                       e.modality AS modality
                LIMIT 200
            """
            params = {"uid": user_id}

        with _driver.session() as s:
            rows = s.run(cypher, **params)
            candidates = [r.data() for r in rows]

        log.info(
            f"[find_entities] (personal) query='{q_clean}' tokens={tokens} "
            f"candidates_found={len(candidates)}"
        )

    # --------------------------------------------------------
    # Step 1b: Global fallback only if no user_id
    # --------------------------------------------------------
    if not user_id:
        if tokens:
            where_clauses = []
            for i in range(len(tokens)):
                where_clauses.append(
                    f"(toLower(e.name) CONTAINS toLower($t{i}) "
                    f"OR toLower(coalesce(e.description,'')) CONTAINS toLower($t{i}))"
                )
            cypher = (
                "MATCH (e:Entity)\n"
                "WHERE " + " OR ".join(where_clauses) + "\n"
                "RETURN e.name AS name,\n"
                "       e.community AS community,\n"
                "       e.description AS description,\n"
                "       e.modality AS modality\n"
                "LIMIT 200"
            )
            params = {f"t{i}": tokens[i] for i in range(len(tokens))}
        else:
            cypher = """
                MATCH (e:Entity)
                RETURN e.name AS name,
                       e.community AS community,
                       e.description AS description,
                       e.modality AS modality
                LIMIT 200
            """
            params = {}

        with _driver.session() as s:
            rows = s.run(cypher, **params)
            candidates = [r.data() for r in rows]

        log.info(
            f"[find_entities] (global) query='{q_clean}' tokens={tokens} "
            f"candidates_found={len(candidates)}"
        )

    # --------------------------------------------------------
    # Step 2: Score + memory + semantic overlap
    # --------------------------------------------------------
    scored: List[Dict] = []
    visual_ok = _query_mentions_visual(q_clean)

    for c in candidates:
        name = c.get("name", "")
        desc = c.get("description") or ""
        modality = c.get("modality")

        base_score = _score_entity_name(q_clean, name)
        desc_score = _overlap_score(q_clean, desc)
        mem_boost = _boost_by_user_memory(name, user_id)
        mod_boost = _modality_bias(q_clean, modality)

        total = base_score + desc_score + mem_boost + mod_boost

        # De-emphasize pure visual fluff for non-visual queries
        if modality == "visual" and not visual_ok:
            total -= 0.2

        # Require some minimum score (balanced mode)
        if total <= 0.15:
            continue

        scored.append(
            {
                **c,
                "score": total,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ================================================================
# K-HOP EVIDENCE
# ================================================================
def get_k_hop_evidence(
    entity_name: str,
    k_hop: int = 1,
    per_entity: int | None = None,
    user_id: str | None = None
) -> Tuple[List[Dict], List[str]]:
    """
    Return:
      - raw node dicts from k_hop_chunks
      - formatted evidence strings tagged by type (TEXT/IMAGE/ENTITY)
    """
    per_entity = per_entity or 3

    chunks = k_hop_chunks(entity_name, k=k_hop, limit=_cfg.neo4j_query_limit,user_id=user_id)

    evidences: List[str] = []
    for c in chunks[:per_entity]:
        text = c.get("text", "") or ""
        cid = c.get("cid")
        ctype = c.get("type", "Chunk")
        label = "TEXT"
        if ctype == "Image":
            label = "IMAGE"
        elif ctype == "Entity":
            label = "ENTITY"

        excerpt = truncate(text, 800)
        evidences.append(f"[{label} {cid}] {excerpt}")

    return chunks, evidences


# ================================================================
# helper: simple token extraction from query
# ================================================================
def _tokenize_query(q: str, max_tokens: int = 8) -> List[str]:
    parts = re.split(r"\W+", q or "")
    parts = [p.lower() for p in parts if p and len(p) > 2]
    return parts[:max_tokens]


# ================================================================
# helper: fetch chunk text by id (used for fallback)
# ================================================================
def _fetch_chunk_text(chunk_id: str) -> str | None:
    try:
        with _driver.session() as s:
            row = s.run("MATCH (c:Chunk {id:$cid}) RETURN c.text AS text", cid=chunk_id).single()
            if row:
                return row.get("text")
    except Exception as e:
        log.warning(f"Failed to fetch chunk text for {chunk_id}: {e}")
    return None


# ================================================================
# TOP-LEVEL EVIDENCE COLLECTION
# ================================================================
def gather_evidence_for_query(
    query: str,
    k_hop: int = 1,
    per_entity: int | None = None,
    top_entities: int | None = None,
    user_id: str | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Find entities → for each, collect k-hop evidence.
    Records:
      - user->entity MATCHED (success) for all selected entities
      - user->chunk interest for TEXT chunks only
    """
    per_entity = per_entity or 3
    top_entities = top_entities or 5

    ents = find_entities(query, top_k=top_entities, user_id=user_id)
    if not ents:
        # Try fallback: use user chunk interests as evidence
        all_evidence = []
        if user_id:
            interests = get_user_interests(user_id, limit=5)
            if interests:
                all_evidence.append("=== Evidence: user chunks of interest ===")
                for i in interests:
                    cid = i.get("chunk_id")
                    text = _fetch_chunk_text(cid)
                    if text:
                        excerpt = truncate(text, 800)
                        all_evidence.append(f"[TEXT {cid}] {excerpt}")
                if all_evidence:
                    return [], all_evidence
        return [], []

    entity_names = [e["name"] for e in ents]

    all_evidence: List[str] = []
    q_tokens = _tokenize_query(query)

    # For each entity, collect k-hop nodes; record interest for text chunks
    for en in entity_names:
        chunks, evidences = get_k_hop_evidence(en, k_hop=k_hop, per_entity=per_entity,user_id=user_id)
        if not chunks:
            continue

        all_evidence.append(f"=== Evidence: {en} ===")
        all_evidence.extend(evidences)

        # NEW: record entity match success (strict personal subgraph)
        if user_id:
            try:
                record_entity_success(user_id, en, query, q_tokens)
            except Exception as e:
                log.warning(f"Failed to record entity success for {en}: {e}")

        # record user -> chunk interest (only for TEXT chunks)
        if user_id:
            for c in chunks[:per_entity]:
                if c.get("type") != "Chunk":
                    continue
                cid = c.get("cid")
                if cid:
                    try:
                        record_user_chunk_interest(user_id, cid, query, q_tokens)
                    except Exception as e:
                        log.warning(f"Failed to record user-chunk interest for {cid}: {e}")

    return entity_names, all_evidence


# ================================================================
# GEMINI SYNTHESIS
# ================================================================
def synthesize_answer(
    query: str,
    evidence: List[str],
    user_id: str | None = None,
    chat_history: List[Tuple[str, str]] | None = None,
    max_tokens: int | None = None,
    use_plan: bool = False,
) -> str:
    """Build prompt → call Gemini → return answer."""
    max_tokens = max_tokens or _cfg.gemini.max_output_tokens

    evidence_block = "\n".join(evidence) or "(no evidence found)"

    # Load search prompt
    prompt_path = _cfg.prompts_dir / "basic_search_system_prompt.txt"
    try:
        template = prompt_path.read_text(encoding="utf-8")
    except Exception:
        template = (
            "Answer the user using ONLY this evidence:\n\n"
            "EVIDENCE:\n{evidence}\n\nQUESTION:\n{question}"
        )

    longterm = get_user_longterm_memory_text(user_id, limit=10) or ""
    working = ""
    if chat_history:
        turns = chat_history[-12:]
        working = "\n".join(f"{r}:{t}" for r, t in turns)

    prompt = (
        template.replace("{question}", query)
        .replace("{evidence}", evidence_block)
        .replace("{user_memory}", longterm)
        .replace("{working_memory}", working)
    )

    # Optional Plan Mode
    if use_plan:
        plan_path = _cfg.prompts_dir / "conversation_plan_prompt.txt"
        try:
            plan_template = plan_path.read_text(encoding="utf-8")
            plan_prompt = (
                plan_template.replace("{question}", query)
                .replace("{evidence}", evidence_block)
                .replace("{user_memory}", longterm)
                .replace("{working_memory}", working)
            )
            return gemini_complete(plan_prompt, max_tokens=max_tokens)
        except Exception as e:
            log.warning(f"Plan mode failed, falling back: {e}")

    # Basic synthesis
    return gemini_complete(prompt, max_tokens=max_tokens)
