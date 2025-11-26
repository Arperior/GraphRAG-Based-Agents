# pipeline/memory.py
from __future__ import annotations
from typing import List, Dict, Optional
import logging
import time
import re

from pipeline.neo4j_client import _driver
from pipeline.utils import truncate

log = logging.getLogger("memory")


def _norm_user(user_id: str) -> str:
    return user_id.strip()


def ensure_user_exists(user_id: str):
    """Ensures user exists (keeps original casing)"""
    user_id = _norm_user(user_id)
    with _driver.session() as s:
        s.run(
            "MERGE (u:User {id:$id}) SET u.last_seen=timestamp()",
            id=user_id
        )


def store_query_and_answer(user_id: str, query: str, answer: str, tags: List[str] | None = None):
    """Store a Query and Answer linked to the user."""
    if not query or not query.strip():
        log.warning("Skipping empty query in memory store.")
        return

    user_id = _norm_user(user_id)
    ts = int(time.time() * 1000)

    try:
        with _driver.session() as s:
            s.run(
                """
                MATCH (u:User {id:$uid})
                CREATE (q:Query {text:$q, created:$ts})
                CREATE (a:Answer {text:$a, created:$ts})
                MERGE (u)-[:ASKED]->(q)
                MERGE (q)-[:ANSWERED_WITH]->(a)
                """,
                uid=user_id,
                q=query.strip(),
                a=answer.strip(),
                ts=ts,
            )
        log.info("Stored query+answer in user memory.")
    except Exception as e:
        log.error(f"Failed to store query+answer: {e}", exc_info=True)


def record_user_chunk_interest(user_id: str, chunk_id: str, query: str, tokens: List[str] | None = None):
    """
    Record that a user interacted with a chunk (non-redundant memory).
    Stores on relationship:
      - last_query: latest full query text
      - tokens: list of short tokens (for quick matching)
      - count: interaction count (incremental)
      - last_seen: timestamp
    """
    if not user_id or not chunk_id:
        return

    ensure_user_exists(user_id)

    # sanitize tokens: short alpha tokens only
    clean_tokens = [t.lower() for t in (tokens or []) if t and re.match(r"^\w{2,}$", t)]
    # dedupe while keeping order
    seen = set()
    clean_tokens = [x for x in clean_tokens if not (x in seen or seen.add(x))]
    try:
        with _driver.session() as s:
            s.run(
                """
                MATCH (u:User {id:$uid})
                MATCH (c:Chunk {id:$cid})
                MERGE (u)-[r:INTERESTED_IN]->(c)
                ON CREATE SET r.count = 1,
                              r.tokens = $tokens,
                              r.last_query = $q,
                              r.last_seen = timestamp()
                ON MATCH SET  r.count = coalesce(r.count,0) + 1,
                              r.tokens = apoc.coll.toSet(coalesce(r.tokens, []) + $tokens),
                              r.last_query = $q,
                              r.last_seen = timestamp()
                """,
                uid=user_id,
                cid=chunk_id,
                tokens=clean_tokens,
                q=query.strip()
            )
        log.info(f"Recorded interest (user={user_id}, chunk={chunk_id}) tokens={clean_tokens}")
    except Exception as e:
        log.error(f"Failed to record user-chunk interest: {e}", exc_info=True)


def get_user_chunk_interest_count(user_id: str, chunk_id: str) -> int:
    """Return count for user->chunk interest relationship if exists."""
    try:
        with _driver.session() as s:
            row = s.run(
                "MATCH (u:User {id:$uid})-[r:INTERESTED_IN]->(c:Chunk {id:$cid}) "
                "RETURN coalesce(r.count,0) AS c",
                uid=_norm_user(user_id),
                cid=chunk_id
            ).single()
        return int(row["c"]) if row and row["c"] is not None else 0
    except Exception as e:
        log.warning(f"Failed to read user-chunk interest count: {e}")
        return 0


def get_user_interest_count(user_id: str, entity_name: str) -> int:
    """Legacy: entity-level interest (keeps backward compatibility)."""
    try:
        with _driver.session() as s:
            row = s.run(
                """
                MATCH (u:User {id:$uid})-[r:INTERESTED_IN]->(e:Entity {name:$ename})
                RETURN coalesce(r.count, 0) AS c
                """,
                uid=_norm_user(user_id),
                ename=entity_name
            ).single()
            return int(row["c"]) if row and row["c"] is not None else 0
    except Exception as e:
        log.warning(f"Failed to read interest count: {e}")
        return 0


def get_user_interests(user_id: str, limit: int = 10) -> List[Dict]:
    """Return list of chunks the user is interested in (chunk-level)."""
    try:
        with _driver.session() as s:
            rows = s.run(
                """
                MATCH (u:User {id:$uid})-[r:INTERESTED_IN]->(c:Chunk)
                RETURN c.id AS chunk_id, r.count AS score, r.tokens AS tokens, r.last_query AS last_query
                ORDER BY score DESC
                LIMIT $limit
                """,
                uid=_norm_user(user_id),
                limit=limit
            ).data()
        return rows
    except Exception as e:
        log.warning(f"Failed to fetch user interests: {e}")
        return []


def get_user_longterm_memory_text(user_id: str, limit: int = 10) -> str:
    """Return a formatted summary of recent Q/A memory plus top interests."""
    with _driver.session() as s:
        rows = s.run(
            """
            MATCH (u:User {id:$uid})-[:ASKED]->(q:Query)-[:ANSWERED_WITH]->(a:Answer)
            RETURN q.text AS q, a.text AS a, q.created AS t
            ORDER BY q.created DESC
            LIMIT $limit
            """,
            uid=_norm_user(user_id),
            limit=limit,
        ).data()

    lines = []
    if rows:
        for r in rows:
            q = r["q"]
            a = truncate(r["a"], 300)
            lines.append(f"Q: {q}\nA: {a}\n---")

    interests = get_user_interests(user_id, limit=5)
    if interests:
        names = [i["chunk_id"] for i in interests]
        lines.append("User Chunks of Interest: " + ", ".join(names))

    return "\n".join(lines)


def get_user_recent_queries(user_id: str, limit: int = 10) -> List[Dict]:
    with _driver.session() as s:
        rows = s.run(
            """
            MATCH (u:User {id:$uid})-[:ASKED]->(q:Query)
            RETURN q.text AS q, q.created AS t
            ORDER BY q.created DESC
            LIMIT $limit
            """,
            uid=_norm_user(user_id),
            limit=limit,
        ).data()
    return rows


def list_user_memories(user_id: str, limit: int = 50) -> List[Dict]:
    with _driver.session() as s:
        rows = s.run(
            """
            MATCH (u:User {id:$uid})-[:ASKED]->(q:Query)-[:ANSWERED_WITH]->(a:Answer)
            RETURN q.text AS value, 'qa' AS type, q.created AS created
            ORDER BY q.created DESC
            LIMIT $limit
            """,
            uid=_norm_user(user_id),
            limit=limit,
        ).data()
    return rows


def list_users(limit: int = 100) -> List[str]:
    try:
        with _driver.session() as s:
            rows = s.run(
                """
                MATCH (u:User)
                RETURN u.id AS id
                ORDER BY u.last_seen DESC
                LIMIT $limit
                """,
                limit=limit
            ).data()
        return [r["id"] for r in rows]
    except Exception as e:
        log.warning(f"Failed to list users: {e}")
        return []


def clear_user_memory(user_id: str):
    user_id = _norm_user(user_id)

    with _driver.session() as s:
        s.run(
            """
            MATCH (u:User {id:$uid})-[:ASKED]->(q:Query)-[:ANSWERED_WITH]->(a:Answer)
            DETACH DELETE q, a
            """,
            uid=user_id
        )
    log.info("Cleared user memory.")


# ============================================================
# NEW: ENTITY MATCH SUCCESS (STRICT PERSONAL + ENRICHED NEIGHBOURS)
# ============================================================
def record_entity_success(user_id: str, entity_name: str, query: str, tokens: List[str]):
    """
    Positive reinforcement when an entity from the *user's personal subgraph*
    (including enriched neighbours) is successfully used in retrieval.

    Only records success if:
      User -[:INTERESTED_IN]-> Source
      Source -[:MENTIONED_IN|DEPICTS]-> RootEntity
      RootEntity -[:RELATION|RELATED*0..2]-> (Entity {name})
    """
    if not user_id or not entity_name:
        return

    ensure_user_exists(user_id)

    clean_tokens = [t.lower() for t in (tokens or []) if t and re.match(r"^\w{2,}$", t)]
    seen = set()
    clean_tokens = [x for x in clean_tokens if not (x in seen or seen.add(x))]

    try:
        with _driver.session() as s:
            s.run(
                """
                MATCH (u:User {id:$uid})
                MATCH (u)-[:INTERESTED_IN]->(src)
                MATCH (src)-[:MENTIONED_IN|DEPICTS]->(root:Entity)
                MATCH (root)-[:RELATION|RELATED*0..2]-(e:Entity {name:$ename})
                WITH DISTINCT u, e
                MERGE (u)-[r:MATCHED]->(e)
                ON CREATE SET r.count = 1,
                              r.tokens = $tokens,
                              r.last_query = $q,
                              r.last_seen = timestamp()
                ON MATCH SET  r.count = coalesce(r.count,0) + 1,
                              r.tokens = apoc.coll.toSet(coalesce(r.tokens, []) + $tokens),
                              r.last_query = $q,
                              r.last_seen = timestamp()
                """,
                uid=_norm_user(user_id),
                ename=entity_name,
                tokens=clean_tokens,
                q=query.strip()
            )
        log.info(f"[Entity Success] user={user_id}, entity={entity_name}, tokens={clean_tokens}")
    except Exception as e:
        log.warning(f"Failed to record entity success: {e}")
