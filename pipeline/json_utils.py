# pipeline/json_utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import re
import ast

# Non-greedy: match smallest {...} or [...] blocks
_JSON_BLOCK_RE = re.compile(r'(\[[\s\S]*?\]|\{[\s\S]*?\})', re.MULTILINE)


def _find_json_block(text: str) -> Optional[str]:
    """Return the first JSON-looking block in the text, or None."""
    if not isinstance(text, str):
        return None
    m = _JSON_BLOCK_RE.search(text)
    return m.group(1) if m else None


def _fix_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] (very common LLM mistake)."""
    return re.sub(r",\s*([\]\}])", r"\1", s)


def parse_llm_json_list(text: str) -> List[Dict[str, Any]]:
    """
    Parse model output that *should* represent a JSON array of objects.

    Handles:
      - Proper JSON list:       [ {...}, {...} ]
      - Single dict:            { ... }
      - Many dicts w/out '[]':  { ... }, { ... }, { ... }
      - JSON embedded in extra text
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # -------- 1) direct json.loads on the whole text --------
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    # Helper: safe decode a single JSON-ish string
    def _decode_one(raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        raw = _fix_trailing_commas(raw)
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                obj = ast.literal_eval(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    results: List[Dict[str, Any]] = []

    # -------- 2) MANY { ... } OBJECTS WITHOUT [] --------
    # e.g.
    #   { ... },
    #   { ... },
    #   { ... }
    object_blocks = re.findall(r"\{[\s\S]*?\}", text)
    if len(object_blocks) > 1:
        for block in object_blocks:
            obj = _decode_one(block)
            if obj is not None:
                results.append(obj)
        if results:
            return results

    # -------- 3) First JSON block (could be [ ... ] or { ... }) --------
    block = _find_json_block(text)
    if block:
        block = block.strip()
        # If it's not an array already, wrap in []
        candidate = block
        if candidate.startswith("{") and not candidate.strip().startswith("["):
            candidate = "[" + candidate + "]"
        candidate = _fix_trailing_commas(candidate)

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

    # -------- 4) nothing worked --------
    return results


def _split_items_into_entities_relations(
    items: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Heuristic splitter:
    - if an item has 'source' and 'target' → treat as relation
    - if it has 'name' → treat as entity
    - else → default to entity
    """
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        if "source" in item and "target" in item:
            relations.append(item)
        elif "name" in item:
            entities.append(item)
        else:
            entities.append(item)

    return {"entities": entities, "relations": relations}


def parse_llm_graph(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse model output that should represent a graph.

    Handles:
      - proper object: { "entities": [...], "relations": [...] }
      - list of entity or relation objects: [ {...}, {...} ]
      - many bare objects: { ... }, { ... }, ...
      - JSON embedded within other text

    Always returns:
      { "entities": [...], "relations": [...] }
    """
    result: Dict[str, List[Dict[str, Any]]] = {
        "entities": [],
        "relations": [],
    }

    if not isinstance(text, str) or not text.strip():
        return result

    def _try_decode(raw: str) -> Any:
        raw = raw.strip()
        raw = _fix_trailing_commas(raw)
        try:
            return json.loads(raw)
        except Exception:
            try:
                return ast.literal_eval(raw)
            except Exception:
                return None

    # ---- 1) direct decode ----
    parsed = _try_decode(text)
    if isinstance(parsed, dict):
        ents = parsed.get("entities")
        rels = parsed.get("relations")
        if isinstance(ents, list) or isinstance(rels, list):
            ent_list = ents if isinstance(ents, list) else []
            rel_list = rels if isinstance(rels, list) else []
            return {
                "entities": [e for e in ent_list if isinstance(e, dict)],
                "relations": [r for r in rel_list if isinstance(r, dict)],
            }
        return _split_items_into_entities_relations([parsed])

    if isinstance(parsed, list):
        return _split_items_into_entities_relations(
            [x for x in parsed if isinstance(x, dict)]
        )

    # ---- 2) try via parse_llm_json_list (many objects / embedded) ----
    items = parse_llm_json_list(text)
    if items:
        return _split_items_into_entities_relations(items)

    # ---- 3) last resort ----
    return result
