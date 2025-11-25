from __future__ import annotations
from typing import Dict, List
import logging
import spacy

from config.config import load_config
from pipeline.utils import read_text, dedup_keep_order
from pipeline.llm_client_local import generate_json

_cfg = load_config()
log = logging.getLogger("entity_extraction")

# Load spaCy for candidate entity seeding (optional)
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None


def spacy_candidates(text: str) -> List[str]:
    """Use spaCy to detect possible named entities or noun chunks as LLM seeds."""
    if not _nlp:
        return []
    ents = [e.text.strip() for e in _nlp(text).ents]
    if not ents:
        ents = [c.text.strip() for c in _nlp(text).noun_chunks]
    return dedup_keep_order([e for e in ents if e])


def _normalize_entities(raw_entities) -> List[Dict]:
    """
    Ensure entities are list of dicts with at least 'name'. Fill defaults.
    Accepts: list[str] or list[dict] or single dict.
    Uses dedup_keep_order from utils to deduplicate.
    """
    out = []
    if isinstance(raw_entities, dict):
        raw_entities = [raw_entities]
    if not raw_entities:
        return []

    for e in raw_entities:
        if isinstance(e, str):
            out.append({"name": e.strip(), "type": "UNKNOWN", "description": ""})
        elif isinstance(e, dict):
            name = e.get("name") or e.get("id") or e.get("label")
            if not name:
                continue
            out.append({
                "name": name.strip(),
                "type": e.get("type", "UNKNOWN"),
                "description": e.get("description", "") or e.get("desc", "")
            })

    # deduplicate via utils
    names = [e["name"] for e in out]
    keep = dedup_keep_order(names)
    final = []
    seen = set()
    for e in out:
        if e["name"] in keep and e["name"] not in seen:
            final.append(e)
            seen.add(e["name"])
    return final


def _normalize_relations(raw_relations) -> List[Dict]:
    """
    Normalize relations to ensure consistent structure.
    Each relation must contain: source, target, relation, evidence, confidence.
    """
    if isinstance(raw_relations, dict):
        raw_relations = [raw_relations]
    if not raw_relations:
        return []

    out = []
    for r in raw_relations:
        if isinstance(r, str):
            # no reliable structure, skip safely
            continue
        if not isinstance(r, dict):
            continue

        src = r.get("source") or r.get("src") or r.get("from")
        tgt = r.get("target") or r.get("tgt") or r.get("to")
        rel = r.get("relation") or r.get("rel") or r.get("type") or "RELATED_TO"

        if not src or not tgt:
            continue

        out.append({
            "source": str(src).strip(),
            "target": str(tgt).strip(),
            "relation": str(rel).strip(),
            "evidence": r.get("evidence", "") or r.get("text", ""),
            "confidence": float(r.get("confidence", 1.0) or 1.0)
        })

    return out


def extract_graph(chunk_text: str, entity_types: str = "PERSON,ORGANIZATION,GEO") -> Dict:
    """
    Extract entities and base relations from text using the local LLM.
    Expected model output: JSON with 'entities' and optional 'relations'.
    """
    tpl_path = _cfg.prompts_dir / "extract_graph.txt"
    tpl = read_text(tpl_path)

    # Use spaCy seeds to guide entity extraction
    seeds = spacy_candidates(chunk_text)
    seed_text = f"\n\nPay special attention to these possible entities: {', '.join(seeds)}" if seeds else ""

    # Fill the template prompt
    prompt = tpl.replace("{entity_types}", entity_types).replace("{input_text}", chunk_text + seed_text)

    try:
        log.info("Running entity and graph extraction LLM...")
        data = generate_json(prompt, max_tokens=768)

        # If model returns a list of entities
        if isinstance(data, list):
            entities = _normalize_entities(data)
            log.info(f"Extracted {len(entities)} entities (no relations).")
            return {"entities": entities, "relations": []}

        # If model returns structured dict with entities/relations
        elif isinstance(data, dict):
            entities = _normalize_entities(data.get("entities", []))
            relations = _normalize_relations(data.get("relations", []))
            log.info(f"Extracted {len(entities)} entities and {len(relations)} relations.")
            return {"entities": entities, "relations": relations}

        else:
            log.warning(f"Unexpected entity extraction output type: {type(data)}")
            return {"entities": [], "relations": []}

    except Exception as e:
        log.error(f"Entity extraction failed: {e}", exc_info=True)
        return {"entities": [], "relations": []}
