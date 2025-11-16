from __future__ import annotations
from typing import Dict, List
import logging
import spacy

from config.config import load_config
from pipeline.utils import read_text, dedup_keep_order
from pipeline.llm_client_local import generate_json

_cfg = load_config()
log = logging.getLogger("entity_extraction")

# Load spaCy for candidate entity seeding
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
            log.info(f"Extracted {len(data)} entities (no relations).")
            return {"entities": data, "relations": []}

        # If model returns structured dict with entities/relations
        elif isinstance(data, dict):
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            log.info(f"Extracted {len(entities)} entities and {len(relations)} relations.")
            return {"entities": entities, "relations": relations}

        else:
            log.warning(f"Unexpected entity extraction output type: {type(data)}")
            return {"entities": [], "relations": []}

    except Exception as e:
        log.error(f"Entity extraction failed: {e}")
        return {"entities": [], "relations": []}
