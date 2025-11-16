from __future__ import annotations
import logging
from typing import List, Dict
from config.config import load_config
from pipeline.utils import read_text
from pipeline.llm_client_local import generate_json

_cfg = load_config()
log = logging.getLogger("relation_extractor")


def normalize_relation_name(name: str) -> str:
    """
    Normalize relation names into uppercase snake-case for Neo4j.
    Example:
        "prime minister" → "PRIME_MINISTER_OF"
        "capital_of" → "CAPITAL_OF"
    """
    if not name:
        return "RELATED_TO"

    # clean up whitespace, special chars
    name = name.strip().replace(" ", "_").replace("-", "_").lower()

    # ensure suffix consistency
    if not name.endswith("_of") and not name.endswith("_to"):
        name = f"{name}_of"

    return name.upper()


def extract_relations(chunk_text: str) -> List[Dict]:
    """
    Extract relationships between entities using the local LLM.
    Expected model output: JSON list of {source, target, relation, evidence, confidence}.
    """
    tpl_path = _cfg.prompts_dir / "extract_relations.txt"
    tpl = read_text(tpl_path)

    prompt = tpl.replace("{input_text}", chunk_text)

    try:
        log.info("Running relation extraction LLM...")
        data = generate_json(prompt, max_tokens=512)
    except Exception as e:
        log.error(f"Relation extraction failed: {e}")
        return []

    # Handle possible return types (dict, list, str)
    if isinstance(data, dict) and "raw" in data:
        log.warning("Received dict with raw key (legacy mode), skipping parse")
        return []
    elif isinstance(data, list):
        # Normalize relation types for Neo4j
        for rel in data:
            if "relation" in rel:
                rel["relation"] = normalize_relation_name(rel["relation"])
        log.info(f"Extracted and normalized {len(data)} relations.")
        return data
    elif isinstance(data, str):
        log.warning("Got string output from model, not JSON")
        return []
    else:
        log.warning(f"Unexpected relation extraction output type: {type(data)}")
        return []
