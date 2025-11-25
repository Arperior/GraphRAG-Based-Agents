# pipeline/relation_extractor.py
from __future__ import annotations
import logging
from typing import List, Dict

from config.config import load_config
from pipeline.llm_client_local import generate_json
from pipeline.utils import read_text

_cfg = load_config()
log = logging.getLogger("relation_extractor")


def extract_relations_from_text(text: str) -> List[Dict]:
    """
    Extract relation triples from a text block using the LLM.
    Returns a list of dicts with keys:
      - source
      - target
      - relation
      - evidence
      - confidence (float)
    NOTE: This function will not aggressively normalize relation labels;
    the model's relation string is used as-is (uppercasing can be done downstream if desired).
    """
    prompt_path = _cfg.prompts_dir / "extract_relations.txt"
    try:
        tpl = read_text(prompt_path)
    except Exception:
        tpl = "Extract relations and return ONLY a JSON array."

    prompt = tpl.replace("{input_text}", text)

    try:
        # generate_json should return parsed JSON (repaired) from llm_client_local
        out = generate_json(prompt, max_tokens=2048)

        if not out:
            log.warning("Relation extractor: model returned no output.")
            return []

        # If the model returns a dict with 'relations' key, handle it
        if isinstance(out, dict) and "relations" in out:
            cand = out.get("relations") or []
        elif isinstance(out, list):
            cand = out
        elif isinstance(out, dict):
            # sometimes the model returns a single dict describing one relation
            cand = [out]
        else:
            log.warning(f"Relation extractor: unexpected output type {type(out)}")
            return []

        cleaned = []
        for r in cand:
            if not isinstance(r, dict):
                continue
            src = (r.get("source") or r.get("src") or "").strip()
            tgt = (r.get("target") or r.get("tgt") or "").strip()
            rel_label = r.get("relation") or r.get("rel") or r.get("predicate") or ""
            if not src or not tgt or not rel_label:
                continue
            evidence = r.get("evidence", "") or ""
            try:
                conf = float(r.get("confidence", 1.0) or 1.0)
            except Exception:
                conf = 1.0

            cleaned.append({
                "source": src,
                "target": tgt,
                "relation": rel_label,
                "evidence": evidence,
                "confidence": conf
            })

        log.info(f"Relation extractor returned {len(cleaned)} relations.")
        return cleaned

    except Exception as e:
        log.error(f"Relation extraction failed: {e}", exc_info=True)
        return []
