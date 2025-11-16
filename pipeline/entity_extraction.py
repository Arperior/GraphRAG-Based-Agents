from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
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


def derive_relation_type(description: str) -> str:
    """Convert relation description to a canonical type."""
    d = (description or "").lower()
    if "founded" in d: return "FOUNDED"
    if "owns" in d or "acquired" in d: return "OWNS"
    if "employ" in d: return "EMPLOYED_BY"
    if "based" in d or "located" in d: return "LOCATED_IN"
    if "partner" in d: return "PARTNER_WITH"
    if "lead" in d or "ceo" in d: return "LEADS"
    if "member" in d: return "MEMBER_OF"
    return "RELATED_TO"


def parse_graphrag_tuples(output: str, tuple_delim: str, record_delim: str) -> Dict[str, List[dict]]:
    """Parse GraphRAG tuples into structured entity/relation dicts."""
    entities, relations = [], []
    if not output:
        return {"entities": entities, "relations": relations}

    items = [x.strip() for x in output.split(record_delim) if x.strip()]
    for it in items:
        try:
            inner = it.strip().lstrip("(").rstrip(")")
            parts = [p.strip().strip('"') for p in inner.split(tuple_delim)]
            if not parts:
                continue
            tag = parts[0].lower()
            if tag == "entity":
                name = parts[1] if len(parts) > 1 else ""
                etype = parts[2] if len(parts) > 2 else "UNKNOWN"
                desc = parts[3] if len(parts) > 3 else ""
                if name:
                    entities.append({"name": name, "type": etype, "description": desc})
            elif tag == "relationship":
                src = parts[1] if len(parts) > 1 else ""
                tgt = parts[2] if len(parts) > 2 else ""
                desc = parts[3] if len(parts) > 3 else ""
                strength_str = parts[4] if len(parts) > 4 else ""
                try:
                    strength = float(strength_str) if strength_str else 1.0
                except Exception:
                    strength = 1.0
                if src and tgt:
                    relations.append({
                        "source": src,
                        "target": tgt,
                        "relation": derive_relation_type(desc),
                        "evidence": desc,
                        "confidence": strength
                    })
        except Exception as e:
            log.warning(f"Failed to parse tuple: {e}")
            continue

    return {"entities": entities, "relations": relations}


def extract_graph(chunk_text: str, entity_types: str = "PERSON,ORGANIZATION,GEO") -> Dict:
    """Extract entities and basic relations from text chunk using GraphRAG prompt."""
    tpl_path = _cfg.prompts_dir / "extract_graph.txt"
    tpl = read_text(tpl_path)

    tuple_delim = "|~|"
    record_delim = "|||"
    completion_delim = "END_OF_OUTPUT"

    seeds = spacy_candidates(chunk_text)
    seed_text = f"\n\nPay special attention to these possible entities: {', '.join(seeds)}" if seeds else ""

    prompt = (
        tpl.replace("{entity_types}", entity_types)
           .replace("{tuple_delimiter}", tuple_delim)
           .replace("{record_delimiter}", record_delim)
           .replace("{completion_delimiter}", completion_delim)
           .replace("{input_text}", chunk_text + seed_text)
    )

    wrapper = f'''Return ONLY JSON with a single key "raw" whose value is the exact output:

INSTRUCTION:
{prompt}
'''

    try:
        data = generate_json(wrapper, max_tokens=512)
        if isinstance(data, dict) and "raw" in data:
            raw = str(data["raw"])
        else:
            raw = json.dumps(data)
    except Exception as e:
        log.error(f"Entity extraction failed: {e}")
        return {"entities": [], "relations": []}

    raw = raw.split(completion_delim)[0]
    return parse_graphrag_tuples(raw, tuple_delim, record_delim)
