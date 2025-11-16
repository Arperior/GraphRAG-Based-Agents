from __future__ import annotations
import json
import logging
from typing import Dict, List

from config.config import load_config
from pipeline.utils import read_text
from pipeline.llm_client_local import generate_json

_cfg = load_config()
log = logging.getLogger("relation_extractor")

def extract_relations(chunk_text: str, entity_specs: str = "PERSON,ORGANIZATION,GEO") -> List[Dict]:
    """Extract additional relations from text chunk using GraphRAG 'extract_claims.txt'."""
    tpl_path = _cfg.prompts_dir / "extract_claims.txt"
    tpl = read_text(tpl_path)

    tuple_delim = "|~|"
    record_delim = "|||"
    completion_delim = "END_OF_OUTPUT"

    prompt = (
        tpl.replace("{entity_specs}", entity_specs)
           .replace("{tuple_delimiter}", tuple_delim)
           .replace("{record_delimiter}", record_delim)
           .replace("{completion_delimiter}", completion_delim)
           .replace("{claim_description}", "general relations or claims between entities")
           .replace("{input_text}", chunk_text)
    )

    wrapper = f'''Return ONLY JSON with key "raw" whose value is the output:

INSTRUCTION:
{prompt}
'''

    try:
        data = generate_json(wrapper, max_tokens=512)
        if isinstance(data, dict) and "raw" in data:
            raw = data["raw"]
        else:
            raw = json.dumps(data)
    except Exception as e:
        log.error(f"Relation extraction failed: {e}")
        return []

    raw = raw.split(completion_delim)[0]
    return _parse_relation_tuples(raw, tuple_delim, record_delim)


def _parse_relation_tuples(output: str, tuple_delim: str, record_delim: str) -> List[Dict]:
    """Parse 'relationship' tuples from GraphRAG relation extraction output."""
    relations = []
    if not output:
        return relations

    items = [x.strip() for x in output.split(record_delim) if x.strip()]
    for it in items:
        try:
            inner = it.strip().lstrip("(").rstrip(")")
            parts = [p.strip().strip('"') for p in inner.split(tuple_delim)]
            if len(parts) < 5 or parts[0].lower() != "relationship":
                continue
            src, tgt, desc, strength = parts[1], parts[2], parts[3], parts[4]
            try:
                conf = float(strength) if strength else 1.0
            except Exception:
                conf = 1.0
            relations.append({
                "source": src,
                "target": tgt,
                "relation": _derive_relation_type(desc),
                "evidence": desc,
                "confidence": conf
            })
        except Exception as e:
            log.warning(f"Failed to parse relation tuple: {e}")
            continue
    return relations


def _derive_relation_type(desc: str) -> str:
    """Canonical mapping for free-text relations."""
    d = (desc or "").lower()
    if "founded" in d: return "FOUNDED"
    if "owns" in d or "acquired" in d: return "OWNS"
    if "employ" in d: return "EMPLOYED_BY"
    if "based" in d or "located" in d: return "LOCATED_IN"
    if "partner" in d: return "PARTNER_WITH"
    if "lead" in d or "ceo" in d: return "LEADS"
    if "member" in d: return "MEMBER_OF"
    return "RELATED_TO"
