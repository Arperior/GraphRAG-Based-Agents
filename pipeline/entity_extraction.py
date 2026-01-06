from __future__ import annotations
from typing import Dict, List
import logging
import spacy
import re # Added for statistical pattern matching

from config.config import load_config
from pipeline.utils import read_text, dedup_keep_order
from pipeline.llm_client_local import generate_json

_cfg = load_config()
log = logging.getLogger("entity_extraction")

# Load spaCy for candidate entity seeding
try:
    # Prefer larger model if available for better accuracy, fallback to sm
    try:
        _nlp = spacy.load("en_core_web_md")
    except:
        _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None


def spacy_candidates(text: str) -> List[str]:
    """Use spaCy to detect possible named entities or noun chunks."""
    if not _nlp:
        return []
    doc = _nlp(text)
    
    # 1. Standard Named Entities (Person, Org, GPE, Date, etc.)
    ents = [e.text.strip() for e in doc.ents if len(e.text.strip()) > 1]
    
    # 2. Noun Chunks (captures "The GDP of China" type phrases)
    chunks = [c.text.strip() for c in doc.noun_chunks if len(c.text.strip()) > 2]
    
    combined = ents + chunks
    return dedup_keep_order([e for e in combined if e])


def extract_statistical_entities(text: str) -> List[Dict]:
    """
    Heuristic extractor for dense statistical data common in Charts.
    Finds percentages, currency, and years to ensure they exist as nodes.
    """
    stats = []
    
    # Pattern 1: Percentages (e.g., "58%", "10.5%")
    # We grab the word *before* the percentage too for context (e.g., "increased 58%")
    pct_matches = re.finditer(r"([a-zA-Z]{3,}\s+)?(\d+(\.\d+)?%)", text)
    for m in pct_matches:
        val = m.group(0).strip()
        stats.append({
            "name": val,
            "type": "METRIC",
            "description": "Statistical percentage value detected in text."
        })

    # Pattern 2: Currency/Values (e.g., "$10 million", "€500")
    cur_matches = re.finditer(r"([$€£¥]\s?\d+(\.\d+)?\s?(million|billion|trillion|k)?)", text, re.IGNORECASE)
    for m in cur_matches:
        stats.append({
            "name": m.group(0).strip(),
            "type": "METRIC",
            "description": "Monetary value detected in text."
        })

    # Pattern 3: Years (e.g., "in 2024", "since 1990") - Context aware
    year_matches = re.finditer(r"\b(in|since|year|from)\s+(19|20)\d{2}\b", text, re.IGNORECASE)
    for m in year_matches:
        # Extract just the year part for the name, but keep phrase for context
        full_phrase = m.group(0).strip()
        year_val = full_phrase.split()[-1]
        stats.append({
            "name": year_val,
            "type": "TIME",
            "description": f"Temporal entity context: {full_phrase}"
        })

    return stats


def _normalize_entities(raw_entities) -> List[Dict]:
    """
    Ensure entities are list of dicts with at least 'name'. 
    """
    out = []
    if isinstance(raw_entities, dict):
        raw_entities = [raw_entities]
    if not raw_entities:
        return []

    for e in raw_entities:
        if isinstance(e, str):
            out.append({"name": e.strip(), "type": "UNKNOWN", "description": "Extracted entity"})
        elif isinstance(e, dict):
            name = e.get("name") or e.get("id") or e.get("label")
            if not name:
                continue
            out.append({
                "name": name.strip(),
                "type": e.get("type", "UNKNOWN"),
                "description": e.get("description", "") or e.get("desc", "Extracted entity")
            })

    # Deduplicate by name
    seen = set()
    final = []
    for e in out:
        clean_name = e["name"].lower().strip()
        if clean_name not in seen and len(clean_name) > 1:
            final.append(e)
            seen.add(clean_name)
    return final


def _normalize_relations(raw_relations) -> List[Dict]:
    if isinstance(raw_relations, dict):
        raw_relations = [raw_relations]
    if not raw_relations:
        return []

    out = []
    for r in raw_relations:
        if not isinstance(r, dict): continue

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


def extract_graph(chunk_text: str, entity_types: str = "PERSON,ORGANIZATION,GEO,EVENT,CONCEPT,METRIC") -> Dict:
    """
    Extract entities and relations using LLM + SpaCy + Statistical Heuristics.
    """
    tpl_path = _cfg.prompts_dir / "extract_graph.txt"
    try:
        tpl = read_text(tpl_path)
    except:
        tpl = "Extract entities (types: {entity_types}) and relations from: {input_text}"

    # 1. Get Seeds (SpaCy + Statistical)
    # We combine linguistic entities with hard data points (numbers)
    nlp_seeds = spacy_candidates(chunk_text)
    stat_entities = extract_statistical_entities(chunk_text)
    stat_seeds = [e["name"] for e in stat_entities]
    
    # Combine seeds for the prompt context
    all_seeds = dedup_keep_order(nlp_seeds + stat_seeds)
    seed_str = ", ".join(all_seeds[:50]) # Increased limit
    
    # 2. Prompt LLM
    # We explicitly tell the LLM to look for data points now
    prompt = tpl.replace("{entity_types}", entity_types).replace("{input_text}", chunk_text)
    if seed_str:
        prompt += f"\n\nIMPORTANT: Pay special attention to these potential entities (names, metrics, data points): {seed_str}"

    entities = []
    relations = []

    try:
        log.info("Running entity extraction LLM...")
        data = generate_json(prompt, max_tokens=1024) 

        if isinstance(data, dict):
            entities = _normalize_entities(data.get("entities", []))
            relations = _normalize_relations(data.get("relations", []))
        elif isinstance(data, list):
            entities = _normalize_entities(data)
        
        log.info(f"LLM Extracted: {len(entities)} entities, {len(relations)} relations.")

    except Exception as e:
        log.error(f"LLM extraction failed: {e}")

    # 3. Aggressive Fallback Integration
    # If the LLM missed any Statistical Entities (e.g., "58%"), we force them in.
    # This is crucial for "Correctness" in Chart QA.
    existing_names = {e['name'].lower() for e in entities}
    added_count = 0
    
    # Add missed statistical entities first (Higher priority for this benchmark)
    for stat in stat_entities:
        if stat["name"].lower() not in existing_names:
            entities.append(stat)
            existing_names.add(stat["name"].lower())
            added_count += 1

    # Add missed NLP keywords (Lower priority but good for context)
    for seed in nlp_seeds:
        if seed.lower() not in existing_names:
            entities.append({
                "name": seed,
                "type": "KEYWORD", 
                "description": "Keyword extracted from text analysis."
            })
            existing_names.add(seed.lower())
            added_count += 1
            
    if added_count > 0:
        log.info(f"Fallback added {added_count} missing entities (stats/keywords).")

    return {"entities": entities, "relations": relations}