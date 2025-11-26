# pipeline/graph_builder.py
from __future__ import annotations
from typing import Dict, List, Iterable
import logging
import uuid
import os
from pathlib import Path

# Existing imports
from pipeline.neo4j_client import store_chunk_with_graph
from pipeline.memory import record_user_chunk_interest

# NEW: Imports for Vision & Fusion
from pipeline.neo4j_client import (
    store_image_scene_graph, 
    search_potential_matches, 
    merge_entities
)
from pipeline.llm_client_local import generate_json  # Mistral (Text) for reasoning

log = logging.getLogger("graph_builder")

# Define path to prompts
PROMPT_DIR = Path(__file__).parent / "prompts"

def _load_prompt(filename: str) -> str:
    """Utility to load a prompt template from the prompts directory."""
    try:
        with open(PROMPT_DIR / filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        log.error(f"Could not load prompt {filename}: {e}")
        # Fallback minimal prompt to prevent crash
        return "Entity Resolution Task: Match '{visual_name}' against {candidates}. Return JSON."

def _normalize_entity(e) -> Dict | None:
    if isinstance(e, dict):
        name = (e.get("name") or "").strip()
        return {
            "name": name,
            "type": e.get("type", "UNKNOWN"),
            "description": e.get("description", "") or ""
        } if name else None
    elif isinstance(e, str):
        n = e.strip()
        return {"name": n, "type": "UNKNOWN", "description": ""} if n else None
    return None

def _normalize_relation(r) -> Dict | None:
    if not r or not isinstance(r, dict):
        return None

    src = (r.get("source") or r.get("src") or r.get("from") or "").strip()
    tgt = (r.get("target") or r.get("tgt") or r.get("to") or "").strip()
    if not src or not tgt:
        return None

    relation = r.get("relation") or r.get("rel") or "RELATED_TO"
    evidence = r.get("evidence") or r.get("ev") or ""
    try:
        conf = float(r.get("confidence", 1.0) or 1.0)
    except Exception:
        conf = 1.0

    return {"src": src, "tgt": tgt, "relation": relation, "evidence": evidence, "confidence": conf}

def build_and_store_graph(
    chunk_id: str,
    chunk_text: str,
    entities: List[Dict] | None,
    relations: List[Dict] | None,
    user_id: str | None = None,
    source: str = "user_text",
    query: str | None = None,
    tokens: List[str] | None = None,
    extra_relations: Iterable[Dict] | None = None,
):
    try:
        # Normalize entities
        ent_list: List[Dict] = []
        for e in (entities or []):
            n = _normalize_entity(e)
            if n:
                ent_list.append(n)

        # Normalize relations
        rels: List[Dict] = []
        for r in (relations or []):
            nr = _normalize_relation(r)
            if nr:
                rels.append(nr)

        # Append/merge any extra relations
        for r in (extra_relations or []):
            nr = _normalize_relation(r)
            if nr:
                key = (nr["src"], nr["tgt"], nr["relation"])
                if not any((x["src"], x["tgt"], x["relation"]) == key for x in rels):
                    rels.append(nr)

        chunk_obj = {"id": chunk_id, "text": chunk_text, "source": source}
        log.info(f"Building graph chunk {chunk_id}: {len(ent_list)} entities, {len(rels)} relations")
        store_chunk_with_graph(chunk_obj, user_id, ent_list, rels)

        if user_id:
            try:
                record_user_chunk_interest(user_id, chunk_id, query or "", tokens or [])
            except Exception as e:
                log.warning(f"Failed to record user-chunk interest for {chunk_id}: {e}")

        log.info(f"Successfully stored graph chunk {chunk_id}")

    except Exception as e:
        log.error(f"Failed to build/store graph chunk {chunk_id}: {e}", exc_info=True)
        raise

# ==============================================================================
# NEW: VISION PIPELINE BUILDER (N-MMKG + FUSION)
# ==============================================================================

def build_and_store_image(
    image_path: str, 
    scene_graph: Dict, 
    user_id: str | None,
    user_context: str | None = None
):
    """
    Orchestrates Image Storage and Cross-Modal Fusion.
    1. Unpacks Scene Graph.
    2. Stores Image Node (using filename as ID) + User Context.
    3. Triggers Fusion to link Visual Entities to existing Text Entities.
    """
    try:
        # UPDATED: Generate deterministic ID from filename
        # e.g., "my_diagram.png" -> "img_my_diagram"
        file_stem = Path(image_path).stem
        # Sanitize to ensure valid ID (replace spaces, etc if needed)
        safe_stem = "".join(c if c.isalnum() else "_" for c in file_stem)
        img_id = f"img_{safe_stem}"
        
        # Unpack data from Qwen/Vision Model
        summary = scene_graph.get("summary", "")
        entities = scene_graph.get("entities", [])
        relations = scene_graph.get("relations", [])
        
        log.info(f"Building Image Graph {img_id} from {image_path}")
        
        # 1. Store (N-MMKG)
        store_image_scene_graph(
            img_id, 
            image_path, 
            summary, 
            entities, 
            relations, 
            user_id, 
            user_context
        )
        
        # 2. Cross-Modal Fusion (Alignment)
        if entities:
            _perform_fusion_check(entities, context=user_context or "No user context provided.")
            
        log.info(f"Successfully stored and fused image graph {img_id}")

    except Exception as e:
        log.error(f"Failed to build/store image {image_path}: {e}", exc_info=True)
        raise


def _perform_fusion_check(visual_entities: List[Dict], context: str):
    """
    Implements the 'Entity Alignment' step using a templated prompt.
    """
    log.info("Starting Cross-Modal Fusion check...")
    
    # Load the prompt template once
    prompt_template = _load_prompt("entity_resolution.txt")

    for vis_ent in visual_entities:
        name = vis_ent.get("name")
        if not name: continue
        
        # A. Find Candidates (Fast DB Lookup)
        candidates = search_potential_matches(name)
        if not candidates:
            continue
            
        # B. LLM Verification (Reasoning Step)
        # Fill the template
        prompt = prompt_template.format(
            visual_name=name,
            visual_description=vis_ent.get('description', ''),
            candidates=candidates,
            image_context=context
        )
        
        try:
            result = generate_json(prompt, max_tokens=128) 
            
            if isinstance(result, list) and result: 
                result = result[0]
            
            if isinstance(result, dict) and result.get("match_found"):
                target = result.get("target_name")
                
                # Double check target exists in our candidate list
                if target and target in candidates:
                    log.info(f"FUSION MATCH: Merging '{name}' (Visual) -> '{target}' (Text)")
                    merge_entities(target, name) 
                else:
                    log.debug(f"Fusion rejected: Target '{target}' not in candidate list")
                    
        except Exception as e:
            log.warning(f"Fusion check failed for {name}: {e}")