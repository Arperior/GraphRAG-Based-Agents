# pipeline/llm_client_vision.py
from __future__ import annotations
import base64
import logging
import time
import json
import io
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from PIL import Image
from ultralytics import YOLO  # Requires: pip install ultralytics
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from config.config import load_config
from pipeline import gpu_manager
from pipeline.llm_client_local import generate_json as generate_text_json

_cfg = load_config()
log = logging.getLogger("llm_vision")

@dataclass
class ImageFeatureBlock:
    index: int
    label: str
    image_data: bytes
    description: str = ""

def _load_prompt(filename: str) -> str:
    path = _cfg.prompts_dir / filename
    if not path.exists():
        log.warning(f"Prompt file '{filename}' not found at {path}. Using fallback.")
        return ""
    return path.read_text(encoding="utf-8")

def _clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Robustly attempts to parse JSON from potentially malformed LLM output.
    """
    if not raw_text:
        return {}
    
    # 1. Try direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 2. Extract JSON block if wrapped in markdown
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    # 3. Try to find the first '{' and last '}'
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1:
            json_str = raw_text[start:end+1]
            return json.loads(json_str)
    except:
        pass
        
    log.warning(f"Failed to parse JSON. Raw output: {raw_text[:100]}...")
    return {}

def _get_yolo_model():
    gpu_manager.request_permission_to_load("yolo")
    if gpu_manager._CURRENT_MODEL_TYPE == "yolo" and gpu_manager._CURRENT_MODEL is not None:
        return gpu_manager._CURRENT_MODEL
        
    log.info(f"Loading YOLOv8 model: {_cfg.yolo.model_file}...")
    try:
        model = YOLO(_cfg.yolo.model_file)
        gpu_manager.register_model(model, "yolo")
        return model
    except Exception as e:
        log.error(f"Failed to load YOLO: {e}")
        raise

def _get_llava_model() -> Llama:
    gpu_manager.request_permission_to_load("vision")
    if gpu_manager._CURRENT_MODEL_TYPE == "vision" and gpu_manager._CURRENT_MODEL is not None:
        return gpu_manager._CURRENT_MODEL

    model_path = Path(_cfg.llava.model_path)
    clip_path = Path(_cfg.llava.clip_path)

    if not model_path.exists() or not clip_path.exists():
        raise FileNotFoundError("LLaVA model or Clip not found.")

    log.info(f"Loading LLaVA from {model_path}")
    try:
        chat_handler = Llava15ChatHandler(clip_model_path=str(clip_path))
        model = Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_ctx=_cfg.llava.n_ctx,
            n_gpu_layers=_cfg.llava.n_gpu_layers,
            verbose=_cfg.llava.verbose,
            logits_all=True
        )
        gpu_manager.register_model(model, "vision")
        return model
    except Exception as e:
        log.error(f"Failed to load LLaVA: {e}")
        raise

def _create_grid_blocks(image_path: str) -> List[ImageFeatureBlock]:
    """
    Fallback: Slices image into a 2x2 grid if YOLO fails.
    This mimics segmentation for text-heavy documents/slides.
    """
    log.info("YOLO failed/skipped. Using 2x2 Grid Fallback.")
    img = Image.open(image_path)
    
    # --- FIX: Convert RGBA to RGB for JPEG compatibility ---
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # -------------------------------------------------------

    w, h = img.size
    
    # Define 2x2 grid
    grid_w = w // 2
    grid_h = h // 2
    
    blocks = []
    
    # Coordinates: (left, top, right, bottom)
    grid_coords = [
        (0, 0, grid_w, grid_h),          # Top-Left
        (grid_w, 0, w, grid_h),          # Top-Right
        (0, grid_h, grid_w, h),          # Bottom-Left
        (grid_w, grid_h, w, h)           # Bottom-Right
    ]
    
    labels = ["Top-Left Quadrant", "Top-Right Quadrant", "Bottom-Left Quadrant", "Bottom-Right Quadrant"]
    
    for i, coords in enumerate(grid_coords):
        crop = img.crop(coords)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        crop.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        blocks.append(ImageFeatureBlock(
            index=i, 
            label=labels[i], 
            image_data=img_bytes,
            description="" 
        ))
        
    return blocks

# =========================================================
# STEP 1: SEGMENTATION (YOLO + FALLBACK) 
# =========================================================
def segment_image(image_path: str) -> List[ImageFeatureBlock]:
    model = _get_yolo_model()
    log.info(f"Segmenting image: {image_path}")
    
    results = model(image_path, verbose=False)
    result = results[0]
    
    blocks = []
    original_img = Image.open(image_path)
    
    # --- FIX: Convert RGBA to RGB for JPEG compatibility ---
    # We do this here too, in case YOLO *does* find objects on a PNG later
    if original_img.mode in ("RGBA", "P"):
        original_img = original_img.convert("RGB")
    # -------------------------------------------------------
    
    # 1. Try YOLO First
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        label = result.names[class_id]
        
        # Filter: Ignore "person" if confidence is low
        if label == "person" and box.conf[0] < 0.5:
            continue

        c = box.xyxy[0].tolist() 
        crop = original_img.crop((c[0], c[1], c[2], c[3]))
        
        img_byte_arr = io.BytesIO()
        crop.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        blocks.append(ImageFeatureBlock(index=i, label=label, image_data=img_bytes))
        
    log.info(f"YOLO detected {len(blocks)} feature blocks.")

    # 2. Fallback if YOLO found nothing
    if len(blocks) == 0:
        blocks = _create_grid_blocks(image_path)
        
    return blocks

# =========================================================
# STEP 2: DESCRIPTION GENERATION (LLaVA) 
# =========================================================
def generate_block_descriptions(blocks: List[ImageFeatureBlock]):
    """
    Iterates through crops and prompts LLaVA to describe them.
    Dynamically switches between 'Object Description' and 'Document Transcription'.
    """
    llm = _get_llava_model()
    
    # Load both prompts
    visual_template = _load_prompt("describe_feature_block.txt") or "Describe this {label} visual attributes."
    document_template = _load_prompt("describe_document_block.txt") or "Transcribe text and describe diagram in this {label}."
    
    for block in blocks:
        log.info(f"Generating description for Block {block.index} ({block.label})...")
        
        base64_img = base64.b64encode(block.image_data).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{base64_img}"
        
        # --- LOGIC SWITCH ---
        # If it's a Grid Fallback (Quadrant), use the Document/OCR prompt
        if "Quadrant" in block.label:
            prompt_text = document_template.format(label=block.label)
        else:
            # It's a YOLO object (Person, Car, etc.), use Visual prompt
            prompt_text = visual_template.format(label=block.label)
        # --------------------

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        }]
        
        try:
            # Increased max_tokens to 512 because OCR can be verbose
            out = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.1)
            content = out["choices"][0]["message"]["content"]
            
            # Ensure we don't save empty strings
            if content and content.strip():
                block.description = content
            else:
                block.description = "No content detected in this region."
                
        except Exception as e:
            log.warning(f"Failed to describe block {block.index}: {e}")
            block.description = "Description generation failed."
# =========================================================
# STEP 3: GLOBAL EXTRACTION (LLaVA) 
# =========================================================
def extract_global_entities(image_path: str, user_context: str | None) -> Dict[str, Any]:
    llm = _get_llava_model()
    
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')
    img_url = f"data:image/jpeg;base64,{base64_img}"
    
    prompt_text = _load_prompt("extract_scene_graph.txt")
    if not prompt_text:
        prompt_text = "Extract a JSON Scene Graph with 'summary', 'entities', and 'relations'."

    if user_context:
        prompt_text += f"\n\n### USER CONTEXT\n{user_context}"
        
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]
    }]
    
    log.info("Extracting global scene graph...")
    
    # 1. Try with strict JSON mode first
    try:
        out = llm.create_chat_completion(
            messages=messages, 
            max_tokens=1024, 
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_content = out["choices"][0]["message"]["content"]
        log.debug(f"LLaVA Raw Output: {raw_content[:100]}...") # Log start of output
        
        return _clean_and_parse_json(raw_content)

    except Exception as e:
        log.warning(f"Strict JSON extraction failed ({e}). Retrying without constraints...")
        
        # 2. Retry WITHOUT strict JSON mode (sometimes models choke on the grammar)
        try:
            out = llm.create_chat_completion(
                messages=messages, 
                max_tokens=1024, 
                temperature=0.1
            )
            raw_content = out["choices"][0]["message"]["content"]
            log.info("Standard extraction completed. Attempting parsing...")
            
            parsed = _clean_and_parse_json(raw_content)
            if parsed: return parsed
            
        except Exception as e2:
            log.error(f"Retry failed: {e2}")

    return {"summary": "Processing Failed", "entities": [], "relations": []}

# =========================================================
# STEP 4: ALIGNMENT (Mistral) 
# =========================================================
def align_blocks_to_entities(blocks: List[ImageFeatureBlock], global_data: Dict[str, Any]):
    if not blocks or not global_data.get("entities"):
        return

    entities_list = [e.get("name") for e in global_data["entities"]]
    template = _load_prompt("align_feature_block.txt")
    if not template: return

    log.info("Aligning feature blocks to entities (Step 4)...")
    
    for block in blocks:
        prompt = template.format(
            label=block.label,
            desc=block.description,
            entities=json.dumps(entities_list)
        )
        
        result = generate_text_json(prompt, max_tokens=128)
        
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        if isinstance(result, dict) and result.get("match_found"):
            target_name = result.get("entity_name")
            for ent in global_data["entities"]:
                if ent["name"] == target_name:
                    ent["fine_grained_description"] = block.description
                    ent["yolo_label"] = block.label
                    log.info(f"Aligned Block {block.index} -> Entity '{target_name}'")

# =========================================================
# MASTER PIPELINE
# =========================================================
def _refine_entities_from_block(block_label: str, description: str) -> Dict[str, List[Any]]:
    """
    Extracts structured Entities AND Relations from a block description.
    Handles both 'Diagram' style and 'Natural Scene' style descriptions.
    """
    if not description or len(description) < 10:
        return {"entities": [], "relations": []}

    template = _load_prompt("refine_entities.txt")
    if not template: 
        # Fallback if file missing
        return {"entities": [], "relations": []}

    prompt = template.format(block_label=block_label, description=description)

    try:
        # Generate (Mistral)
        result = generate_text_json(prompt, max_tokens=512)
        
        # Robust Parsing
        if isinstance(result, dict):
            # Expected format: {"entities": [], "relations": []}
            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", [])
            }
        elif isinstance(result, list):
            # Legacy format fallback (just a list of entities)
            return {"entities": result, "relations": []}
            
        return {"entities": [], "relations": []}

    except Exception as e:
        log.warning(f"Refinement failed for {block_label}: {e}")
        return {"entities": [], "relations": []}

def process_image_pipeline(image_path: str, user_context: str | None = None) -> Dict[str, Any]:
    try:
        # 1. Segmentation (YOLO or Grid Fallback)
        blocks = segment_image(image_path)
        
        # 2. Block Description (LLaVA)
        if blocks:
            generate_block_descriptions(blocks)
        else:
            log.info("Step 2 Skipped: No feature blocks detected.")
            
        # 3. Global Extraction (LLaVA)
        scene_graph = extract_global_entities(image_path, user_context)
        global_entities = scene_graph.get("entities", [])
        
        # 4. Alignment / Promotion Logic
        has_blocks = len(blocks) > 0
        has_globals = len(global_entities) > 0

        if has_blocks and has_globals:
            # Best Case: We have global structure. Align detailed blocks to it.
            align_blocks_to_entities(blocks, scene_graph)
            
        elif has_blocks and not has_globals:
            # Fallback Case: Global extraction returned nothing (common for diagrams/abstract).
            # We rely on the blocks to build the graph from scratch.
            log.info("Global Entities missing. Performing Smart Extraction (Entities+Relations).")
            
            promoted_entities = []
            promoted_relations = []
            
            for block in blocks:
                # Extract rich structure from the description
                data = _refine_entities_from_block(block.label, block.description)
                
                # A. Handle Entities
                current_block_entities = []
                for item in data["entities"]:
                    item["source"] = f"extracted_from_{block.label}"
                    # Inject user context into the first block's items for searchability
                    if block.index == 0 and user_context:
                        item["description"] = f"{item.get('description','')} [Context: {user_context}]"
                    
                    promoted_entities.append(item)
                    current_block_entities.append(item["name"]) # Keep track for sanity check

                # B. Handle Relations
                for rel in data["relations"]:
                    # Optional: Only keep relations if both source/target were actually found
                    # (Prevents hallucinations linking to non-existent nodes)
                    if rel["source"] in current_block_entities and rel["target"] in current_block_entities:
                        promoted_relations.append(rel)
                    else:
                        # Looser check: Just add it, GraphBuilder usually handles missing nodes gracefully
                        promoted_relations.append(rel)

            # Update the Scene Graph object
            scene_graph["entities"] = promoted_entities
            scene_graph["relations"] = promoted_relations
            
            # Fallback Summary if needed
            if not scene_graph.get("summary"):
                scene_graph["summary"] = f"Image content analyzed via grid segmentation. Extracted {len(promoted_entities)} entities and {len(promoted_relations)} relations."

        return scene_graph
        
    except Exception as e:
        log.error(f"Image pipeline failed: {e}", exc_info=True)
        return {"summary": "Processing Failed", "entities": [], "relations": []}