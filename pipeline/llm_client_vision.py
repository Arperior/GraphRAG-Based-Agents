# pipeline/llm_client_vision.py
from __future__ import annotations
import base64
import logging
import json
import io
import re
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from PIL import Image
from ultralytics import YOLO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import sys
# New Import for OCR
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from config.config import load_config
from pipeline import gpu_manager
from pipeline.llm_client_local import generate_json as generate_text_json

_cfg = load_config()
log = logging.getLogger("llm_vision")

@contextmanager
def suppress_stdout_stderr():
    """
    Redirects C-level stdout/stderr to /dev/null to silence llama.cpp logs.
    """
    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Save original handles
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    try:
        # Open devnull
        devnull = os.open(os.devnull, os.O_RDWR)
        # Replace stdout/stderr with devnull
        os.dup2(devnull, original_stdout_fd)
        os.dup2(devnull, original_stderr_fd)
        yield
    finally:
        # Restore stdout/stderr
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

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
    Robustly parses JSON. If parsing fails, returns a partial dict with raw text 
    to prevent pipeline crashes.
    """
    default_structure = {"summary": raw_text, "entities": [], "relations": []}
    
    if not raw_text: 
        return {"summary": "", "entities": [], "relations": []}
    
    parsed = {}
    
    # 1. Try direct parse
    try: 
        parsed = json.loads(raw_text)
    except: 
        # 2. Markdown extraction
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            try: parsed = json.loads(match.group(1))
            except: pass
        
        # 3. Brute force bracket finding
        if not parsed:
            try:
                start, end = raw_text.find("{"), raw_text.rfind("}")
                if start != -1 and end != -1: 
                    parsed = json.loads(raw_text[start:end+1])
            except: pass

    # Fallback: If parsing totally failed, return raw text as summary
    if not parsed:
        log.warning(f"JSON parsing failed. Using raw text fallback.")
        return default_structure

    # Ensure critical keys exist (Guard clause)
    if "entities" not in parsed or not isinstance(parsed["entities"], list):
        parsed["entities"] = []
    if "relations" not in parsed:
        parsed["relations"] = []
    if "summary" not in parsed:
        parsed["summary"] = raw_text[:200] + "..." # Fallback summary

    return parsed

# =========================================================
# MODEL LOADERS (GPU MANAGED)
# =========================================================

def _get_yolo_model(is_document: bool = False):
    model_type = "yolo_doc" if is_document else "yolo_general"
    model_path = _cfg.yolo.doc_model_file if is_document else _cfg.yolo.model_file

    gpu_manager.request_permission_to_load(model_type)
    
    if gpu_manager._CURRENT_MODEL_TYPE == model_type and gpu_manager._CURRENT_MODEL is not None:
        return gpu_manager._CURRENT_MODEL
        
    log.info(f"Loading YOLO ({model_type})...")
    # YOLO is usually quiet, but we can wrap it if needed
    try:
        model = YOLO(model_path)
        gpu_manager.register_model(model, model_type)
        return model
    except Exception as e:
        log.error(f"Failed to load YOLO: {e}")
        raise

def _get_paddle_model():
    model_type = "ocr"
    gpu_manager.request_permission_to_load(model_type)

    if gpu_manager._CURRENT_MODEL_TYPE == model_type and gpu_manager._CURRENT_MODEL is not None:
        return gpu_manager._CURRENT_MODEL
    
    if PaddleOCR is None:
        return None

    try:
        # show_log=False already helps, but Paddle prints C++ warnings too
        with suppress_stdout_stderr():
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        gpu_manager.register_model(ocr, model_type)
        return ocr
    except Exception as e:
        log.error(f"Failed to load PaddleOCR: {e}")
        return None

def _get_llava_model() -> Llama:
    model_type = "vision"
    gpu_manager.request_permission_to_load(model_type)
    
    if gpu_manager._CURRENT_MODEL_TYPE == model_type and gpu_manager._CURRENT_MODEL is not None:
        return gpu_manager._CURRENT_MODEL

    model_path = Path(_cfg.llava.model_path)
    clip_path = Path(_cfg.llava.clip_path)

    log.info(f"Loading LLaVA...")
    try:
        # SILENCE C++ LOGS HERE
        with suppress_stdout_stderr():
            chat_handler = Llava15ChatHandler(clip_model_path=str(clip_path))
            model = Llama(
                model_path=str(model_path),
                chat_handler=chat_handler,
                n_ctx=_cfg.llava.n_ctx,
                n_gpu_layers=_cfg.llava.n_gpu_layers,
                verbose=False, # Python level silence
                logits_all=False
            )
        gpu_manager.register_model(model, model_type)
        return model
    except Exception as e:
        log.error(f"Failed to load LLaVA: {e}")
        raise

# =========================================================
# UTILITIES
# =========================================================

def classify_image_mode(image_path: str) -> str:
    """
    Step 0: Router. Uses LLaVA to decide if image is Document or Natural.
    """
    llm = _get_llava_model()
    
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')
    img_url = f"data:image/jpeg;base64,{base64_img}"

    prompt = (
        "Classify this image into one of two categories:\n"
        "1. 'document': Charts, graphs, tables, receipts, slides, diagrams, text-heavy pages.\n"
        "2. 'natural': Real-world photos, people, animals, landscapes, objects.\n"
        "Reply with ONLY the category name in JSON format: {\"category\": \"...\"}."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_url}}]}]
    
    try:
        out = llm.create_chat_completion(messages=messages, max_tokens=32, temperature=0.0)
        raw = out["choices"][0]["message"]["content"]
        res = _clean_and_parse_json(raw)
        
        cat = res.get("category", "").lower().strip()
        
        doc_keywords = ["chart", "graph", "table", "text", "document", "diagram", "plot", "slide", "receipt"]
        if any(k in cat for k in doc_keywords):
            return "document"
            
        return "natural"

    except Exception as e:
        log.warning(f"Classification failed: {e}. Defaulting to natural.")
        return "natural"

def _create_grid_blocks(image_path: str, grid_size: int = 3) -> List[ImageFeatureBlock]:
    """
    Fallback: Slices image into grid_size x grid_size.
    Changed default to 3x3 (9 blocks) for better chart coverage.
    """
    log.info(f"Using {grid_size}x{grid_size} Grid Segmentation.")
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")

    w, h = img.size
    step_w = w // grid_size
    step_h = h // grid_size
    blocks = []
    count = 0
    
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * step_w
            top = row * step_h
            right = w if col == grid_size - 1 else (col + 1) * step_w
            bottom = h if row == grid_size - 1 else (row + 1) * step_h
            
            crop = img.crop((left, top, right, bottom))
            img_byte_arr = io.BytesIO()
            crop.save(img_byte_arr, format='JPEG')
            
            blocks.append(ImageFeatureBlock(
                index=count, 
                label=f"Region R{row+1} C{col+1}", 
                image_data=img_byte_arr.getvalue()
            ))
            count += 1
    return blocks

# =========================================================
# PIPELINE STEPS
# =========================================================

def segment_image(image_path: str, is_document: bool) -> List[ImageFeatureBlock]:
    """
    Step 1: Segmentation. Selects YOLO model based on mode.
    """
    model = _get_yolo_model(is_document=is_document)
    
    log.info(f"Segmenting {image_path} (Mode: {'DOC' if is_document else 'NATURAL'})")
    results = model(image_path, verbose=False)
    result = results[0]
    
    blocks = []
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        conf = float(box.conf[0])

        if is_document:
            # Keep everything in doc mode (low threshold)
            if conf < 0.15: continue 
        else:
            if label == "person" and conf < 0.5: continue

        c = box.xyxy[0].tolist()
        crop = img.crop((c[0], c[1], c[2], c[3]))
        
        img_byte_arr = io.BytesIO()
        crop.save(img_byte_arr, format='JPEG')
        
        blocks.append(ImageFeatureBlock(index=i, label=label, image_data=img_byte_arr.getvalue()))
        
    log.info(f"YOLO found {len(blocks)} blocks.")

    if len(blocks) == 0:
        # 3x3 is often better for charts than 4x4 (too small) or 2x2 (too big)
        size = 3 if is_document else 2
        blocks = _create_grid_blocks(image_path, grid_size=size)
        
    return blocks

def generate_block_descriptions(blocks: List[ImageFeatureBlock], use_ocr: bool):
    """
    Step 2: Description. Injects OCR AND appends raw data for lossless retrieval.
    """
    llm = _get_llava_model()
    ocr = _get_paddle_model() if use_ocr else None
    
    visual_tpl = _load_prompt("describe_feature_block.txt") or "Describe this {label} visual attributes."
    doc_tpl = _load_prompt("describe_document_block.txt") or "Transcribe text and describe diagram in this {label}."
    
    for block in blocks:
        log.info(f"Describing Block {block.index} ({block.label})...")
        
        ocr_text = ""
        temp_path = None
        
        # 1. OCR with Safer Temp File Handling
        if use_ocr and ocr:
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(block.image_data)
                    temp_path = tmp.name
                
                res = ocr.ocr(temp_path, cls=True)
                if res and res[0]:
                    lines = [line[1][0] for line in res[0] if line[1][1] > 0.6]
                    ocr_text = "\n".join(lines)
                    
            except Exception as e:
                log.warning(f"OCR Error on block {block.index}: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass 

        # 2. Prompt Construction
        base64_img = base64.b64encode(block.image_data).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{base64_img}"
        
        is_grid = "Region" in block.label
        prompt_fmt = doc_tpl if (use_ocr or is_grid) else visual_tpl
        full_prompt = prompt_fmt.format(label=block.label)
        
        if ocr_text:
            full_prompt += (
                f"\n\n[OCR DATA START]\n"
                f"The following text was detected in this region:\n{ocr_text}\n"
                f"[OCR DATA END]\n\n"
                f"Use this text to ensure numerical precision and correct spelling."
            )

        # 3. LLaVA Inference
        try:
            out = llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]
                }],
                max_tokens=512,
                temperature=0.1
            )
            content = out["choices"][0]["message"]["content"]
            
            # --- CRITICAL FIX FOR DENSE DATA ---
            # Append the raw OCR text to the description. 
            # LLaVA often summarizes "values from 10 to 90" but loses the specific "58%".
            # By appending it, we ensure the number exists in the database for retrieval.
            if ocr_text:
                content += f"\n\n=== EXTRACTED DATA ===\n{ocr_text}"
            
            block.description = content if content.strip() else "No content."
            
        except Exception as e:
            log.error(f"Description failed for {block.index}: {e}")
            block.description = f"Processing failed. Raw Text: {ocr_text}" if ocr_text else "Processing failed."

def extract_global_entities(image_path: str, user_context: str | None) -> Dict[str, Any]:
    """
    Step 3: Global Scene Graph (Standard LLaVA)
    """
    llm = _get_llava_model()
    
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')
    
    prompt = _load_prompt("extract_scene_graph.txt") or "Extract JSON with summary, entities, relations."
    if user_context:
        prompt += f"\n\nCONTEXT: {user_context}"
        
    try:
        out = llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }],
            max_tokens=1024,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return _clean_and_parse_json(out["choices"][0]["message"]["content"])
    except Exception as e:
        log.error(f"Global extraction failed: {e}")
        return {"summary": "Extraction Failed", "entities": [], "relations": []}

def align_blocks_to_entities(blocks: List[ImageFeatureBlock], global_data: Dict[str, Any]):
    """
    Step 4: Alignment (Mistral - CPU/Text Model)
    """
    if not blocks or not global_data.get("entities"): return
    
    ents = [e.get("name") for e in global_data["entities"]]
    tpl = _load_prompt("align_feature_block.txt")
    if not tpl: return

    log.info("Aligning blocks to entities...")
    for block in blocks:
        prompt = tpl.format(label=block.label, desc=block.description, entities=json.dumps(ents))
        res = generate_text_json(prompt, max_tokens=128)
        
        if isinstance(res, list) and res: res = res[0]
        if isinstance(res, dict) and res.get("match_found"):
            target = res.get("entity_name")
            for e in global_data["entities"]:
                if e["name"] == target:
                    e["fine_grained_description"] = block.description
                    e["yolo_label"] = block.label

# =========================================================
# MAIN ENTRY POINT
# =========================================================

def process_image_pipeline(image_path: str, user_context: str | None = None) -> Dict[str, Any]:
    try:
        # Router
        # (Router calls LLaVA, so it will now be silent)
        mode = classify_image_mode(image_path)
        is_doc = (mode == "document")
        log.info(f"Pipeline Mode: {mode.upper()}")

        # Segmentation
        blocks = segment_image(image_path, is_document=is_doc)
        
        # Description
        if blocks:
            # We silence the loop because LLaVA generates logs for every block decoding
            with suppress_stdout_stderr(): 
                generate_block_descriptions(blocks, use_ocr=is_doc)
        else:
            log.info("No feature blocks detected.")
            
        # Global Graph
        with suppress_stdout_stderr():
            scene_graph = extract_global_entities(image_path, user_context)
        
        # Alignment
        if scene_graph.get("entities") and blocks:
            align_blocks_to_entities(blocks, scene_graph)
        elif blocks and not scene_graph.get("entities"):
            ents = []
            for b in blocks:
                ents.append({
                    "name": f"{b.label} {b.index}", 
                    "type": "VISUAL_BLOCK", 
                    "description": b.description,
                    "source": "yolo_block"
                })
            scene_graph["entities"] = ents
            
        return scene_graph

    except Exception as e:
        log.error(f"Pipeline fatal error: {e}", exc_info=True)
        return {"summary": "Error", "entities": [], "relations": []}