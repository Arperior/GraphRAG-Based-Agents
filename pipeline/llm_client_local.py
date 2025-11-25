# pipeline/llm_client_local.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import logging
import time
import os
import base64
from io import BytesIO

# Heavy imports happen here, NOT in config.py
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image

from config.config import load_config
from pipeline.json_utils import parse_llm_json_list, parse_llm_graph

_cfg = load_config()
log = logging.getLogger("llm_local")

_text_model: Optional[Llama] = None
_vision_model: Optional[Llama] = None


def _get_text_model() -> Llama:
    """Loads the text-only model (e.g., Mistral)"""
    global _text_model
    if _text_model is not None:
        return _text_model

    model_path = Path(_cfg.local_llm.model_dir) / _cfg.local_llm.model_file
    log.info(f"Loading Local Text LLM: {model_path}")

    if not model_path.exists():
        log.error(f"Text Model not found: {model_path}")
        raise FileNotFoundError(f"Text Model not found at: {model_path}")

    start = time.time()
    try:
        _text_model = Llama(
            model_path=str(model_path),
            n_ctx=_cfg.local_llm.n_ctx,
            n_gpu_layers=_cfg.local_llm.n_gpu_layers,
            verbose=_cfg.local_llm.verbose,
        )
        log.info(f"Loaded Text Model in {time.time() - start:.2f}s")
    except Exception as e:
        log.error(f"Failed to load Text Model: {e}")
        raise
    return _text_model


def _get_vision_model() -> Llama:
    """Loads the multimodal model (LLaVA) from Config"""
    global _vision_model
    if _vision_model is not None:
        return _vision_model

    model_path = _cfg.llava.model_path
    clip_path = _cfg.llava.clip_path

    log.info(f"Loading Local Vision LLM: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LLaVA Model not found at: {model_path}\n"
            "Please ensure the model file exists in 'C:\\models\\'."
        )
    
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"LLaVA Projector (mmproj) not found at: {clip_path}")

    start = time.time()
    try:
        chat_handler = Llava15ChatHandler(clip_model_path=clip_path)
        _vision_model = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=_cfg.llava.n_ctx,
            n_gpu_layers=_cfg.llava.n_gpu_layers,
            logits_all=True,
            verbose=_cfg.llava.verbose
        )
        log.info(f"Loaded Vision Model in {time.time() - start:.2f}s")
    except Exception as e:
        log.error(f"Failed to load Vision Model: {e}")
        raise
    return _vision_model


def generate_text(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Generate text using the Text Model."""
    llm = _get_text_model()
    full_prompt = f"[INST] {prompt} [/INST]"
    
    try:
        out = llm(
            full_prompt, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            stop=["</s>", "[/INST]"]
        )
        return out["choices"][0]["text"].strip()
    except Exception as e:
        log.error(f"Text generation failed: {e}")
        return "Error generating text."


def generate_json(prompt: str, max_tokens: int = 1024) -> Any:
    """Generate JSON using the Text Model."""
    llm = _get_text_model()
    full_prompt = f"[INST] {prompt} [/INST]"

    try:
        out = llm(full_prompt, max_tokens=max_tokens, temperature=0.0, stop=["</s>"])
        text = out["choices"][0]["text"]
        
        try:
            graph_res = parse_llm_graph(text)
            if graph_res["entities"] or graph_res["relations"]:
                return graph_res
        except Exception:
            pass

        try:
            list_res = parse_llm_json_list(text)
            return list_res
        except Exception:
            pass
            
        return []
    except Exception as e:
        log.error(f"JSON extraction failed: {e}")
        return []


def describe_image(image: Image.Image) -> str:
    """
    Generate a detailed narrative description of an image using LLaVA.
    """
    llm = _get_vision_model()
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{img_b64}"

    prompt = (
        "You are an expert Data Analyst. "
        "Write a detailed, dense narrative paragraph describing this image for a Knowledge Graph. "
        "1. Identify specific entities (people, objects, text labels, diagrams). "
        "2. Explicitly describe the relationships and connections between them. "
        "3. Do NOT use bullet points or lists. "
        "4. Do NOT repeat the same information. "
        "5. Transcribe any visible text exactly."
    )

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes images in detailed paragraphs."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            max_tokens=768,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.2
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"Image description failed: {e}")
        return ""