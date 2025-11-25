from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import time
import re, json

from llama_cpp import Llama
from config.config import load_config
from json_repair import repair_json

_cfg = load_config()
_model: Optional[Llama] = None
log = logging.getLogger("llm_local")


def _get_model() -> Llama:
    global _model
    if _model is not None:
        return _model

    model_path = Path(_cfg.local_llm.model_dir) / _cfg.local_llm.model_file
    log.info(f"Loading local LLM model from: {model_path}")

    if not model_path.exists():
        log.error(f"Local GGUF model not found: {model_path}")
        raise FileNotFoundError(f"Local GGUF not found: {model_path}")

    start = time.time()
    try:
        _model = Llama(
            model_path=str(model_path),
            n_ctx=_cfg.local_llm.n_ctx,
            n_gpu_layers=_cfg.local_llm.n_gpu_layers,
            verbose=_cfg.local_llm.verbose,
        )
        duration = time.time() - start
        log.info(
            f"Loaded model '{model_path.name}' "
            f"(ctx={_cfg.local_llm.n_ctx}, gpu_layers={_cfg.local_llm.n_gpu_layers}) "
            f"in {duration:.2f}s"
        )
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise

    return _model


def generate_json(prompt: str, max_tokens: int = 1024):
    """
    Generate raw text from LLM and return repaired JSON.
    This function handles:
    - JSON inside text
    - truncated JSON (missing closing brackets)
    - malformed arrays/objects
    - trailing commas
    - extra commentary
    """
    llm = _get_model()
    full_prompt = f"[INST] {prompt} [/INST]"

    # Generate output
    out = llm(full_prompt, max_tokens=max_tokens, temperature=0.0, stop=["</s>"])
    text = out["choices"][0]["text"]

    log.info(f"Model generation completed, output length={len(text)} chars")
    
    # Optional debug print
    # print("=== RAW MODEL OUTPUT ===")
    # print(text)
    # print("=========================")

    # Strategy 1: Attempt direct repair of the whole text
    # json_repair is very good at adding missing ']' or '}' automatically
    try:
        decoded = repair_json(text, return_objects=True)
        if decoded:
            return decoded
    except:
        pass

    # Strategy 2: Extract from first '[' to the very end of the string
    # This handles cases where the model starts a list but gets cut off.
    # We ignore the missing ']' and let repair_json fix it.
    if "[" in text:
        start_idx = text.find("[")
        candidate = text[start_idx:] # Take everything from [ onwards
        try:
            return repair_json(candidate, return_objects=True)
        except:
            pass
            
    # Strategy 3: Extract from first '{' (for single objects)
    if "{" in text:
        start_idx = text.find("{")
        candidate = text[start_idx:]
        try:
            return repair_json(candidate, return_objects=True)
        except:
            pass

    # Strategy 4: Fallback Regex (The old strict way)
    # Only looks for content inside matched brackets, ignoring outer noise
    try:
        m = re.search(r'(\[.*\]|\{.*\})', text, re.S)
        if m:
            return repair_json(m.group(1), return_objects=True)
    except:
        pass

    log.error("JSON extraction failed; returning empty list/dict to prevent crash.")
    return []