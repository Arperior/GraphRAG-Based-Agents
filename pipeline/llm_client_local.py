from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import time
import re, json

from llama_cpp import Llama
from config.config import load_config

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


def generate_json(prompt: str, max_tokens: int = 256) -> dict | list | str:
    llm = _get_model()
    full_prompt = f"""[INST] You are a precise information extraction model.
Return ONLY valid JSON. Do not add commentary.

{prompt} [/INST]"""

    log.debug(f"Generating JSON with max_tokens={max_tokens}")
    start = time.time()

    try:
        out = llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["</s>"],
        )
    except Exception as e:
        log.error(f"LLM generation failed: {e}")
        return {"error": "generation_failed", "detail": str(e)}

    duration = time.time() - start
    text = out["choices"][0]["text"]
    log.info(f"Model generation completed in {duration:.2f}s, output length={len(text)} chars")

    m = re.search(r'(\{.*\}|\[.*\])', text, re.S)
    if not m:
        log.warning("No valid JSON detected in LLM output")
        return {"error": "no_json", "raw": text}

    try:
        parsed = json.loads(m.group(1))
        log.debug(f"Successfully parsed JSON: type={type(parsed).__name__}")
        return parsed
    except Exception as e:
        log.error(f"Invalid JSON format: {e}")
        return {"error": "invalid_json", "raw": text}
