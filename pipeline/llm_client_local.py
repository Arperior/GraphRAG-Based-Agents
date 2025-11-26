# pipeline/llm_client_local.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import logging
import time

from llama_cpp import Llama
from config.config import load_config
from pipeline.json_utils import parse_llm_json_list, parse_llm_graph

# NEW: Import the traffic cop
from pipeline import gpu_manager

_cfg = load_config()
# We no longer keep a global _model variable here; the gpu_manager holds it.
log = logging.getLogger("llm_local")


def _get_model() -> Llama:
    # 1. Ask manager if we need to unload Vision model first
    gpu_manager.request_permission_to_load("text")

    # 2. If manager says we already have the text model loaded, return it
    if gpu_manager._CURRENT_MODEL_TYPE == "text" and gpu_manager._CURRENT_MODEL is not None:
        return gpu_manager._CURRENT_MODEL

    # 3. Otherwise, load Mistral from scratch
    model_path = Path(_cfg.local_llm.model_dir) / _cfg.local_llm.model_file
    log.info(f"Loading local TEXT model from: {model_path}")

    if not model_path.exists():
        log.error(f"Local GGUF model not found: {model_path}")
        raise FileNotFoundError(f"Local GGUF not found: {model_path}")

    start = time.time()
    try:
        model = Llama(
            model_path=str(model_path),
            n_ctx=_cfg.local_llm.n_ctx,
            n_gpu_layers=_cfg.local_llm.n_gpu_layers,
            verbose=_cfg.local_llm.verbose,
        )
        duration = time.time() - start
        log.info(f"Loaded text model in {duration:.2f}s")
        
        # 4. Register with manager
        gpu_manager.register_model(model, "text")
        return model

    except Exception as e:
        log.error(f"Failed to load text model: {e}")
        raise


def generate_json(prompt: str, max_tokens: int = 1024) -> Any:
    """
    Generate raw text from LLM and return robustly parsed JSON.
    """
    llm = _get_model()
    # Mistral v0.1 specific formatting
    full_prompt = f"[INST] {prompt} [/INST]"

    try:
        out = llm(full_prompt, max_tokens=max_tokens, temperature=0.0, stop=["</s>"])
        text = out["choices"][0]["text"]

        log.info(f"Model generation completed, output length={len(text)} chars")
        
        # Robust parsing (Graph first, then List)
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
        log.error(f"LLM generation failed: {e}")
        return []