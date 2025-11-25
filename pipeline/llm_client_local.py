# pipeline/llm_client_local.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import logging
import time

from llama_cpp import Llama
from config.config import load_config

# NEW: Import the robust parsers
from pipeline.json_utils import parse_llm_json_list, parse_llm_graph

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


def generate_json(prompt: str, max_tokens: int = 1024) -> Any:
    """
    Generate raw text from LLM and return robustly parsed JSON.
    Uses json_utils to handle missing brackets, extra text, etc.
    """
    llm = _get_model()
    # Mistral v0.1 specific formatting (adapt if using different model)
    full_prompt = f"[INST] {prompt} [/INST]"

    # Generate output
    out = llm(full_prompt, max_tokens=max_tokens, temperature=0.0, stop=["</s>"])
    text = out["choices"][0]["text"]

    log.info(f"Model generation completed, output length={len(text)} chars")
    
    # 1. Try to parse as a specific graph structure (entities + relations)
    #    This handles cases where the model returns { "entities": [...], "relations": [...] }
    #    or just a list of mixed objects.
    try:
        graph_res = parse_llm_graph(text)
        # heuristic: if we got results in either bucket, assume it was graph-like
        if graph_res["entities"] or graph_res["relations"]:
            return graph_res
    except Exception:
        pass

    # 2. Fallback: parse as a generic list of objects
    #    Useful if the prompt requested just "a list of relations"
    try:
        list_res = parse_llm_json_list(text)
        return list_res
    except Exception as e:
        log.error(f"JSON extraction failed completely: {e}")
        return []