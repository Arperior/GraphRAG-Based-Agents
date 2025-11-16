from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

def _req(name: str) -> str:
    v = os.getenv(name)
    if not v or not v.strip():
        raise RuntimeError(f"Env var {name} is required but missing.")
    return v.strip()

@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str

@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    # Free-tier safe 
    model: str = "gemini-2.0-flash"   
    max_output_tokens: int = 512
    temperature: float = 0.2
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models"

@dataclass(frozen=True)
class LocalLLMConfig:
    # GGUF dir + filename (your setup: D:\models\...\mistral-7b-instruct-v0.1.Q4_K_M.gguf)
    model_dir: Path
    model_file: str
    n_ctx: int = 2048
    n_gpu_layers: int = 32  
    verbose: bool = False

@dataclass(frozen=True)
class AppConfig:
    root: Path
    data_dir: Path
    uploads_dir: Path
    cache_dir: Path
    logs_dir: Path
    prompts_dir: Path
    neo4j: Neo4jConfig
    gemini: GeminiConfig
    local_llm: LocalLLMConfig
    leiden_resolution: float = float(os.getenv("LEIDEN_RESOLUTION", "1.0"))
    retrieval_search_limit: int = int(os.getenv("RETRIEVAL_SEARCH_LIMIT", "10"))
    neo4j_query_limit: int = int(os.getenv("NEO4J_QUERY_LIMIT", "100"))
    run_relation_extraction: bool = os.getenv("RUN_RELATION_EXTRACTION", "true").lower() in ["1", "true", "yes"]

def load_config() -> AppConfig:
    neo4j = Neo4jConfig(
        uri=_req("NEO4J_URI"),
        user=_req("NEO4J_USER"),
        password=_req("NEO4J_PASSWORD"),
    )

    # Gemini can be optional during local-only testing; raise only if you use it.
    gem_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gem_key:
        # still build, you just won't be able to call Gemini
        gem_key = "MISSING"

    local_model_dir = Path(os.getenv("LOCAL_LLM_MODEL", "") or ".")
    local_model_file = os.getenv("LOCAL_LLM_FILE", "") or ""
    if not local_model_dir.exists():
        # Don't fail here; you might set later. Client will validate on first use.
        pass

    app = AppConfig(
        root=ROOT,
        data_dir=ROOT / "data",
        uploads_dir=ROOT / "data" / "uploads",
        cache_dir=ROOT / "data" / "cache",
        logs_dir=ROOT / "logs",
        prompts_dir=ROOT / "pipeline" / "prompts",
        neo4j=neo4j,
        gemini=GeminiConfig(api_key=gem_key),
        local_llm=LocalLLMConfig(
            model_dir=local_model_dir,
            model_file=local_model_file,
            n_ctx=int(os.getenv("LOCAL_N_CTX", "2048")),
            n_gpu_layers=int(os.getenv("LOCAL_N_GPU_LAYERS", "32")),
            verbose=os.getenv("LOCAL_VERBOSE", "0") == "1"
        ),
    )

    # ensure dirs
    for d in [app.data_dir, app.uploads_dir, app.cache_dir, app.logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return app
