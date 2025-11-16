from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.config import load_config

_cfg = load_config()
LOG_DIR = _cfg.logs_dir

# Ensure directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Common log format
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Root logger configuration
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT
)

# Stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
logging.getLogger().addHandler(console_handler)

# Define per-module log files (rotating handlers)
modules = [
    "app",
    "neo4j",
    "retrieval",
    "llm_local",
    "clustering",
    "entity_extraction",
    "relation_extractor",
    "graph_builder"
]

for mod in modules:
    file_path = LOG_DIR / f"{mod}.log"
    handler = RotatingFileHandler(file_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger = logging.getLogger(mod)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = True  # Keep console output too

logging.info("Logging system initialized. Separate rotating log files created per module.")
