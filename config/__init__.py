"""
Configuration package for environment and runtime settings.
Exposes `load_config()` for all modules.
"""
from .config import load_config

__all__ = ["load_config"]
