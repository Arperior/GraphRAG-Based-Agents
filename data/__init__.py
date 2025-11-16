"""
Data package — stores raw uploads, cache, and generated graph files.
Ensures required subfolders exist at import time.
"""
import os
from pathlib import Path

base = Path(__file__).parent
for sub in ["cache", "uploads", "graphs", "samples"]:
    os.makedirs(base / sub, exist_ok=True)
