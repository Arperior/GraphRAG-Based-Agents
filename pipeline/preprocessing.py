from __future__ import annotations
import re
from typing import List

try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None

def clean_basic(text: str) -> str:
    t = text.replace("\r\n", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_tokens(text: str, max_tokens: int = 600, overlap: int = 100) -> List[str]:
    if not text:
        return []
    if ENCODER:
        toks = ENCODER.encode(text)
        chunks = []
        start = 0
        while start < len(toks):
            end = min(start + max_tokens, len(toks))
            chunks.append(ENCODER.decode(toks[start:end]))
            start += max_tokens - overlap
        return chunks
    # char fallback (~4000 chars ~ 600 tokens)
    size = 4000
    step = size - int(overlap * 6)
    return [text[i:i+size] for i in range(0, len(text), step)]