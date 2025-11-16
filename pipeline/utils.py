from __future__ import annotations
from pathlib import Path
from typing import Iterable

def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")

def format_with_vars(template: str, **kv) -> str:
    out = template
    for k, v in kv.items():
        out = out.replace("{" + k + "}", str(v))
    return out

def truncate(s: str, n: int = 300) -> str:
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

def dedup_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
