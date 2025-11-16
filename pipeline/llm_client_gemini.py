from __future__ import annotations
import requests, time, logging
from config.config import load_config

_cfg = load_config()
log = logging.getLogger("app")

def gemini_complete(prompt: str,
                    max_tokens: int | None = None,
                    temperature: float | None = None,
                    retries: int = 3) -> str:
    """Gemini REST client with retries and safe parsing."""
    if not _cfg.gemini.api_key or _cfg.gemini.api_key == "MISSING":
        raise RuntimeError("GEMINI_API_KEY missing. Set it in .env.")

    model = _cfg.gemini.model
    url = f"{_cfg.gemini.endpoint}/{model}:generateContent?key={_cfg.gemini.api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens or _cfg.gemini.max_output_tokens,
            "temperature": temperature if temperature is not None else _cfg.gemini.temperature,
            "topP": 0.9,
            "topK": 40,
        },
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=body, timeout=60)
            if r.status_code != 200:
                log.warning(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(1.5 * attempt)
                continue

            data = r.json()

            # Error or safety blocked
            if "error" in data:
                return f"**Gemini Error:** {data['error'].get('message','unknown')}"

            cand = data.get("candidates", [])
            if not cand:
                return "**No candidates returned.**"

            parts = cand[0].get("content", {}).get("parts", [])
            if not parts:
                return "**Empty content parts returned.**"

            return parts[0].get("text", "").strip() or "**Empty text response.**"

        except Exception as e:
            log.warning(f"Gemini call failed (attempt {attempt}): {e}")
            time.sleep(1.5 * attempt)

    return "**Error:** Gemini could not process the request after multiple attempts."
