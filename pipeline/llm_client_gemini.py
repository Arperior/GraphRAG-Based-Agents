from __future__ import annotations
import time
import logging
import requests
from typing import Any, Dict, Optional

from config.config import load_config

_cfg = load_config()
log = logging.getLogger("llm_gemini")


def _endpoint_for_model(model: str) -> str:
    """Constructs the endpoint url for the configured Gemini REST API."""
    base = _cfg.gemini.endpoint.rstrip("/")
    # Example: base + "/v1/models/{model}:generateContent?key="
    return f"{base}/{model}:generateContent?key={_cfg.gemini.api_key}"


def gemini_complete(prompt: str,
                    model: Optional[str] = None,
                    max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None,
                    top_p: Optional[float] = None,
                    top_k: Optional[int] = None,
                    retries: int = 3,
                    parse_json: bool = False,
                    timeout: int = 60) -> Any:
    """
    Gemini REST client with retries and safe parsing.

    If parse_json=True, attempt to parse the returned text as JSON and return the Python object.
    Otherwise return the assistant text string.
    """
    model = model or _cfg.gemini.model
    if not _cfg.gemini.api_key or _cfg.gemini.api_key == "MISSING":
        raise RuntimeError("GEMINI_API_KEY missing. Set it in config/.env.")

    url = _endpoint_for_model(model)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": int(max_tokens or _cfg.gemini.max_output_tokens),
            "temperature": float(temperature if temperature is not None else _cfg.gemini.temperature),
            "topP": float(top_p if top_p is not None else 0.9),
            "topK": int(top_k if top_k is not None else 40),
        },
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            if r.status_code != 200:
                log.warning(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(1.5 * attempt)
                continue

            data = r.json()

            # Basic error handling
            if "error" in data:
                msg = data["error"].get("message", "unknown")
                log.error(f"Gemini returned error: {msg}")
                return f"**Gemini Error:** {msg}"

            candidates = data.get("candidates") or []
            if not candidates:
                log.warning("Gemini returned no candidates.")
                return "**No candidates returned.**"

            text = candidates[0].get("content", {}).get("parts", [])
            if not text:
                return "**Empty content parts returned.**"
            out = text[0].get("text", "").strip()

            if parse_json:
                import json
                try:
                    return json.loads(out)
                except Exception as je:
                    log.warning(f"Failed to parse Gemini output as JSON: {je}")
                    # return raw string as fallback
                    return out

            return out

        except Exception as e:
            last_err = e
            log.warning(f"Gemini call failed attempt {attempt}: {e}")
            time.sleep(1.5 * attempt)

    # After retries
    log.error(f"Gemini failed after {retries} attempts: {last_err}")
    return f"**Error:** Gemini failed after {retries} attempts: {last_err}"
