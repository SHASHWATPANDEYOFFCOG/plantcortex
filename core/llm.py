"""LLMClient — a thin, cache-first, demo-safe wrapper over Gemini.

Why this shape:
* **Disk cache keyed by prompt hash.** The scripted demo path is pre-warmed, so a
  live demo replays from cache and survives flaky wifi. ``DEMO_STRICT=1`` refuses to
  hit the network at all (cache-miss -> empty result + warning, never a crash).
* **Retry + backoff** on transient (429/5xx) errors.
* **Structured logging** of every call (prompt hash, latency, tokens, cache hit) to
  ``data/cache/llm/calls.jsonl`` — this is the Technical-Excellence audit trail.
* **Graceful degradation.** With no API key, ``get_llm()`` returns ``None`` and callers
  fall back to the deterministic rule-based extractor, so the pipeline still runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import settings

log = logging.getLogger("plantcortex.llm")


def _hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _loads_lenient(text: str) -> dict:
    """Parse JSON that may be wrapped in ```json fences or have stray prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _parse_retry_delay(msg: str) -> Optional[float]:
    """Pull Google's suggested retryDelay (seconds) out of a 429 error string."""
    m = re.search(r"retryDelay['\"]?[:=]\s*['\"]?(\d+)", msg)
    return float(m.group(1)) if m else None


def _safe_text(resp) -> str:
    """Get text from a Gemini response without raising on blocked/truncated output."""
    try:
        t = resp.text
        if t:
            return t
    except Exception:
        pass
    # fall back to concatenating candidate parts
    try:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts)
    except Exception:
        return ""


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    total_tokens: int = 0


class LLMClient:
    """Gemini-backed client with caching, retry, logging."""

    def __init__(self, api_key: str, model: str, vision_model: str,
                 cache_dir: Path, demo_strict: bool = False) -> None:
        from google import genai  # imported lazily so no-key envs need not install it

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.vision_model = vision_model
        self.cache_dir = cache_dir
        self.demo_strict = demo_strict
        self.stats = LLMStats()
        self.min_interval = settings.llm_min_interval_s
        self.max_retry_wait = settings.llm_max_retry_wait_s
        self._last_call = 0.0
        self.quota_blocked = False   # set once the daily/short quota is clearly hit
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._logfile = cache_dir / "calls.jsonl"

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    # -- cache ------------------------------------------------------------- #
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[str]:
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["text"]
        return None

    def _write_cache(self, key: str, text: str, meta: dict) -> None:
        self._cache_path(key).write_text(
            json.dumps({"text": text, "meta": meta}), encoding="utf-8")

    def _log(self, rec: dict) -> None:
        with self._logfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # -- core call --------------------------------------------------------- #
    def _generate(self, *, model: str, contents, config_kwargs: dict,
                  cache_payload: dict, disable_thinking: bool = False) -> str:
        key = _hash({"model": model, **cache_payload})
        cached = self._read_cache(key)
        if cached is not None:
            self.stats.calls += 1
            self.stats.cache_hits += 1
            self._log({"hash": key, "model": model, "cached": True})
            return cached

        if self.demo_strict:
            log.warning("DEMO_STRICT cache miss %s -> empty result", key)
            self._log({"hash": key, "model": model, "cached": False,
                       "strict_miss": True})
            return ""

        from google.genai import types

        if disable_thinking:
            # Gemini 2.5 Flash is a thinking model; thinking tokens eat the output
            # budget and can yield EMPTY JSON. For structured extraction we want no
            # thinking. (thinking_budget=0 is supported by 2.5-flash.)
            config_kwargs = {**config_kwargs,
                             "thinking_config": types.ThinkingConfig(thinking_budget=0)}
        # Once quota is clearly exhausted, stop hitting the network (which only burns
        # more requests-per-day). Callers fall back to the rule-based extractor.
        if self.quota_blocked:
            return ""

        cfg = types.GenerateContentConfig(**config_kwargs)
        last_err: Optional[Exception] = None
        for attempt in range(3):
            self._throttle()
            t0 = time.time()
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=contents, config=cfg)
                dt = int((time.time() - t0) * 1000)
                text = _safe_text(resp)
                toks = getattr(resp, "usage_metadata", None)
                total = getattr(toks, "total_token_count", 0) or 0
                self.stats.calls += 1
                self.stats.total_tokens += total
                # Never cache an empty/blocked response — don't poison future runs.
                if text.strip():
                    self._write_cache(key, text, {"latency_ms": dt, "tokens": total})
                self._log({"hash": key, "model": model, "cached": False,
                           "latency_ms": dt, "tokens": total, "attempt": attempt,
                           "empty": not text.strip()})
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                is_net = any(c in msg for c in ("503", "500", "502", "UNAVAILABLE",
                                                "deadline", "timeout", "ReadError",
                                                "ConnectError", "10053", "getaddrinfo"))
                if is_quota:
                    delay = _parse_retry_delay(msg)
                    # RPM (per-minute) limits return a short retryDelay — wait it out.
                    if attempt < 2 and delay and delay <= self.max_retry_wait:
                        log.info("rate-limited; waiting %ss then retrying", delay)
                        time.sleep(delay + 1)
                        continue
                    # No usable delay / still limited after waiting -> stop burning quota.
                    self.quota_blocked = True
                    log.warning("LLM quota exhausted; switching to offline fallback.")
                    break
                if is_net and attempt < 2:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                break
        log.error("LLM call failed: %s", str(last_err)[:200])
        self._log({"hash": key, "model": model, "error": str(last_err)[:200]})
        return ""

    # -- public API -------------------------------------------------------- #
    def complete_json(self, system: str, user: str, *,
                      max_tokens: int = 3072, temperature: float = 0.0) -> dict:
        contents = f"{system}\n\n---\n\n{user}" if system else user
        text = self._generate(
            model=self.model, contents=contents,
            config_kwargs=dict(response_mime_type="application/json",
                               temperature=temperature,
                               max_output_tokens=max_tokens),
            cache_payload={"system": system, "user": user, "mode": "json",
                           "max_tokens": max_tokens, "temperature": temperature},
            disable_thinking=True)
        return _loads_lenient(text)

    def complete_text(self, prompt: str, *, max_tokens: int = 1024,
                      temperature: float = 0.2) -> str:
        return self._generate(
            model=self.model, contents=prompt,
            config_kwargs=dict(temperature=temperature, max_output_tokens=max_tokens),
            cache_payload={"prompt": prompt, "mode": "text",
                           "max_tokens": max_tokens, "temperature": temperature})

    def vision_json(self, prompt: str, image_bytes: bytes, *,
                    mime: str = "image/png", max_tokens: int = 4096) -> dict:
        from google.genai import types

        img_key = hashlib.sha256(image_bytes).hexdigest()[:16]
        contents = [types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt]
        text = self._generate(
            model=self.vision_model, contents=contents,
            config_kwargs=dict(response_mime_type="application/json",
                               temperature=0.0, max_output_tokens=max_tokens),
            cache_payload={"prompt": prompt, "img": img_key, "mode": "vision",
                           "max_tokens": max_tokens},
            disable_thinking=True)
        return _loads_lenient(text)


_LLM_SINGLETON: Optional[LLMClient] = None


def get_llm() -> Optional[LLMClient]:
    """Return a shared LLMClient, or None if no key is configured (offline mode)."""
    global _LLM_SINGLETON
    if _LLM_SINGLETON is not None:
        return _LLM_SINGLETON
    if settings.llm_provider == "gemini" and settings.has_llm_key:
        _LLM_SINGLETON = LLMClient(
            api_key=settings.gemini_api_key, model=settings.llm_model,
            vision_model=settings.llm_vision_model, cache_dir=settings.llm_cache_dir,
            demo_strict=settings.demo_strict)
        return _LLM_SINGLETON
    log.warning("No LLM key configured; using deterministic rule-based fallback.")
    return None
