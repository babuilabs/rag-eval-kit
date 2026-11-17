"""Minimal OpenAI-compatible chat client with a disk cache and rate-limit handling.

Every response is cached by (model, messages, params), so re-running an experiment costs no
API calls and gives identical outputs. Works with Groq by default; set LLM_BASE_URL and
LLM_API_KEY to point at any OpenAI-compatible server (OpenRouter, vLLM, Ollama, ...).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", "results/cache/llm"))


class DailyLimitReached(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
    if key:
        return key
    keyfile = os.environ.get("LLM_KEY_FILE")  # optional KEY=value file
    name = os.environ.get("LLM_KEY_NAME", "GROQ_API_KEY")
    if keyfile and Path(keyfile).exists():
        for line in Path(keyfile).read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("Set LLM_API_KEY (or GROQ_API_KEY) to call the model API.")


def _seconds(value: str | None) -> float:
    """Parse Groq-style durations such as '7.66s', '2m59.5s', '120ms'."""
    if not value:
        return 0.0
    if re.fullmatch(r"[\d.]+", value):
        return float(value)
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * units[u] for n, u in re.findall(r"([\d.]+)(ms|h|m|s)", value))


class _Budget:
    """Tracks the last seen per-model token budget so we wait *before* hitting a 429."""

    def __init__(self):
        self.state: dict[str, tuple[float, float]] = {}  # model -> (remaining_tokens, reset_at)

    def wait(self, model: str, need: int) -> None:
        remaining, reset_at = self.state.get(model, (1e9, 0.0))
        if remaining < need and time.time() < reset_at:
            time.sleep(reset_at - time.time() + 0.5)

    def update(self, model: str, headers) -> None:
        rem = headers.get("x-ratelimit-remaining-tokens")
        if rem is not None:
            self.state[model] = (float(rem), time.time() + _seconds(headers.get("x-ratelimit-reset-tokens")))


_budget = _Budget()


def chat(model: str, messages: list[dict], *, max_tokens: int = 512, temperature: float = 0.0,
         json_mode: bool = False, extra: dict | None = None, use_cache: bool = True) -> dict:
    """Return {'content', 'usage', 'latency_s', 'cached'}."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    body.update(extra or {})
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / model.replace("/", "__") / f"{key}.json"
    if use_cache and path.exists():
        out = json.loads(path.read_text(encoding="utf-8"))
        out["cached"] = True
        return out

    need = sum(len(m["content"]) for m in messages) // 3 + max_tokens
    for attempt in range(20):  # per-minute limits clear quickly; daily limits raise above
        _budget.wait(model, need)
        req = urllib.request.Request(
            BASE_URL + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": "Bearer " + _api_key(), "Content-Type": "application/json",
                     "User-Agent": "rag-eval-kit"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                _budget.update(model, r.headers)
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 429:
                if re.search(r"per day|TPD|RPD", msg):
                    raise DailyLimitReached(f"{model}: {msg[:200]}")
                time.sleep((_seconds(e.headers.get("retry-after")) or min(60, 2 ** attempt)) + random.uniform(0, 2))
                continue
            if e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code} from {model}: {msg[:300]}")
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"{model}: gave up after retries")

    out = {"content": data["choices"][0]["message"].get("content") or "",
           "usage": data.get("usage", {}), "latency_s": round(time.time() - t0, 3)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    out["cached"] = False
    return out


def chat_json(model: str, messages: list[dict], **kw) -> dict:
    """chat() in JSON mode, parsed; returns {} if the model produced invalid JSON."""
    out = chat(model, messages, json_mode=True, **kw)
    try:
        return json.loads(out["content"])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", out["content"], re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}
