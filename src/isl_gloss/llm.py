"""LLM access. One small interface, two HTTP implementations (Groq, Gemini) sharing one retry loop; standard library only.

Failure policy (from the brief): never silently substitute another translator. Errors are raised as typed exceptions
and recorded on the affected sentences. The API key is read from the environment, sent in a header (never in the URL),
and scrubbed from every message.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from .config import TranslatorConfig

USER_AGENT = "isl-gloss/0.1 (python-urllib)"  # some providers' gateways reject the default Python-urllib agent


class LLMError(Exception):
    kind = "llm_error"


class MissingApiKey(LLMError):
    kind = "missing_api_key"


class AuthError(LLMError):
    kind = "auth_error"


class BadRequest(LLMError):
    kind = "bad_request"


class ModelUnavailable(BadRequest):
    """HTTP 404: the model id does not exist or is closed to this account. Every batch would fail the same way."""
    kind = "model_unavailable"


class QuotaExhausted(LLMError):
    """Daily (or otherwise non-retryable) quota is gone. The run stops and can be resumed later."""
    kind = "quota_exhausted"


class TransientError(LLMError):
    kind = "transient_error"


class BlockedResponse(LLMError):
    kind = "blocked_response"


class TruncatedResponse(LLMError):
    kind = "truncated_response"


class RequestTooLarge(TruncatedResponse):
    """HTTP 413: the request exceeds a size/token limit. Splitting the group into smaller requests can help."""
    kind = "request_too_large"


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, schema: dict) -> LLMResponse: ...


def load_env_file(path: str | Path = ".env") -> None:
    """Minimal .env reader. Existing environment variables win. Values are never printed."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class RateLimiter:
    """Spaces requests at least 60/rpm seconds apart (free tiers are per-minute limited)."""

    def __init__(self, rpm: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.min_interval = 60.0 / rpm
        self._clock, self._sleep = clock, sleep
        self._last: Optional[float] = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            gap = self.min_interval - (now - self._last)
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()


class TokenPacer:
    """Keeps the tokens sent in any rolling 60-second window under a budget. Groq limits tokens per minute (8K on the
    reported free tier for gpt-oss-120b) as well as requests, and that is usually the limit that binds."""

    def __init__(self, tokens_per_minute: Optional[int], clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.tpm = tokens_per_minute
        self._clock, self._sleep = clock, sleep
        self._events: deque = deque()  # (time, tokens)

    def wait(self, estimate: int) -> None:
        if not self.tpm:
            return
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= 60:
                self._events.popleft()
            used = sum(t for _, t in self._events)
            if not self._events or used + estimate <= self.tpm:
                return  # (a single request larger than the budget is let through: waiting would never help)
            self._sleep(max(0.05, self._events[0][0] + 60 - now) + 0.05)

    def record(self, tokens: int) -> None:
        if self.tpm:
            self._events.append((self._clock(), tokens))


def to_json_schema(schema: dict) -> dict:
    """Gemini-dialect schema (upper-case type names) -> standard JSON Schema for strict structured outputs:
    lower-case types, and additionalProperties=false on every object."""
    def walk(node):
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items()}
            if isinstance(out.get("type"), str):
                out["type"] = out["type"].lower()
                if out["type"] == "object":
                    out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


def parse_wait_seconds(message: str) -> Optional[float]:
    """Groq messages end with e.g. 'Please try again in 7.5s' / '6m52.8s' / '850ms'."""
    m = re.search(r"try again in\s+((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?(?:[\d.]+ms)?)", message, re.I)
    if not m or not m.group(1):
        return None
    text, total = m.group(1), 0.0
    for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", text):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total or None


class _HttpClient:
    """Shared request loop: pacing, bounded retries with backoff, typed errors, key scrubbing."""

    provider = ""
    missing_key_hint = ""

    def __init__(self, cfg: TranslatorConfig, api_key: Optional[str] = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.model = cfg.model
        self._key = api_key if api_key is not None else os.environ.get(cfg.api_key_env, "")
        if not self._key:
            raise MissingApiKey(f"{cfg.api_key_env} is not set. {self.missing_key_hint} Set it as an environment variable "
                                f"or in a .env file next to run_translator.py. Nothing was sent.")
        self._sleep = sleep
        self._limiter = RateLimiter(cfg.requests_per_minute, clock=clock, sleep=sleep)
        self.requests_made = 0

    # ---- hooks -----------------------------------------------------------------------------------------------
    def _url(self) -> str:
        raise NotImplementedError

    def _auth_headers(self) -> dict:
        raise NotImplementedError

    def _body(self, system: str, user: str, schema: dict) -> dict:
        raise NotImplementedError

    def _parse(self, payload: dict) -> LLMResponse:
        raise NotImplementedError

    def _is_daily_quota(self, body: str, retry_after: Optional[float]) -> bool:
        return bool(re.search(r"per ?day|PerDay|daily", body, re.I))

    def _retry_after(self, exc: urllib.error.HTTPError, body: str) -> Optional[float]:
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header and header.replace(".", "", 1).isdigit():
            return float(header)
        m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', body)
        return float(m.group(1)) if m else None

    def _before_request(self, system: str, user: str) -> None:
        pass

    def _after_response(self, resp: LLMResponse) -> None:
        pass

    # ---- shared loop -----------------------------------------------------------------------------------------
    def complete(self, system: str, user: str, schema: dict) -> LLMResponse:
        url = self._url()
        data = json.dumps(self._body(system, user, schema)).encode("utf-8")
        last: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_attempts + 1):
            self._limiter.wait()
            self._before_request(system, user)
            self.requests_made += 1
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **self._auth_headers()},
            )
            retry_after: Optional[float] = None
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_seconds) as resp:
                    out = self._parse(json.loads(resp.read().decode("utf-8")))
                self._after_response(out)
                return out
            except urllib.error.HTTPError as exc:
                body = self._scrub(exc.read().decode("utf-8", "replace"))
                retry_after = self._retry_after(exc, body)
                code = exc.code
                if code == 429:
                    if self._is_daily_quota(body, retry_after):
                        raise QuotaExhausted(f"Daily quota exhausted (HTTP 429). {self._short(body)}") from None
                    last = TransientError(f"Rate limited (HTTP 429). {self._short(body)}")
                elif code in (401, 403):
                    raise AuthError(f"HTTP {code}: check the API key and that the model is enabled. {self._short(body)}") from None
                elif code == 404:
                    raise ModelUnavailable(f"HTTP 404 for model {self.model}: {self._short(body)}") from None
                elif code == 413:
                    raise RequestTooLarge(f"HTTP 413: request too large. {self._short(body)}") from None
                elif code == 400:
                    raise BadRequest(f"HTTP 400: {self._short(body)}") from None
                elif code in (408, 498) or code >= 500:
                    last = TransientError(f"HTTP {code}. {self._short(body)}")
                else:
                    raise LLMError(f"HTTP {code}: {self._short(body)}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = TransientError(f"Network error: {self._scrub(str(exc))}")
            if attempt < self.cfg.max_attempts:
                self._sleep(self._backoff(attempt, retry_after))
        raise TransientError(f"Gave up after {self.cfg.max_attempts} attempts. Last error: {last}") from None

    # ---- helpers ---------------------------------------------------------------------------------------------
    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        base = min(self.cfg.backoff_base_seconds * (2 ** (attempt - 1)), self.cfg.backoff_max_seconds)
        wait = base + random.uniform(0, base * 0.25)
        if retry_after is not None:
            wait = max(wait, min(retry_after, self.cfg.backoff_max_seconds * 2))
        return wait

    @staticmethod
    def _short(body: str, limit: int = 300) -> str:
        try:
            msg = json.loads(body)["error"]["message"]
        except Exception:  # noqa: BLE001 - body may not be JSON
            msg = body
        return re.sub(r"\s+", " ", msg)[:limit]


class GeminiClient(_HttpClient):
    provider = "gemini"
    missing_key_hint = "Create a free key in Google AI Studio."

    def _url(self) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/models/{self.model}:generateContent"

    def _auth_headers(self) -> dict:
        return {"x-goog-api-key": self._key}

    def _body(self, system: str, user: str, schema: dict) -> dict:
        gen = {
            "temperature": self.cfg.temperature,
            "maxOutputTokens": self.cfg.max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }
        if self.cfg.thinking_budget is not None and self.model.startswith("gemini-2.5"):
            gen["thinkingConfig"] = {"thinkingBudget": self.cfg.thinking_budget}
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }

    def _parse(self, payload: dict) -> LLMResponse:
        block = (payload.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise BlockedResponse(f"Prompt blocked by the API (blockReason={block}).")
        cands = payload.get("candidates") or []
        if not cands:
            raise BlockedResponse("The API returned no candidates.")
        cand = cands[0]
        finish = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if finish == "MAX_TOKENS":
            raise TruncatedResponse("Response hit the output token limit.")
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION") and not text:
            raise BlockedResponse(f"Response blocked (finishReason={finish}).")
        if not text.strip():
            raise BlockedResponse(f"Empty response (finishReason={finish}).")
        usage = payload.get("usageMetadata") or {}
        return LLMResponse(text=text, model=payload.get("modelVersion") or self.model, finish_reason=finish,
                           usage={k: usage[k] for k in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount") if k in usage})


class GroqClient(_HttpClient):
    """Groq's OpenAI-compatible chat completions API with strict JSON-schema output (supported by openai/gpt-oss-120b)."""

    provider = "groq"
    missing_key_hint = "Create a free API key in the Groq console (console.groq.com, API Keys)."
    EXPECTED_OUTPUT_TOKENS = 500  # used only to pace tokens-per-minute before the real usage is known

    def __init__(self, cfg: TranslatorConfig, api_key: Optional[str] = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        super().__init__(cfg, api_key, sleep, clock)
        self._pacer = TokenPacer(cfg.tokens_per_minute, clock=clock, sleep=sleep)
        self._last_estimate = 0

    def _url(self) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/chat/completions"

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"}

    def _body(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_completion_tokens": self.cfg.max_output_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "isl_gloss_results", "strict": True, "schema": to_json_schema(schema)}},
        }
        if "gpt-oss" in self.model:  # reasoning controls exist only on the reasoning models
            body["include_reasoning"] = False
            if self.cfg.reasoning_effort:
                body["reasoning_effort"] = self.cfg.reasoning_effort
        return body

    def _is_daily_quota(self, body: str, retry_after: Optional[float]) -> bool:
        message = self._short(body, 600)
        waits = parse_wait_seconds(message)
        long_wait = max(retry_after or 0, waits or 0) > 300  # a per-minute limit clears within a minute
        return bool(re.search(r"per day|\(RPD\)|\(TPD\)", message, re.I)) or long_wait

    def _retry_after(self, exc: urllib.error.HTTPError, body: str) -> Optional[float]:
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header and header.replace(".", "", 1).isdigit():
            return float(header)
        return parse_wait_seconds(self._short(body, 600))

    def _before_request(self, system: str, user: str) -> None:
        self._last_estimate = (len(system) + len(user)) // 4 + self.EXPECTED_OUTPUT_TOKENS
        self._pacer.wait(self._last_estimate)

    def _after_response(self, resp: LLMResponse) -> None:
        self._pacer.record(int(resp.usage.get("total_tokens") or self._last_estimate))

    def _parse(self, payload: dict) -> LLMResponse:
        choices = payload.get("choices") or []
        if not choices:
            raise BlockedResponse("The API returned no choices.")
        choice = choices[0]
        finish = choice.get("finish_reason")
        text = (choice.get("message") or {}).get("content") or ""
        if finish == "length":
            raise TruncatedResponse("Response hit max_completion_tokens (reasoning tokens count toward it).")
        if finish == "content_filter" and not text.strip():
            raise BlockedResponse("Response blocked by the content filter.")
        if not text.strip():
            raise BlockedResponse(f"Empty response (finish_reason={finish}).")
        usage = payload.get("usage") or {}
        return LLMResponse(text=text, model=payload.get("model") or self.model, finish_reason=finish,
                           usage={k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage})


def make_client(cfg: TranslatorConfig, api_key: Optional[str] = None) -> LLMClient:
    """The configured provider's client. No other provider is ever substituted silently."""
    if cfg.provider == "groq":
        return GroqClient(cfg, api_key)
    if cfg.provider == "gemini":
        return GeminiClient(cfg, api_key)
    raise ValueError(f"Unknown provider {cfg.provider!r}")


def parse_results_json(text: str) -> list[dict]:
    """Bounded repair of the model's reply: strip code fences, fall back to the outermost JSON object.
    Raises ValueError if nothing parses; never invents content."""
    candidates = [text.strip()]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    candidates.append(fenced)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        results = obj.get("results") if isinstance(obj, dict) else obj
        if isinstance(results, list) and all(isinstance(r, dict) for r in results):
            return results
    raise ValueError("model reply is not the expected JSON ({'results': [...]})")
