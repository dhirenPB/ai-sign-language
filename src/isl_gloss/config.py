"""Translator configuration. Secrets are never stored here: the API key is read from an environment variable."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional


# Per-provider defaults. The dataclass defaults below are the Groq ones (the project default); for_provider() swaps them.
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "groq": {
        "provider": "groq", "model": "openai/gpt-oss-120b", "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1", "requests_per_minute": 20.0, "tokens_per_minute": 7000,
    },
    # gemini-3.6-flash is listed as free of charge on Google's pricing page; gemini-2.5-flash now answers HTTP 404
    # "no longer available to new users".
    "gemini": {
        "provider": "gemini", "model": "gemini-3.6-flash", "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta", "requests_per_minute": 5.0, "tokens_per_minute": None,
    },
}


@dataclass
class TranslatorConfig:
    # --- LLM: Groq + openai/gpt-oss-120b by default (OpenAI-compatible chat completions with strict JSON-schema output) ---
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    api_key_env: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0
    reasoning_effort: Optional[str] = "low"  # Groq gpt-oss only (low | medium | high); reasoning tokens count toward limits
    thinking_budget: Optional[int] = 0  # Gemini 2.5-series only; set null in a config file to omit the field
    max_output_tokens: int = 8192  # Groq: max_completion_tokens (includes reasoning tokens)
    prompt_version: str = "7.0"  # see prompt.PROMPTS: "7.0" is current; "6.0" and "5.0" are earlier prompts kept for comparison
    request_timeout_seconds: float = 120.0

    # --- free-tier friendliness: conservative pacing, bounded retries, batching ---
    requests_per_minute: float = 20.0
    tokens_per_minute: Optional[int] = 7000  # Groq's free gpt-oss-120b limit is reported as 8K TPM; keep below it. null = no token pacing
    max_attempts: int = 5
    backoff_base_seconds: float = 5.0
    backoff_max_seconds: float = 60.0
    batch_size: int = 10
    max_batch_chars: int = 3000
    context_chunks: int = 0  # earlier sentences sent as read-only context; 0 = the model sees only the sentences to translate

    # --- stage 2: sentence chunking ---
    runon_min_words: int = 14  # split unpunctuated chunks at capitalised openers only if at least this long; 0 = never

    # --- stage 3: context-based grouping (limits are per group; a group is one LLM request) ---
    group_max_sentences: int = 8
    group_max_chars: int = 1200
    group_max_tokens: int = 300  # estimated as characters / 4 (no tokenizer dependency)
    group_max_gap_seconds: Optional[float] = 8.0  # start a new group after a longer pause; null = ignore timing
    group_break_on_turn_change: bool = True  # a new speaker turn / paragraph starts a new group
    context_sentences_before: int = 1  # read-only context from the previous group, only when the group's first sentence depends on it
    context_sentences_after: int = 1  # read-only context from the next group, only when the group ends mid-thought

    # --- prompt 8.0+ only (earlier prompts ignore these, so their cached results stay valid) ---
    empty_gloss_retry: bool = True  # one targeted re-ask when a sentence comes back with an empty gloss; never rewrites the gloss itself
    references_file: Optional[str] = None  # reviewed ISL reference entries (references/reference_set.json format); null = none
    reference_max_entries: int = 5  # cap per category on entries retrieved for one request

    # --- rules / validation (the prompt version lives in prompt.py) ---
    enable_rules: tuple = ()  # extra rule ids to switch on (e.g. "R-COP-01"); defaults are in rules.py
    disable_rules: tuple = ()
    vocabulary_file: Optional[str] = None  # optional: one allowed gloss token per line; unknown tokens get flagged
    exclude_flag_codes: tuple = ("unclear_audio", "non_speech_tag")  # cleaner flags whose spans are never translated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def for_provider(cls, provider: str) -> "TranslatorConfig":
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"Unknown provider {provider!r}. Choose one of {sorted(PROVIDER_DEFAULTS)}.")
        return replace(cls(), **PROVIDER_DEFAULTS[provider])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslatorConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        base = cls.for_provider(data.get("provider", cls().provider))  # provider's own defaults first, then the file's values
        cfg = replace(base, **{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "TranslatorConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def validate(self) -> None:
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"Unknown provider {self.provider!r}. Choose one of {sorted(PROVIDER_DEFAULTS)}.")
        from .prompt import PROMPTS
        if self.prompt_version not in PROMPTS:
            raise ValueError(f"Unknown prompt_version {self.prompt_version!r}. Available: {sorted(PROMPTS)}")
        if self.reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("reasoning_effort must be low, medium, high or null")
        if self.tokens_per_minute is not None and self.tokens_per_minute < 500:
            raise ValueError("tokens_per_minute must be >= 500 (or null)")
        if self.batch_size < 1 or self.max_attempts < 1 or self.requests_per_minute <= 0:
            raise ValueError("batch_size, max_attempts must be >= 1 and requests_per_minute > 0")
        if self.reference_max_entries < 1:
            raise ValueError("reference_max_entries must be >= 1")
        if self.context_chunks < 0:
            raise ValueError("context_chunks must be >= 0")
        if self.group_max_sentences < 1 or self.group_max_chars < 20 or self.group_max_tokens < 5:
            raise ValueError("group_max_sentences >= 1, group_max_chars >= 20 and group_max_tokens >= 5 are required")
        if self.context_sentences_before < 0 or self.context_sentences_after < 0 or self.runon_min_words < 0:
            raise ValueError("context_sentences_* and runon_min_words must be >= 0")
