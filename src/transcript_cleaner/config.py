"""Cleaning configuration. Every behaviour that could change meaning is an explicit, default-safe option."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SOURCE_TYPES = ("youtube_prerecorded", "livestream", "unknown")

# Whole-chunk matches only (case-insensitive). A chunk is dropped only if it is *entirely* one of these.
DEFAULT_BOILERPLATE = (
    r"(?:auto-?generated|automatically generated|automatic captions?) (?:by|with|from) youtube.*",
    r"subtitles? (?:by|created by|provided by) .*",
    r"captions? (?:by|created by|provided by) .*",
    r"transcribed by .*",
    r"transcript (?:by|provided by) .*",
)

# Watchlist: known mis-recognised terms. Kept verbatim and marked skip_translation, never rewritten.
DEFAULT_WATCHLIST = (
    {
        "pattern": r"\bJane\b",
        "note": "Likely ASR error for the domain term 'Jain'. Kept as written; marked so translation skips it.",
    },
    {
        "pattern": r"\b(?:Janindra|Jinindra|Jhinendra|Jin Indra|Jai Jin)\b",
        "note": "Likely mis-hearing of the greeting 'Jai Jinendra'. Kept as written; marked so translation skips it.",
    },
)

# Romanised Gujarati/Hindi function words that are not English words. Runs of them are marked skip_translation.
DEFAULT_NON_ENGLISH_MARKERS = (
    "hu", "tu", "tara", "taro", "tari", "tamne", "amne", "mane", "nathi", "chhu", "chhe", "che", "jya", "tya",
    "kem", "ke", "aavu", "aaje", "aavshe", "evi", "jevo", "ane", "shu", "kai", "kaik", "divas", "ek", "vishwas",
    "hoish", "sarkha", "hashu", "nahi", "nahin", "hai", "hain", "mera", "tera", "kya", "aur", "bhi", "toh", "kuch", "koi", "hum",
    "tum", "aap", "kyun", "lekin", "bahut", "yeh", "woh",
)

# Interjections / sound-like tokens that ASR often emits for non-speech audio (chanting, humming, whistling).
DEFAULT_SUSPECT_TOKENS = ("hmm", "hm", "mm", "mmm", "ooh", "whistle")

# Immediate word repeats that are normal English and should not be flagged as stutters.
DEFAULT_REPEAT_ALLOWLIST = ("had", "that", "is", "very", "bye", "ha", "no", "so", "knock", "far", "never")


@dataclass
class CleaningConfig:
    language: str = "en"
    source_type: str = "unknown"

    # --- parsing -----------------------------------------------------------------------------------
    speaker_label_pattern: str = r"Speaker\s+\d+"
    header_keys: tuple = ("source", "title", "url", "channel", "date", "language", "uploader", "description", "duration")
    dedupe_rolling_captions: bool = True

    # --- cleaning (all risky options default OFF) -----------------------------------------------------
    remove_fillers: bool = True  # fillers are handled later through facial expression; removals stay in the edit log
    filler_words: tuple = ("um", "uh", "uhm", "er", "erm")
    collapse_repeated_words: bool = False
    repeat_allowlist: tuple = DEFAULT_REPEAT_ALLOWLIST
    drop_sound_tags: bool = False
    boilerplate_patterns: tuple = DEFAULT_BOILERPLATE

    # --- segmentation ------------------------------------------------------------------------------------
    max_chunk_words: int = 60

    # --- detection ---------------------------------------------------------------------------------------
    short_chunk_words: int = 2
    suspect_tokens: tuple = DEFAULT_SUSPECT_TOKENS
    watchlist: tuple = DEFAULT_WATCHLIST
    non_english_markers: tuple = DEFAULT_NON_ENGLISH_MARKERS
    non_english_min_hits: int = 2  # markers needed in one cluster
    non_english_max_gap: int = 4  # max non-marker words allowed between markers of one cluster

    # --- validation --------------------------------------------------------------------------------------
    loss_warning_ratio: float = 0.05  # warn if more than this share of non-space characters disappears

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CleaningConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            kwargs[key] = tuple(value) if isinstance(value, list) else value
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "CleaningConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def validate(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SOURCE_TYPES}, got {self.source_type!r}")
        if self.max_chunk_words < 5:
            raise ValueError("max_chunk_words must be >= 5")
