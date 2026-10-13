"""Conservative, configurable cleaning rules. Each rule logs an EditDraft; the defaults change nothing risky."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import CleaningConfig
from .detect import SOUND_TAG_RE, UNCLEAR_RE

_WORD = r"[^\W\d_][\w'’]*"


@dataclass
class EditDraft:
    operation: str
    original: str
    cleaned: str
    reason: str
    confidence: str = "high"
    review_required: bool = False


@lru_cache(maxsize=16)
def _boilerplate(patterns: tuple) -> list:
    return [re.compile(p, re.I | re.S) for p in patterns]


@lru_cache(maxsize=16)
def _filler_re(words: tuple) -> re.Pattern:
    alt = "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))
    return re.compile(
        rf"(?<![\w'’-])({alt})(?![\w'’-])(?:\s*(?:,|\.\.\.|…|-|—))?\s*", re.I
    )


def clean_chunk_text(text: str, cfg: CleaningConfig) -> tuple[str, list[EditDraft]]:
    drafts: list[EditDraft] = []

    # 1. platform boilerplate: only when the WHOLE chunk is boilerplate
    for pat in _boilerplate(tuple(cfg.boilerplate_patterns)):
        if pat.fullmatch(text.strip()):
            drafts.append(EditDraft("remove_boilerplate", text, "", "Entire chunk matches a platform-boilerplate rule.",
                                    "high", False))
            return "", drafts

    # 2. bracketed sound tags (optional); unclear-audio markers are never removed
    if cfg.drop_sound_tags:
        def drop_tag(m: re.Match) -> str:
            if UNCLEAR_RE.fullmatch(m.group()):
                return m.group()
            drafts.append(EditDraft("drop_sound_tag", m.group(), "", "drop_sound_tags is enabled.", "high", False))
            return ""
        new = SOUND_TAG_RE.sub(drop_tag, text)
        if new != text:
            text = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", new)).strip()

    # 3. repeated words (optional, off by default): may be real speech, so every collapse needs review
    if cfg.collapse_repeated_words:
        allow = {w.casefold() for w in cfg.repeat_allowlist}

        def collapse(m: re.Match) -> str:
            if m.group(1).casefold() in allow:
                return m.group()
            drafts.append(EditDraft("collapse_repeated_word", m.group(), m.group(1),
                                    "collapse_repeated_words is enabled; repetition may have been intentional.",
                                    "medium", True))
            return m.group(1)
        text = re.sub(rf"(?<![\w'’])({_WORD})(?:\s+\1)+(?![\w'’])", collapse, text, flags=re.I)

    # 4. fillers (optional, off by default)
    if cfg.remove_fillers and cfg.filler_words:
        fre = _filler_re(tuple(cfg.filler_words))
        at_start_capitalised = False

        def drop_filler(m: re.Match) -> str:
            nonlocal at_start_capitalised
            if m.start() == 0 and m.group(1)[:1].isupper():
                at_start_capitalised = True
            drafts.append(EditDraft("remove_filler", m.group().strip(), "", "remove_fillers is enabled.", "high", False))
            return ""
        new = fre.sub(drop_filler, text)
        if new != text:
            new = re.sub(r"\s+([,.;:!?])", r"\1", new)
            new = re.sub(r",\s*([.!?])", r"\1", new)
            new = re.sub(r"^[,;\s]+", "", new).strip()
            if at_start_capitalised and new:
                new = new[0].upper() + new[1:]
            text = new

    return text, drafts
