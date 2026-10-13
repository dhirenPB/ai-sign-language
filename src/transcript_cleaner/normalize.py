"""Unicode / whitespace normalisation. Changes characters, never words."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# Exotic spaces -> ordinary space.
_SPACES = {ord(c): " " for c in "               　"}
# Invisible characters with no function in English text. ZWJ/ZWNJ (U+200D/U+200C) are deliberately kept
# because they are meaningful in Indic scripts that may be added later.
_INVISIBLE = {ord(c): None for c in "​⁠﻿­"}
_CONTROL_RE = re.compile("[\x00-\x08\x0e-\x1f\x7f-\x9f]")

# Typical UTF-8-read-as-cp1252 damage, e.g. "â€™" for an apostrophe or "Ã©" for "é".
MOJIBAKE_RE = re.compile(
    "[ÂÃâ][\u0080-¿ŒœŠšŸŽžƒˆ˜"
    "–—‘-„†-•…‰‹›€™]"
)


@dataclass
class NormalizeResult:
    text: str
    counts: Counter = field(default_factory=Counter)
    mojibake: Optional[tuple[str, str]] = None  # (before, after) when a repair was applied


def fix_mojibake(text: str) -> Optional[str]:
    if not MOJIBAKE_RE.search(text):
        return None
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None  # not a clean round-trip: leave it alone (detect.py will flag it)
    return repaired if repaired != text else None


def normalize_text(text: str) -> NormalizeResult:
    res = NormalizeResult(text=text)
    counts = res.counts

    repaired = fix_mojibake(text)
    if repaired is not None:
        res.mojibake = (text, repaired)
        counts["mojibake_repaired"] += 1
        text = repaired

    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        counts["unicode_nfc_normalized"] += 1
        text = nfc

    n = sum(1 for ch in text if ord(ch) in _SPACES)
    if n:
        counts["unicode_spaces_replaced"] += n
        text = text.translate(_SPACES)

    n = sum(1 for ch in text if ord(ch) in _INVISIBLE)
    if n:
        counts["invisible_chars_removed"] += n
        text = text.translate(_INVISIBLE)

    text, n = _CONTROL_RE.subn("", text)
    if n:
        counts["control_chars_removed"] += n

    text = text.strip()
    counts["line_breaks_joined"] += len(re.findall(r"\r\n|\r|\n", text))
    runs = [m.group() for m in re.finditer(r"\s+", text) if m.group() != " " and not re.search(r"[\r\n]", m.group())]
    counts["whitespace_collapsed"] += len(runs)
    text = re.sub(r"\s+", " ", text)

    res.text = text
    for key in [k for k, v in counts.items() if v == 0]:
        del counts[key]
    return res


def comparable_tokens(text: str) -> list[str]:
    """Word tokens used for the word-accounting check. Applies only the *invisible* normalisations so that
    stripping a zero-width space between two letters does not count as losing a word."""
    text = unicodedata.normalize("NFC", text).translate(_SPACES).translate(_INVISIBLE)
    return [t.casefold() for t in re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text)]
