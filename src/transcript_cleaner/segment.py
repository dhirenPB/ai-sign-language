"""Sentence / chunk segmentation. Works on raw text and returns (start, end) spans, so offsets stay exact."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_TERM_RE = re.compile(r"[.!?…]+[\"'’”)\]]*")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "e.g", "i.e", "fig", "approx", "dept", "inc", "ltd",
}


@dataclass
class Span:
    start: int
    end: int
    boundary_before: bool  # starts right after a sentence terminator (False for the first span / clause splits)
    split: Optional[str] = None  # "clause" | "forced": how it was separated from the previous span


_NEXT_CHAR_RE = re.compile(r"\s*(\S)")


def _is_boundary(text: str, m: re.Match) -> bool:
    end = m.end()
    if end < len(text) and not text[end].isspace():
        return False  # "3.5", "e.g.,", "U.S.A"
    term = m.group()
    nxt = _NEXT_CHAR_RE.match(text, end)  # match at a position: no slicing, so long texts stay linear
    if len(term.rstrip("\"'’”)]")) > 1 or term.startswith("…"):  # ellipsis
        return bool(nxt) and not nxt.group(1).islower()
    if term.startswith("."):
        i = m.start()
        while i > 0 and not text[i - 1].isspace():
            i -= 1
        word = text[i:m.start()].lstrip("\"'([‘“")
        if word.casefold() in _ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha() and word.isupper() and word != "I":
            return False  # an initial, as in "J. Young"
    return True


def sentence_spans(text: str) -> list[Span]:
    cuts = [m.end() for m in _TERM_RE.finditer(text) if _is_boundary(text, m)]
    spans: list[Span] = []
    cursor, after_boundary = 0, False
    for cut in cuts + [len(text)]:
        seg = text[cursor:cut]
        s = cursor + (len(seg) - len(seg.lstrip()))
        e = cursor + len(seg.rstrip())
        if e > s:
            spans.append(Span(s, e, boundary_before=after_boundary and bool(spans)))
            after_boundary = True
        cursor = cut
    return spans


def split_overlong(text: str, span: Span, max_words: int) -> list[Span]:
    """Split a span longer than max_words, preferring a clause punctuation mark, else a hard word boundary."""
    words = [(m.start() + span.start, m.end() + span.start) for m in re.finditer(r"\S+", text[span.start:span.end])]
    if len(words) <= max_words:
        return [span]
    out: list[Span] = []
    i, prev_kind = 0, None
    while len(words) - i > max_words:
        window = words[i:i + max_words]
        cut, kind = None, "forced"
        for j in range(len(window) - 1, max_words // 2 - 1, -1):
            if text[window[j][1] - 1] in ",;:—–":
                cut, kind = j, "clause"
                break
        if cut is None:
            cut = len(window) - 1
        out.append(Span(words[i][0], window[cut][1], boundary_before=span.boundary_before if not out else False,
                        split=prev_kind))
        prev_kind = kind
        i += cut + 1
    out.append(Span(words[i][0], words[-1][1], boundary_before=False, split=prev_kind))
    return out
