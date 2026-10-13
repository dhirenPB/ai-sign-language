"""Stage 2: sentence-level chunking.

Input: the cleaner's structured JSON. The cleaner already splits at sentence punctuation (abbreviation, initial, decimal
and ellipsis aware), so most of its chunks ARE sentences and are carried over unchanged. What this stage adds:

* a stable `sentence_id` (`<source_id>-s0001`, in transcript order) with the spec'd fields;
* a second look at long chunks that have NO internal punctuation (typical ASR run-ons such as "Hey Prabhu If you had not
  grasped this land If you had not taught us ..."): they are split before a capitalised sentence-opening word
  (If, But, They, ...) instead of at a word-count interval. Sentence text is only ever sliced, never rewritten;
* per-sentence dependency signals (does this sentence need the previous one?) that the grouping stage uses so related
  sentences stay together. Signals are deterministic, documented heuristics, not a parse.

Timestamps and speakers are copied through when present and stay None otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import TranslatorConfig

# Capitalised words that start a sentence/clause but are rarely capitalised mid-sentence in normal English.
# "I", "The", "A" are deliberately absent: they are capitalised (or common) anywhere, so they prove nothing.
_OPENERS = {
    "If", "But", "When", "Whenever", "Because", "So", "Then", "Now", "Today", "And", "Or", "They", "We", "You", "He",
    "She", "It", "This", "That", "These", "Those", "There", "Here", "Why", "What", "How", "Where", "Who", "Which",
    "Although", "However", "Also", "Once", "While", "Since", "Before", "After", "Every", "Each", "Let", "Please",
}
_TERMINAL = re.compile(r"[.!?…][\"'’”)\]]*$")
_SOFT_END = re.compile(r"[,;:—–-]$")
_CONTINUATION = {"and", "but", "or", "so", "because", "which", "then", "also", "however", "therefore", "yet", "nor", "plus"}
_ANAPHORIC = {"he", "she", "it", "they", "them", "this", "that", "these", "those", "such", "its", "his", "her", "their",
              "him", "hers", "theirs"}
MIN_SEGMENT_WORDS = 3


@dataclass
class Sentence:
    sentence_id: str
    source_text: str
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    speaker_id: Optional[str] = None
    turn_index: Optional[int] = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    skip_spans: list[dict] = field(default_factory=list)  # cleaner's skip_translation_spans, re-based to this sentence
    excluded_spans: list[dict] = field(default_factory=list)  # [inaudible]/[Music]-style spans, re-based
    incomplete: bool = False  # no terminal punctuation, or ends in , ; : (the thought may continue)
    depends_on_previous: bool = False
    dependency_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "source_text": self.source_text,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "speaker_id": self.speaker_id,
            "turn_index": self.turn_index,
            "incomplete": self.incomplete,
            "depends_on_previous": self.depends_on_previous,
            "dependency_reasons": list(self.dependency_reasons),
            "skip_spans": self.skip_spans,
            "excluded_spans": self.excluded_spans,
            "source_metadata": self.source_metadata,
        }


# --------------------------------------------------------------------------------------------------------------
def _inside(pos: int, spans: list[dict]) -> bool:
    return any(s["start"] < pos < s["end"] for s in spans)


def split_run_on(text: str, protected: list[dict], min_words: int) -> list[tuple[int, int]]:
    """Return (start, end) slices of `text`. Only splits when the text is long (>= min_words), has no sentence
    punctuation inside, and has capitalised openers in the middle. Never cuts inside a protected (skip/excluded) span."""
    words = list(re.finditer(r"\S+", text))
    if min_words <= 0 or len(words) < min_words:
        return [(0, len(text))]
    cuts: list[int] = []
    last = 0
    for i, w in enumerate(words):
        if i - last < MIN_SEGMENT_WORDS or len(words) - i < MIN_SEGMENT_WORDS:
            continue
        token = w.group().strip("\"'([‘“").rstrip(",;:")  # "Today," is still the opener "Today"
        if token not in _OPENERS:
            continue
        if _TERMINAL.search(words[i - 1].group()) or _SOFT_END.search(words[i - 1].group()):
            continue  # already punctuated: not a missing-punctuation case
        if _inside(w.start(), protected):
            continue
        cuts.append(w.start())
        last = i
    if not cuts:
        return [(0, len(text))]
    edges = [0] + cuts + [len(text)]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def _rebase(spans: list[dict], start: int, end: int, text: str) -> list[dict]:
    out = []
    for s in spans:
        if s["start"] >= start and s["end"] <= end:
            out.append({**s, "start": s["start"] - start, "end": s["end"] - start})
    for s in out:  # keep spans consistent with the sliced text even after .strip()
        assert text[s["start"]:s["end"]] == s["text"], "span no longer matches text after slicing"
    return out


def dependency_reasons(text: str, previous: Optional["Sentence"]) -> list[str]:
    """Why this sentence probably needs the previous one. Empty list = it can stand alone (or there is no previous)."""
    if previous is None:
        return []
    reasons = []
    first = re.search(r"[A-Za-zÀ-ɏ']+", text)
    if first:
        word = first.group()
        if word[0].islower():
            reasons.append("lowercase_start")
        if word.lower() in _CONTINUATION:
            reasons.append("continuation_word")
        if word.lower() in _ANAPHORIC:
            reasons.append("pronoun_start")
    if previous.incomplete:
        reasons.append("previous_incomplete")
    if len(re.findall(r"\w+", text)) <= 2 and _TERMINAL.search(text):
        reasons.append("very_short")
    return reasons


def build_sentences(doc: dict, cfg: TranslatorConfig, excluded_flag_codes: Optional[tuple] = None) -> list[Sentence]:
    """cleaned-transcript JSON -> ordered Sentence objects."""
    source_id = doc["source_id"]
    exclude = set(excluded_flag_codes if excluded_flag_codes is not None else cfg.exclude_flag_codes)
    sentences: list[Sentence] = []
    for ch in doc["chunks"]:
        text = ch["text"]
        skip = list(ch.get("skip_translation_spans", []))
        excl = [{"start": f["start"], "end": f["end"], "text": f["text"], "reason": f["code"]}
                for f in ch.get("uncertainty_flags", []) if f["code"] in exclude]
        pieces = split_run_on(text, skip + excl, cfg.runon_min_words)
        for k, (s, e) in enumerate(pieces):
            raw = text[s:e]
            lead = len(raw) - len(raw.lstrip())
            s2, e2 = s + lead, s + lead + len(raw.strip())
            piece = text[s2:e2]
            split = len(pieces) > 1
            sent = Sentence(
                sentence_id=f"{source_id}-s{len(sentences) + 1:04d}",
                source_text=piece,
                start_timestamp=ch.get("start_time"),
                end_timestamp=ch.get("end_time"),
                speaker_id=ch.get("speaker_id"),
                turn_index=ch.get("turn_index"),
                skip_spans=_rebase(skip, s2, e2, piece),
                excluded_spans=_rebase(excl, s2, e2, piece),
                source_metadata={
                    "source_id": source_id,
                    "parent_chunk_id": ch["chunk_id"],
                    "split_method": "runon_capital_opener" if split else None,
                    "piece": [k + 1, len(pieces)],
                    "source_start_offset": ch.get("source_start_offset") if not split else None,
                    "source_end_offset": ch.get("source_end_offset") if not split else None,
                    "cue_indices": ch.get("cue_indices"),
                    "cue_ids": ch.get("cue_ids"),
                },
            )
            sent.incomplete = (not _TERMINAL.search(piece)) or bool(_SOFT_END.search(piece))
            reasons = dependency_reasons(piece, sentences[-1] if sentences else None)
            sent.dependency_reasons = reasons
            sent.depends_on_previous = bool(reasons)
            sentences.append(sent)
    return sentences
