"""Issue detection. Produces flags only; this module never edits text."""

from __future__ import annotations

import re
from functools import lru_cache

from .config import CleaningConfig
from .models import Flag
from .normalize import MOJIBAKE_RE

_UNCLEAR_WORDS = (
    r"inaudible|unclear|unintelligible|indistinct|incomprehensible|garbled|crosstalk|cross-talk|"
    r"overlapping(?: speech| voices)?|illegible"
)
UNCLEAR_RE = re.compile(rf"\[\s*(?:{_UNCLEAR_WORDS})[^\]]*\]|\(\s*(?:{_UNCLEAR_WORDS})[^)]*\)|\[\s*\?+\s*\]", re.I)
SOUND_TAG_RE = re.compile(r"\[[^\]\n]{1,60}\]|♪[^♪]*♪?")
_WORD = r"[^\W\d_][\w'’]*"
_TERMINAL_RE = re.compile(r"[.!?…][\"'’”)\]]*$")


@lru_cache(maxsize=32)
def _watch_patterns(watchlist_key: tuple) -> list:
    return [(re.compile(p, 0 if cs else re.I), note) for p, note, cs in watchlist_key]


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < e and s < span[1] for s, e in taken)


def detect_flags(text: str, cfg: CleaningConfig) -> list[Flag]:
    flags: list[Flag] = []
    taken: list[tuple[int, int]] = []

    def add(code, start, end, note, review=False, track=False, skip=False):
        flags.append(Flag(code=code, text=text[start:end], start=start, end=end, note=note, review_required=review,
                          skip_translation=skip))
        if track:
            taken.append((start, end))

    # --- unclear audio / sound tags -------------------------------------------------------------------------
    for m in UNCLEAR_RE.finditer(text):
        add("unclear_audio", m.start(), m.end(),
            "Speech marked unclear by the transcript source. Preserved as written; missing words are not guessed.",
            review=True, track=True)
    for m in SOUND_TAG_RE.finditer(text):
        if not _overlaps((m.start(), m.end()), taken):
            add("non_speech_tag", m.start(), m.end(), "Bracketed non-speech tag; kept unless drop_sound_tags is on.")

    # --- repetition (flag only; never deleted by detection) -----------------------------------------------------
    allow = {w.casefold() for w in cfg.repeat_allowlist}
    word_spans: list[tuple[int, int]] = []
    for m in re.finditer(rf"(?<![\w'’])({_WORD})(?:\s+\1)+(?![\w'’])", text, re.I):
        if m.group(1).casefold() in allow:
            continue
        word_spans.append(m.span())
        add("repeated_word", m.start(), m.end(),
            "Immediate word repetition: may be a stutter, emphasis or genuine content. Kept.")
    for n in (4, 3, 2):
        pat = rf"(?<![\w'’])((?:[\w'’]+\s+){{{n - 1}}}[\w'’]+)\s+\1(?![\w'’])"
        for m in re.finditer(pat, text, re.I):
            if _overlaps(m.span(), word_spans):
                continue
            word_spans.append(m.span())
            add("repeated_phrase", m.start(), m.end(),
                "Repeated phrase: may be disfluency, emphasis or an ingestion artifact. Kept.")

    # --- suspicious tokens / terms ----------------------------------------------------------------------------
    for tok in cfg.suspect_tokens:
        for m in re.finditer(rf"(?<![\w'’]){re.escape(tok)}(?![\w'’])", text, re.I):
            add("expression_cue", m.start(), m.end(),
                f"'{m.group()}' is an ASR-captured sound/interjection. Kept in the text for the later "
                "facial-expression stage; marked so translation skips it.", skip=True)
    key = tuple((w["pattern"], w["note"], bool(w.get("case_sensitive", False))) for w in cfg.watchlist)
    for pat, note in _watch_patterns(key):
        for m in pat.finditer(text):
            add("skip_term", m.start(), m.end(), note, skip=True)

    # --- code-switching (romanised Gujarati/Hindi): mark each run of marker words to be skipped ----------------
    words = list(re.finditer(r"[A-Za-z']+", text))
    markers = {w.casefold() for w in cfg.non_english_markers}
    hit_idx = [k for k, m in enumerate(words) if m.group().casefold() in markers]
    clusters: list[list[int]] = []
    for k in hit_idx:
        if clusters and k - clusters[-1][-1] - 1 <= cfg.non_english_max_gap:
            clusters[-1].append(k)
        else:
            clusters.append([k])
    for cl in clusters:
        if len(cl) >= cfg.non_english_min_hits:
            add("skip_non_english", words[cl[0]].start(), words[cl[-1]].end(),
                "Looks like romanised Gujarati/Hindi. Kept as written; marked so translation skips it. "
                "Only the marker-word run is covered, so neighbouring non-English words may be missed.", skip=True)

    # --- encoding damage -----------------------------------------------------------------------------------------
    for m in re.finditer("�+", text):
        add("replacement_character", m.start(), m.end(), "Undecodable bytes in the source; original text unknown.",
            review=True)
    for m in MOJIBAKE_RE.finditer(text):
        add("possible_mojibake", m.start(), m.end() + 1, "Looks like encoding damage that could not be repaired safely.",
            review=True)

    # --- structure ----------------------------------------------------------------------------------------------------
    if len(words) <= cfg.short_chunk_words and not _TERMINAL_RE.search(text):
        add("short_fragment", 0, len(text), "Very short chunk without terminal punctuation; may be a fragment.")

    flags.sort(key=lambda f: (f.start, f.end, f.code))
    return flags
