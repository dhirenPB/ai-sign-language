"""Integrity checks run after cleaning and segmentation (requirements section 9)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from .config import CleaningConfig
from .ingest import timestamp_to_seconds
from .models import Chunk, Edit, ParsedSource
from .normalize import comparable_tokens

_TIME_RE = re.compile(r"^\d{2,}:\d{2}:\d{2}\.\d{3}$")


def _issue(code: str, message: str, chunk_id: Optional[str] = None) -> dict:
    d = {"code": code, "message": message}
    if chunk_id:
        d["chunk_id"] = chunk_id
    return d


def validate_chunks(ps: ParsedSource, chunks: list[Chunk], edits: list[Edit], cfg: CleaningConfig) -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []

    if not chunks:
        errors.append(_issue("no_chunks", "The pipeline produced no text chunks (empty or metadata-only input)."))
        return {"passed": False, "errors": errors, "warnings": warnings}

    # ---- identifiers, order, metadata leakage ----------------------------------------------------------------
    seen_ids = set()
    prev_end_off = -1
    prev_start_t: Optional[float] = None
    speakers = {p.speaker_id for p in ps.pieces}
    label_re = re.compile(rf"(?<!\S)(?:{cfg.speaker_label_pattern})\s*:|(?<!\S)>>|-->|\d{{1,2}}:\d{{2}}:\d{{2}}[.,]\d{{3}}")
    for c in chunks:
        if c.chunk_id in seen_ids:
            errors.append(_issue("duplicate_chunk_id", "Chunk id is not unique.", c.chunk_id))
        seen_ids.add(c.chunk_id)
        if not c.text.strip():
            errors.append(_issue("empty_chunk", "Chunk text is empty.", c.chunk_id))
        if re.search(r"\s{2,}|[\r\n\t]", c.text):
            errors.append(_issue("unnormalised_whitespace", "Chunk contains repeated whitespace or line breaks.", c.chunk_id))
        if label_re.search(c.text):
            errors.append(_issue("metadata_in_text", "Speaker label, '>>' or timestamp found inside spoken text.", c.chunk_id))
        if c.speaker_id not in speakers:
            errors.append(_issue("unknown_speaker", f"Speaker {c.speaker_id!r} was never present in the source.", c.chunk_id))

        so, eo = c.source_start_offset, c.source_end_offset
        if (so is None) != (eo is None):
            errors.append(_issue("offset_pair", "Only one of start/end offset is set.", c.chunk_id))
        elif so is not None:
            if not (0 <= so < eo <= len(ps.raw_text)):
                errors.append(_issue("offset_range", f"Offsets {so}-{eo} outside the source (length {len(ps.raw_text)}).", c.chunk_id))
            elif so < prev_end_off:
                errors.append(_issue("offset_order", "Chunk overlaps or precedes the previous chunk in the source.", c.chunk_id))
            prev_end_off = max(prev_end_off, eo)
            has_edit = any("[edit " in n for n in c.edit_notes)
            if not has_edit and comparable_tokens(ps.raw_text[so:eo]) != comparable_tokens(c.text):
                errors.append(_issue("offset_content_mismatch", "Source slice at the stated offsets does not hold this chunk's words.", c.chunk_id))

        for label, t in (("start_time", c.start_time), ("end_time", c.end_time)):
            if t is not None and not _TIME_RE.match(t):
                errors.append(_issue("invalid_timestamp", f"{label} {t!r} is not HH:MM:SS.mmm.", c.chunk_id))
        if c.start_time and c.end_time and _TIME_RE.match(c.start_time) and _TIME_RE.match(c.end_time):
            if timestamp_to_seconds(c.end_time) < timestamp_to_seconds(c.start_time):
                warnings.append(_issue("end_before_start", "end_time precedes start_time (kept as in the source).", c.chunk_id))
        if c.start_time and _TIME_RE.match(c.start_time):
            s = timestamp_to_seconds(c.start_time)
            if prev_start_t is not None and s + 1e-9 < prev_start_t:
                warnings.append(_issue("timestamps_out_of_order", "Chunk starts earlier than the previous chunk.", c.chunk_id))
            prev_start_t = s if prev_start_t is None else max(prev_start_t, s)

        for tok in re.findall(r"\S+", c.text):
            if len(tok) > 40 and not re.match(r"https?://|www\.", tok, re.I):
                warnings.append(_issue("possible_word_concatenation", f"Unusually long token ({len(tok)} chars): {tok[:30]}...", c.chunk_id))
                break

    # ---- duplicates --------------------------------------------------------------------------------------------
    for a, b in zip(chunks, chunks[1:]):
        if a.text.casefold() == b.text.casefold() and a.speaker_id == b.speaker_id and len(a.text.split()) >= 3:
            warnings.append(_issue("duplicate_consecutive_chunk", f"Identical to the previous chunk ({a.chunk_id}).", b.chunk_id))

    # ---- word accounting: source words - logged removals + logged additions == chunk words ------------------
    expected = Counter(t for p in ps.pieces for t in comparable_tokens(p.text))
    for e in edits:
        if e.stage in ("normalize", "clean"):
            expected.subtract(comparable_tokens(e.original))
            expected.update(comparable_tokens(e.cleaned))
    actual = Counter(t for c in chunks for t in comparable_tokens(c.text))
    diff = {t: actual[t] - expected[t] for t in set(actual) | set(expected) if actual[t] != expected[t]}
    if diff:
        sample = ", ".join(f"{t}({d:+d})" for t, d in sorted(diff.items())[:8])
        errors.append(_issue("word_accounting_mismatch",
                             f"Words lost or duplicated without a matching log entry: {sample}"))

    # ---- suspiciously large text loss ------------------------------------------------------------------------------
    raw_chars = sum(len(re.sub(r"\s", "", p.text)) for p in ps.pieces)
    out_chars = sum(len(re.sub(r"\s", "", c.text)) for c in chunks)
    if raw_chars and (raw_chars - out_chars) / raw_chars > cfg.loss_warning_ratio:
        warnings.append(_issue("large_text_loss", f"{raw_chars - out_chars} of {raw_chars} non-space characters removed "
                                                  f"({(raw_chars - out_chars) / raw_chars:.1%}); see the edit log."))

    return {"passed": not errors, "errors": errors, "warnings": warnings}
