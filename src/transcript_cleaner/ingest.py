"""Format-specific parsing (TXT / SRT / VTT) into a common list of Pieces.

Timestamps, cue numbers, speaker labels and header fields are metadata. They never appear in Piece.text.
Anything removed at this stage is logged as an Edit (stage="ingest").
"""

from __future__ import annotations

import codecs
import html
import re
from collections import Counter
from typing import Optional

from .config import CleaningConfig
from .models import Edit, ParsedSource, Piece

SUPPORTED_SUFFIXES = (".txt", ".srt", ".vtt")

_TS = r"(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_TS_RE = re.compile(rf"^\s*{_TS}\s*$")
_CUE_LINE_RE = re.compile(rf"^\s*(?P<start>{_TS})\s*-->\s*(?P<end>{_TS})(?P<settings>.*)$")
_TAG_RE = re.compile(r"<[^>]*>")
_ASS_RE = re.compile(r"\{\\[^}]*\}")
_VOICE_RE = re.compile(r"^\s*<v(?:\.[\w.]+)?\s+([^>]+)>")
_VIDEO_ID_RE = re.compile(r"(?:[?&]v=|youtu\.be/|/live/|/shorts/|/embed/)([\w-]{11})")


# --------------------------------------------------------------------------------------------------------
# decoding / format detection
# --------------------------------------------------------------------------------------------------------
def decode_bytes(data: bytes) -> tuple[str, str, list[dict[str, str]]]:
    """Decode raw bytes. Returns (text, encoding_name, warnings). The BOM, if any, is not part of the text."""
    warnings: list[dict[str, str]] = []
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8):].decode("utf-8"), "utf-8-sig", warnings
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16"), "utf-16", warnings
    try:
        return data.decode("utf-8"), "utf-8", warnings
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
        warnings.append({
            "code": "encoding_fallback",
            "message": "File is not valid UTF-8; decoded as cp1252. Check the text for encoding damage.",
        })
        bad = text.count("�")
        if bad:
            warnings.append({
                "code": "undecodable_bytes",
                "message": f"{bad} byte(s) could not be decoded and appear as U+FFFD in the text.",
            })
        return text, "cp1252 (fallback)", warnings


def detect_format(filename: str, text: str) -> str:
    lower = filename.lower()
    first = text.lstrip("﻿ \t\r\n")[:6]
    if first.upper().startswith("WEBVTT") or lower.endswith(".vtt"):
        return "vtt"
    if lower.endswith(".srt"):
        return "srt"
    return "txt"


# --------------------------------------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------------------------------------
def canonical_timestamp(raw: str) -> Optional[str]:
    """Return HH:MM:SS.mmm, or None if the value is not a valid timestamp. Never guesses."""
    m = _TS_RE.match(raw)
    if not m:
        return None
    hours, minutes, seconds, frac = m.groups()
    if int(minutes) > 59 or int(seconds) > 59:
        return None
    millis = frac.ljust(3, "0")
    return f"{int(hours or 0):02d}:{int(minutes):02d}:{int(seconds):02d}.{millis}"


def timestamp_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


# --------------------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------------------
def parse(data: bytes, filename: str, source_id: str, cfg: CleaningConfig) -> ParsedSource:
    text, encoding, enc_warnings = decode_bytes(data)
    fmt = detect_format(filename, text)
    ps = ParsedSource(source_id=source_id, filename=filename, fmt=fmt, raw_text=text, encoding=encoding)
    ps.warnings.extend(enc_warnings)
    marker_re = _marker_regex(cfg)
    if fmt == "txt":
        _parse_txt(ps, cfg, marker_re)
    else:
        _parse_cues(ps, cfg, marker_re, fmt)
    ps.metadata["source_language"] = cfg.language
    return ps


def _marker_regex(cfg: CleaningConfig) -> re.Pattern:
    # group 1: a speaker label ("Speaker 3"); group 2: a bare ">>" turn-change marker (speaker unknown)
    return re.compile(rf"(?<!\S)(?:({cfg.speaker_label_pattern})\s*:|(>>))")


def _split_on_markers(text, start, end, speaker, marker_re):
    """Split text[start:end] at speaker markers. Yields (speaker, seg_start, seg_end, explicit)."""
    segments = []
    cursor, current, explicit = start, speaker, False
    for m in marker_re.finditer(text, start, end):
        segments.append((current, cursor, m.start(), explicit))
        current = re.sub(r"\s+", " ", m.group(1)) if m.group(1) else None
        cursor, explicit = m.end(), True
    segments.append((current, cursor, end, explicit))
    return segments


# --------------------------------------------------------------------------------------------------------
# plain text
# --------------------------------------------------------------------------------------------------------
def _parse_txt(ps: ParsedSource, cfg: CleaningConfig, marker_re: re.Pattern) -> None:
    text = ps.raw_text
    lines = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\n]*\n|[^\n]+\Z", text)]

    # 1) header block: contiguous "Key: value" lines at the very top
    header_re = re.compile(rf"^\s*({'|'.join(re.escape(k) for k in cfg.header_keys)})\s*:\s*(.*?)\s*$", re.I)
    idx = 0
    while idx < len(lines):
        m = header_re.match(lines[idx][2].rstrip("\r\n"))
        if not m:
            break
        key, value = m.group(1).lower(), m.group(2)
        _record_header(ps, key, value)
        ps.edits.append(Edit(
            stage="ingest", operation="extract_header_metadata", original=lines[idx][2].strip(), cleaned="",
            reason="Header field kept as metadata; it is not spoken text.",
            source_start_offset=lines[idx][0], source_end_offset=lines[idx][1],
        ))
        ps.counts["header_fields_extracted"] += 1
        idx += 1

    # 2) paragraphs (blank-line separated) -> pieces, split at speaker labels
    speaker: Optional[str] = None
    turn = -1
    para_start: Optional[int] = None
    para_end = 0

    def flush(p_start: Optional[int], p_end: int) -> None:
        nonlocal speaker, turn
        if p_start is None:
            return
        for spk, s, e, explicit in _split_on_markers(text, p_start, p_end, speaker, marker_re):
            if explicit:
                ps.counts["speaker_labels_extracted"] += 1
            raw_seg = text[s:e]
            stripped = raw_seg.strip()
            speaker = spk
            if not stripped:
                if explicit:
                    ps.counts["empty_turns_skipped"] += 1
                continue
            s2 = s + (len(raw_seg) - len(raw_seg.lstrip()))
            e2 = s2 + len(stripped)
            turn += 1
            ps.pieces.append(Piece(text=text[s2:e2], speaker_id=spk, turn_index=turn, raw_start=s2, raw_end=e2))

    for ls, le, line in lines[idx:]:
        if line.strip() == "":
            flush(para_start, para_end)
            para_start = None
        else:
            if para_start is None:
                para_start = ls
            para_end = le
    flush(para_start, para_end)

    if ps.counts["speaker_labels_extracted"]:
        ps.warn("speaker_labels_anonymous",
                "Speaker labels are kept exactly as written (e.g. 'Speaker 1'); they are not mapped to real identities.")


def _record_header(ps: ParsedSource, key: str, value: str) -> None:
    if key in ("source", "url"):
        if re.match(r"https?://", value, re.I):
            ps.metadata["source_url"] = value
            m = _VIDEO_ID_RE.search(value)
            if m:
                ps.metadata["video_id"] = m.group(1)
        else:
            ps.metadata["source_label"] = value
    else:
        ps.metadata[key] = value


# --------------------------------------------------------------------------------------------------------
# SRT / VTT
# --------------------------------------------------------------------------------------------------------
def _parse_cues(ps: ParsedSource, cfg: CleaningConfig, marker_re: re.Pattern, fmt: str) -> None:
    text = ps.raw_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n+", text.strip("\n"))
    speaker: Optional[str] = None
    turn = -1
    cue_no = 0
    prev_lines: list[str] = []

    for block in blocks:
        lines = block.split("\n")
        if fmt == "vtt" and re.match(r"\s*(WEBVTT|NOTE|STYLE|REGION)\b", lines[0], re.I):
            ps.counts["vtt_non_cue_blocks_skipped"] += 1
            continue
        time_idx = next((i for i, ln in enumerate(lines) if _CUE_LINE_RE.match(ln)), None)
        if time_idx is None:
            ps.counts["non_cue_blocks_skipped"] += 1
            if block.strip():
                ps.warn("unparsed_block", f"Block without a timing line was skipped: {block.strip()[:60]!r}")
            continue
        cue_no += 1
        m = _CUE_LINE_RE.match(lines[time_idx])
        start = canonical_timestamp(m.group("start"))
        end = canonical_timestamp(m.group("end"))
        cue_id = " ".join(x.strip() for x in lines[:time_idx] if x.strip()) or None
        if cue_id:
            ps.edits.append(Edit(stage="ingest", operation="extract_cue_identifier", original=cue_id, cleaned="",
                                 reason="Cue number kept as metadata; it is not spoken text."))
        if start is None or end is None:
            ps.warn("invalid_timestamp", f"Cue {cue_no} has an invalid timestamp; it is set to null (not guessed).")
        elif timestamp_to_seconds(end) < timestamp_to_seconds(start):
            ps.warn("end_before_start", f"Cue {cue_no} ends before it starts; timestamps kept as written.")

        cue_lines = _clean_cue_lines(lines[time_idx + 1:], ps)
        if cfg.dedupe_rolling_captions:
            cue_lines = _drop_rolling_duplicates(cue_lines, prev_lines, start, end, ps, cue_no)
        prev_lines = _clean_cue_lines(lines[time_idx + 1:], None) or prev_lines
        if not cue_lines:
            ps.counts["empty_cues_skipped"] += 1
            continue

        voice = _VOICE_RE.match(lines[time_idx + 1]) if len(lines) > time_idx + 1 else None
        joined = " ".join(cue_lines)
        if voice:
            speaker, turn = voice.group(1).strip(), turn + 1
            ps.counts["speaker_labels_extracted"] += 1
        for spk, s, e, explicit in _split_on_markers(joined, 0, len(joined), speaker, marker_re):
            if explicit:
                ps.counts["speaker_labels_extracted"] += 1
                turn += 1
            speaker = spk
            piece_text = joined[s:e].strip()
            if not piece_text:
                if explicit:
                    ps.counts["empty_turns_skipped"] += 1
                continue
            if turn < 0:
                turn = 0
            ps.pieces.append(Piece(text=piece_text, speaker_id=spk, turn_index=turn, start_time=start,
                                   end_time=end, cue_index=cue_no, cue_id=cue_id))

    ps.counts["cues"] = cue_no
    ps.warn("offsets_unavailable",
            "Cue formats have no stable character offsets; chunks carry cue_indices/cue_ids and timestamps instead.")
    if ps.counts["speaker_labels_extracted"]:
        ps.warn("speaker_labels_anonymous", "Speaker labels are kept exactly as written; no identities are inferred.")


def _clean_cue_lines(raw_lines: list[str], ps: Optional[ParsedSource]) -> list[str]:
    out = []
    for ln in raw_lines:  # <v Name> voice tags are read by the caller before they are stripped here
        without_tags = _ASS_RE.sub("", _TAG_RE.sub("", ln))
        if ps is not None and without_tags != ln:
            ps.counts["markup_tags_removed"] += 1
        unescaped = html.unescape(without_tags)
        if ps is not None and unescaped != without_tags:
            ps.counts["html_entities_unescaped"] += 1
        cleaned = re.sub(r"\s+", " ", unescaped).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _drop_rolling_duplicates(cue_lines, prev_lines, start, end, ps, cue_no):
    """YouTube auto-captions repeat the previous cue's last line as this cue's first line, plus near-zero-duration
    stub cues that only repeat it. Both are ingestion artifacts. A lone repeated line in a normal cue is kept."""
    if not cue_lines or not prev_lines:
        return cue_lines
    is_stub = (start is not None and end is not None and timestamp_to_seconds(end) - timestamp_to_seconds(start) <= 0.05)
    drop = 0
    if len(cue_lines) >= 2 and cue_lines[0] == prev_lines[-1]:
        drop = 1
    elif is_stub and cue_lines == prev_lines[-len(cue_lines):]:
        drop = len(cue_lines)
    if not drop:
        return cue_lines
    removed = " ".join(cue_lines[:drop])
    ps.edits.append(Edit(
        stage="ingest", operation="remove_rolling_caption_duplicate", original=removed, cleaned="",
        reason=f"Cue {cue_no} repeated the previous cue's line (rolling-caption artifact).", confidence="high",
    ))
    ps.counts["rolling_duplicate_lines_removed"] += drop
    return cue_lines[drop:]
