"""Phase D: the structured transcript. This JSON is the CANONICAL output of transcription; the plain text and the VTT are
derived from it and never replace it. All times are seconds from the start of the original video.

Segment ids (`seg-0001`, ...) are written into the VTT as cue ids, so the existing cleaner records which cue(s) every cleaned
chunk came from; that is how cleaned text, sentences and glosses are mapped back to the original timeline.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .errors import InvalidTranscriptError
from .transcribe import Segment, TranscriptionResult

TRANSCRIPT_SCHEMA_VERSION = "1.0"
TIMELINE_NOTE = "All times are seconds from the start of the original YouTube video."


def segment_id(i: int) -> str:
    return f"seg-{i:04d}"


def build_transcript(job_id: str, video: dict, audio: dict, result: TranscriptionResult, engine_info: dict, settings: dict) -> dict[str, Any]:
    segments = []
    for i, s in enumerate(result.segments, 1):
        segments.append({"id": segment_id(i), "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
                         "words": None if not s.words else [{"start": round(w.start, 3), "end": round(w.end, 3), "text": w.text.strip(),
                                                             "probability": None if w.probability is None else round(float(w.probability), 4)} for w in s.words]})
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION, "report_type": "youtube_transcript", "job_id": job_id, "review_status": "unreviewed",
        "video": video, "audio": audio,
        "transcription": {
            "engine": engine_info.get("engine"), "engine_details": engine_info, "settings": settings, "language": result.language,
            "language_probability": None if result.language_probability is None else round(result.language_probability, 4),
            "timestamp_level": result.timestamp_level, "timeline": TIMELINE_NOTE,
            "chunking": {"windows": result.windows, "window_seconds": settings.get("chunk_seconds"), "overlap_seconds": settings.get("overlap_seconds")},
            "warnings": result.warnings, "segments": segments,
        },
        "note": ("Machine transcription: it can contain errors. Word-level times, where present, are the engine's estimates, not forced alignment. "
                 "Segment-level times are what the engine reported for each segment."),
    }


def validate_transcript(doc: dict, tolerance: float = 0.5) -> list[str]:
    """Technical problems with the structure and the timeline (empty list = well-formed). Says nothing about accuracy."""
    problems: list[str] = []
    t = (doc or {}).get("transcription")
    if not isinstance(t, dict) or not isinstance(t.get("segments"), list):
        return ["transcription.segments is missing"]
    duration = ((doc.get("audio") or {}).get("duration_seconds"))
    prev_start, prev_id, ids = -1.0, None, set()
    levels = set()
    for i, s in enumerate(t["segments"]):
        sid = s.get("id", f"#{i}")
        if sid in ids:
            problems.append(f"duplicate segment id {sid}")
        ids.add(sid)
        a, b = s.get("start"), s.get("end")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (a, b)):
            problems.append(f"{sid}: start/end are not finite numbers")
            continue
        if a < 0 or b < a:
            problems.append(f"{sid}: invalid interval {a}-{b}")
        if a < prev_start - 1e-9:
            problems.append(f"{sid}: starts at {a} before the previous segment ({prev_id}, {prev_start}); order is not chronological")
        if isinstance(duration, (int, float)) and b > duration + tolerance:
            problems.append(f"{sid}: ends at {b}s, after the audio ends ({duration}s)")
        if not (isinstance(s.get("text"), str) and s["text"].strip()):
            problems.append(f"{sid}: empty text")
        words = s.get("words")
        levels.add(bool(words))
        if words:
            last = a - tolerance
            for w in words:
                ws, we = w.get("start"), w.get("end")
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (ws, we)) or we < ws:
                    problems.append(f"{sid}: a word has an invalid interval")
                    break
                if ws < last - 1e-9 or ws < a - tolerance or we > b + tolerance:
                    problems.append(f"{sid}: word timing '{w.get('text')}' ({ws}-{we}) is outside or out of order within the segment ({a}-{b})")
                    break
                last = ws
        prev_start, prev_id = a, sid
    declared = t.get("timestamp_level")
    actual = "word" if levels == {True} else "segment" if levels == {False} else "mixed"
    if levels and declared != actual:
        problems.append(f"timestamp_level is '{declared}' but the segments are '{actual}'")
    if not t["segments"]:
        problems.append("there are no segments")
    return problems


def require_valid(doc: dict) -> None:
    problems = validate_transcript(doc)
    if problems:
        raise InvalidTranscriptError("The transcript failed validation: " + "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""),
                                     details={"problems": problems})


# --- derived files ----------------------------------------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def to_vtt(doc: dict) -> str:
    """One cue per segment, with the segment id as the cue id. Text is written as transcribed (only a literal '-->' is neutralised)."""
    lines = ["WEBVTT", ""]
    for s in doc["transcription"]["segments"]:
        lines += [s["id"], f"{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}", s["text"].replace("-->", "->"), ""]
    return "\n".join(lines)


def to_text(doc: dict) -> str:
    """Plain English text, one segment per line, no timestamps. The structured transcript stays the source of truth."""
    return "\n".join(s["text"] for s in doc["transcription"]["segments"]) + "\n"


def timestamp_to_seconds(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    h, m, rest = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)
