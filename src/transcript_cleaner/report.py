"""Human- and machine-readable outputs derived from a ProcessResult: clean text, review queue, cleaning report."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from . import SCHEMA_VERSION, __version__
from .config import CleaningConfig
from .normalize import comparable_tokens

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import ProcessResult

ACCURACY_NOTE = (
    "The source audio/video was not available. Nothing here has been checked against what was actually said; the "
    "cleaner only makes the transcript consistent and traceable. ASR errors in the source text remain possible."
)


def render_clean_text(r: "ProcessResult") -> str:
    """Spoken words only: one chunk per line. No title, source, speaker labels or timestamps; all of that stays in the
    structured JSON and the report."""
    return "\n".join(c.text for c in r.chunks) + "\n"


def build_review_items(r: "ProcessResult") -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    chunk_by_id = {c.chunk_id: c for c in r.chunks}

    def base(chunk, kind, code, text, note):
        return {
            "review_id": f"{r.source_id}-R{len(items) + 1:04d}",
            "source_id": r.source_id,
            "chunk_id": chunk.chunk_id if chunk else None,
            "kind": kind,
            "code": code,
            "text": text,
            "note": note,
            "chunk_text": chunk.text if chunk else None,
            "speaker_id": chunk.speaker_id if chunk else None,
            "start_time": chunk.start_time if chunk else None,
            "end_time": chunk.end_time if chunk else None,
            "source_start_offset": chunk.source_start_offset if chunk else None,
            "source_end_offset": chunk.source_end_offset if chunk else None,
            "decision": None,  # for the human reviewer to fill in
        }

    for c in r.chunks:
        for f in c.uncertainty_flags:
            if f.review_required:
                items.append(base(c, "flag", f.code, f.text, f.note))
    for e in r.edits:
        if e.review_required:
            items.append(base(chunk_by_id.get(e.chunk_id), "edit", e.operation, f"{e.original!r} -> {e.cleaned!r}", e.reason))
    return items


def build_report(r: "ProcessResult", cfg: CleaningConfig) -> dict[str, Any]:
    ps = r.parsed
    flag_counts = Counter(f.code for c in r.chunks for f in c.uncertainty_flags)
    review_counts = Counter(i["code"] for i in r.review_items)
    source_words = sum(len(comparable_tokens(p.text)) for p in ps.pieces)
    clean_words = sum(len(comparable_tokens(c.text)) for c in r.chunks)
    lengths = [len(comparable_tokens(c.text)) for c in r.chunks]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "cleaning_report",
        "tool_version": __version__,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_id": r.source_id,
        "source_filename": r.filename,
        "source_format": r.fmt,
        "source_sha256": r.sha256,
        "source_bytes": len(r.raw_bytes),
        "raw_artifact": r.raw_artifact_name,
        "encoding": ps.encoding,
        "cleaning_status": r.status,
        "review_required": bool(r.review_items),
        "accuracy_note": ACCURACY_NOTE,
        "config": cfg.to_dict(),
        "stats": {
            "source_characters": len(ps.raw_text),
            "source_words": source_words,
            "cleaned_words": clean_words,
            "chunks": len(r.chunks),
            "words_per_chunk_avg": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "words_per_chunk_max": max(lengths) if lengths else 0,
            "turns": len({c.turn_index for c in r.chunks}),
            "speakers": sorted({c.speaker_id for c in r.chunks if c.speaker_id}),
            "timestamps_available": ps.timestamps_available,
            "speakers_available": ps.speakers_available,
            "header_metadata": {k: v for k, v in ps.metadata.items() if k != "source_language"},
        },
        "parse_counts": dict(ps.counts),
        "normalization_counts": dict(r.normalization_counts),
        "edit_counts_by_operation": dict(Counter(e.operation for e in r.edits)),
        "flag_counts": dict(flag_counts),
        "review_item_counts": dict(review_counts),
        "review_item_total": len(r.review_items),
        "warnings": ps.warnings + r.validation["warnings"],
        "validation": r.validation,
        "edits": [e.to_dict() for e in r.edits],
    }


def build_failure_report(source_id: str, filename: str, error: str, sha256: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "cleaning_report",
        "tool_version": __version__,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_id": source_id,
        "source_filename": filename,
        "source_sha256": sha256,
        "cleaning_status": "failed",
        "review_required": True,
        "error": error,
        "note": "No cleaned outputs were written for this run. Earlier outputs for the same source_id, if any, "
                "may be stale.",
        "accuracy_note": ACCURACY_NOTE,
    }
