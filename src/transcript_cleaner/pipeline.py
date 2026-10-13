"""Orchestration: bytes in -> ProcessResult (in memory). Nothing here touches the filesystem."""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Optional

from . import SCHEMA_VERSION
from .clean import clean_chunk_text
from .config import CleaningConfig
from .detect import detect_flags
from .ingest import parse
from .models import Chunk, Edit, Flag, ParsedSource
from .normalize import normalize_text
from .report import build_report, build_review_items, render_clean_text
from .schema import validate_against_schema
from .segment import sentence_spans, split_overlong
from .validate import validate_chunks


@dataclass
class ProcessResult:
    source_id: str
    filename: str
    fmt: str
    sha256: str
    raw_bytes: bytes
    parsed: ParsedSource
    chunks: list[Chunk]
    edits: list[Edit]
    normalization_counts: Counter
    validation: dict
    status: str  # complete | needs_review | failed
    doc: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    clean_text: str = ""
    review_items: list[dict] = field(default_factory=list)

    @property
    def raw_artifact_name(self) -> str:
        suffix = self.filename[self.filename.rfind("."):] if "." in self.filename else ".txt"
        return f"{self.source_id}.raw{suffix}"


def make_source_id(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    slug = re.sub(r"[^a-z0-9]+", "-", stem.casefold()).strip("-")
    return slug or "source"


def process_bytes(data: bytes, filename: str, cfg: Optional[CleaningConfig] = None,
                  source_id: Optional[str] = None) -> ProcessResult:
    cfg = cfg or CleaningConfig()
    cfg.validate()
    source_id = source_id or make_source_id(filename)
    sha = hashlib.sha256(data).hexdigest()

    ps = parse(data, filename, source_id, cfg)
    for i, e in enumerate(ps.edits, start=1):
        e.edit_id = i
    edits: list[Edit] = list(ps.edits)
    norm_counts: Counter = Counter()
    chunks = _build_chunks(ps, cfg, edits, norm_counts)

    validation = validate_chunks(ps, chunks, edits, cfg)
    result = ProcessResult(
        source_id=source_id, filename=filename, fmt=ps.fmt, sha256=sha, raw_bytes=data, parsed=ps, chunks=chunks,
        edits=edits, normalization_counts=norm_counts, validation=validation, status="complete",
    )
    result.review_items = build_review_items(result)
    review_required = bool(result.review_items)
    result.status = "failed" if validation["errors"] else ("needs_review" if review_required else "complete")

    result.doc = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "source_filename": filename,
        "source_type": cfg.source_type,
        "source_language": cfg.language,
        "source_format": ps.fmt,
        "source_sha256": sha,
        "raw_artifact": result.raw_artifact_name,
        "offset_basis": ("character offsets (Python str indices / Unicode code points) into the raw file decoded as "
                         f"{ps.encoding}, BOM removed, original line endings kept" if ps.fmt == "txt" else None),
        "source_metadata": {k: v for k, v in ps.metadata.items() if k != "source_language"},
        "timestamps_available": ps.timestamps_available,
        "speakers_available": ps.speakers_available,
        "cleaning_status": result.status,
        "review_required": review_required,
        "warnings": ps.warnings + validation["warnings"] + validation["errors"],
        "chunks": [c.to_dict() for c in chunks],
    }
    schema_errors = validate_against_schema(result.doc)
    if schema_errors:
        validation["errors"].extend({"code": "schema_violation", "message": m} for m in schema_errors)
        validation["passed"] = False
        result.status = result.doc["cleaning_status"] = "failed"
        result.doc["warnings"] = ps.warnings + validation["warnings"] + validation["errors"]

    result.clean_text = render_clean_text(result)
    result.report = build_report(result, cfg)
    return result


# --------------------------------------------------------------------------------------------------------
def _build_chunks(ps: ParsedSource, cfg: CleaningConfig, edits: list[Edit], norm_counts: Counter) -> list[Chunk]:
    chunks: list[Chunk] = []
    for _, group_iter in groupby(ps.pieces, key=lambda p: p.turn_index):
        group = list(group_iter)
        joined, starts, ends = "", [], []
        for p in group:
            if joined:
                joined += " "
            starts.append(len(joined))
            joined += p.text
            ends.append(len(joined))

        spans = []
        for sp in sentence_spans(joined):
            spans.extend(split_overlong(joined, sp, cfg.max_chunk_words))

        for sp in spans:
            raw_chunk = joined[sp.start:sp.end]
            first = max(bisect_right(starts, sp.start) - 1, 0)
            last = max(bisect_left(starts, sp.end) - 1, first)
            covered = group[first:last + 1]

            so = eo = None
            if len(covered) == 1 and covered[0].raw_start is not None:
                base = covered[0].raw_start - starts[first]
                so, eo = base + sp.start, base + sp.end

            nres = normalize_text(raw_chunk)
            norm_counts.update(nres.counts)
            pending: list[tuple[str, str, Any]] = []  # (stage, kind, draft-like)
            if nres.mojibake:
                pending.append(("normalize", "repair_mojibake", (nres.mojibake[0], nres.mojibake[1],
                                "Text looked like UTF-8 read as cp1252; round-trip repair succeeded cleanly.",
                                "high", False)))
            text, drafts = clean_chunk_text(nres.text, cfg)
            for d in drafts:
                pending.append(("clean", d.operation, (d.original, d.cleaned, d.reason, d.confidence, d.review_required)))

            chunk_id = f"{ps.source_id}-{len(chunks) + 1:04d}" if text else None
            notes: list[str] = [f"normalized:{k}={v}" for k, v in sorted(nres.counts.items())]
            for stage, op, (orig, new, reason, conf, review) in pending:
                edit = Edit(stage=stage, operation=op, original=orig, cleaned=new, reason=reason, confidence=conf,
                            review_required=review, chunk_id=chunk_id, source_start_offset=so, source_end_offset=eo,
                            edit_id=len(edits) + 1)
                edits.append(edit)
                notes.append(f"{op}: {orig!r} -> {new!r} [edit {edit.edit_id}]")
            if not text:
                if not pending:
                    edits.append(Edit(stage="clean", operation="drop_empty_chunk", original=raw_chunk, cleaned="",
                                      reason="Chunk contained only invisible characters.", source_start_offset=so,
                                      source_end_offset=eo, edit_id=len(edits) + 1))
                continue

            flags: list[Flag] = detect_flags(text, cfg)
            if sp.split == "forced":
                lead = text.split(" ")[0]
                flags.append(Flag("forced_split", lead, 0, len(lead),
                                  "Chunk starts at a hard word-count split inside a long unpunctuated run; the "
                                  "meaning boundary is uncertain.", False))
            if sp.split:
                notes.append(f"split_overlong:{sp.split}")

            times_s = [p.start_time for p in covered if p.start_time is not None]
            times_e = [p.end_time for p in covered if p.end_time is not None]
            cue_idx = sorted({p.cue_index for p in covered if p.cue_index is not None})
            cue_ids = list(dict.fromkeys(p.cue_id for p in covered if p.cue_id is not None))
            chunks.append(Chunk(
                chunk_id=chunk_id, text=text, speaker_id=covered[0].speaker_id, turn_index=covered[0].turn_index,
                start_time=times_s[0] if times_s else None, end_time=times_e[-1] if times_e else None,
                source_start_offset=so, source_end_offset=eo, cue_indices=cue_idx or None, cue_ids=cue_ids or None,
                uncertainty_flags=flags, edit_notes=notes,
            ))
    return chunks
