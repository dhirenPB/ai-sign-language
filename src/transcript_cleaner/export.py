"""Filesystem output. The raw copy is written once and never overwritten."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from .pipeline import ProcessResult
from .report import build_failure_report

SUBDIRS = ("raw", "cleaned", "structured", "reports", "review_queue")


class RawConflictError(RuntimeError):
    """A different file already exists at the raw-copy path. Raw copies are immutable."""


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write(path, (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def write_raw_copy(out_root: Path, raw_name: str, data: bytes) -> Path:
    path = out_root / "raw" / raw_name
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest():
            return path  # identical: nothing to do
        raise RawConflictError(
            f"{path} already exists with different content. Raw copies are never overwritten; "
            "rename the new file (new source_id) or remove the old raw copy yourself if it was a mistake."
        )
    _atomic_write(path, data)
    return path


def write_outputs(r: ProcessResult, out_root: Path) -> dict[str, str]:
    out_root = Path(out_root)
    paths = {
        "raw": write_raw_copy(out_root, r.raw_artifact_name, r.raw_bytes),
        "structured": out_root / "structured" / f"{r.source_id}.cleaned.json",
        "cleaned": out_root / "cleaned" / f"{r.source_id}.cleaned.txt",
        "review_json": out_root / "review_queue" / f"{r.source_id}.review.json",
        "review_csv": out_root / "review_queue" / f"{r.source_id}.review.csv",
        "report": out_root / "reports" / f"{r.source_id}.cleaning_report.json",
    }
    _write_json(paths["structured"], r.doc)
    _atomic_write(paths["cleaned"], r.clean_text.encode("utf-8"))
    _write_json(paths["review_json"], {"source_id": r.source_id, "items": r.review_items})
    _write_review_csv(paths["review_csv"], r.review_items)
    _write_json(paths["report"], r.report)  # written last: its presence marks a finished run
    return {k: str(v) for k, v in paths.items()}


def _write_review_csv(path: Path, items: list[dict]) -> None:
    cols = ["review_id", "chunk_id", "kind", "code", "text", "note", "chunk_text", "speaker_id", "start_time",
            "end_time", "source_start_offset", "source_end_offset", "decision"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_failure(out_root: Path, source_id: str, filename: str, data: Optional[bytes], error: str) -> Optional[str]:
    """Best effort: keep the raw copy (if readable) and a failed-status report. Never raises."""
    try:
        sha = hashlib.sha256(data).hexdigest() if data is not None else None
        path = Path(out_root) / "reports" / f"{source_id}.cleaning_report.json"
        _write_json(path, build_failure_report(source_id, filename, error, sha))
        return str(path)
    except Exception:  # noqa: BLE001 - failure reporting must not mask the original error
        return None


def write_batch_report(out_root: Path, entries: list[dict], config: dict) -> Path:
    from datetime import datetime, timezone
    from . import __version__

    summary = {
        "report_type": "batch_report",
        "tool_version": __version__,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_total": len(entries),
        "files_complete": sum(e["status"] == "complete" for e in entries),
        "files_needs_review": sum(e["status"] == "needs_review" for e in entries),
        "files_failed": sum(e["status"] == "failed" for e in entries),
        "review_items_total": sum(e.get("review_items", 0) for e in entries),
        "config": config,
        "files": entries,
    }
    path = Path(out_root) / "reports" / "batch_report.json"
    _write_json(path, summary)
    return path
