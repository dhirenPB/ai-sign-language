"""Command-line / batch interface. One bad file never stops the batch."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

from .config import SOURCE_TYPES, CleaningConfig
from .export import SUBDIRS, RawConflictError, write_batch_report, write_failure, write_outputs
from .ingest import SUPPORTED_SUFFIXES
from .pipeline import make_source_id, process_bytes


def discover_inputs(path: Path) -> list[Path]:
    """A file is used as-is. A folder contributes its own .txt/.srt/.vtt files (not sub-folders, so the generated
    raw/ cleaned/ structured/ ... folders are never re-read)."""
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def run_batch(files: Sequence[Path], out_root: Path, cfg: CleaningConfig, log=print) -> list[dict]:
    entries: list[dict] = []
    seen_ids: dict[str, str] = {}
    for path in files:
        source_id = make_source_id(path.name)
        entry: dict = {"file": str(path), "source_id": source_id, "status": "failed"}
        data: Optional[bytes] = None
        try:
            if source_id in seen_ids:
                raise ValueError(f"source_id '{source_id}' already used by {seen_ids[source_id]}; rename one of the files.")
            seen_ids[source_id] = path.name
            data = path.read_bytes()
            if not data.strip():
                raise ValueError("file is empty")
            result = process_bytes(data, path.name, cfg, source_id)
            outputs = write_outputs(result, out_root)
            entry.update({
                "status": result.status,
                "format": result.fmt,
                "chunks": len(result.chunks),
                "flags": sum(len(c.uncertainty_flags) for c in result.chunks),
                "review_items": len(result.review_items),
                "validation_errors": result.validation["errors"],
                "validation_warnings": len(result.validation["warnings"]),
                "outputs": outputs,
            })
            if result.status == "failed":
                entry["error"] = "; ".join(e["message"] for e in result.validation["errors"][:3])
        except RawConflictError as exc:
            entry["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: report and continue with the next file
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=6)
        if entry["status"] == "failed" and "outputs" not in entry:
            entry["report"] = write_failure(out_root, source_id, path.name, data, entry["error"])
        entries.append(entry)
        log(_summary_line(entry))
    return entries


def _summary_line(e: dict) -> str:
    name = Path(e["file"]).name
    if e["status"] == "failed":
        return f"[FAILED ] {name}: {e.get('error', 'unknown error')}"
    tag = "OK" if e["status"] == "complete" else "REVIEW"
    return (f"[{tag:7}] {name}: {e['chunks']} chunks, {e['flags']} flags, {e['review_items']} review items, "
            f"{e['validation_warnings']} warnings")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcript_cleaner",
        description="Phase 1: clean and segment English transcripts (.txt/.srt/.vtt). No gloss translation.",
    )
    p.add_argument("input", type=Path, help="a transcript file, or a folder containing .txt/.srt/.vtt files")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output root holding raw/ cleaned/ structured/ reports/ review_queue/ "
                        "(default: the input folder, or the input file's folder)")
    p.add_argument("--config", type=Path, help="JSON file overriding CleaningConfig fields")
    p.add_argument("--language", default=None, help="source language code (default: en)")
    p.add_argument("--source-type", choices=SOURCE_TYPES, default=None,
                   help="youtube_prerecorded | livestream | unknown (default: unknown; never guessed from the URL)")
    p.add_argument("--keep-fillers", action="store_true", help="keep um/uh/er in the text (they are removed by default; every removal is logged)")
    p.add_argument("--collapse-repeats", action="store_true",
                   help="collapse immediate word repeats (OFF by default; every collapse is queued for review)")
    p.add_argument("--drop-sound-tags", action="store_true", help="remove bracketed tags like [Music] (OFF by default)")
    p.add_argument("--max-chunk-words", type=int, default=None, help="split chunks longer than this (default: 60)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = CleaningConfig.from_json(args.config) if args.config else CleaningConfig()
        if args.language:
            cfg.language = args.language
        if args.source_type:
            cfg.source_type = args.source_type
        if args.keep_fillers:
            cfg.remove_fillers = False
        if args.collapse_repeats:
            cfg.collapse_repeated_words = True
        if args.drop_sound_tags:
            cfg.drop_sound_tags = True
        if args.max_chunk_words:
            cfg.max_chunk_words = args.max_chunk_words
        cfg.validate()
    except (ValueError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    files = discover_inputs(args.input)
    if not files:
        print(f"No {'/'.join(SUPPORTED_SUFFIXES)} files found in {args.input}", file=sys.stderr)
        return 2
    out_root = args.out or (args.input if args.input.is_dir() else args.input.parent)
    for sub in SUBDIRS:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    entries = run_batch(files, out_root, cfg)
    report = write_batch_report(out_root, entries, cfg.to_dict())
    failed = sum(e["status"] == "failed" for e in entries)
    print(f"\n{len(entries) - failed}/{len(entries)} processed, {failed} failed. Batch report: {report}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
