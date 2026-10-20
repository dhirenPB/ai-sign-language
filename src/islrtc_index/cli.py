"""Command line: python run_islrtc_index.py [--data-dir DIR] [--out data/islrtc] [--no-probe] [--workers N] [--reviews review.csv]"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .index import run_index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="islrtc-index", description="Index the ISLRTC dictionary (read-only; videos are not copied or loaded).")
    ap.add_argument("--data-dir", default=os.environ.get("ISLRTC_DATA_DIR", ""), help="dataset root (default: $ISLRTC_DATA_DIR)")
    ap.add_argument("--out", default="data/islrtc", help="output folder for the SQLite index, manifest, review queue and audit")
    ap.add_argument("--no-probe", action="store_true", help="skip ffprobe (no codec/duration; unreadable files go undetected)")
    ap.add_argument("--workers", type=int, default=6, help="parallel ffprobe processes (default 6)")
    ap.add_argument("--reviews", default="", help="review CSV with a filled 'decision' column to apply")
    ap.add_argument("--synonyms", default="data/vocab/synonyms.json", help="curated synonym table, validated against the real vocabulary")
    ap.add_argument("--yap-vocab", default="../yap-render/lib/islVocabulary.ts", help="yap-render lib/islVocabulary.ts (the avatar's words)")
    args = ap.parse_args(argv)
    if not args.data_dir:
        print("error: pass --data-dir or set ISLRTC_DATA_DIR", file=sys.stderr)
        return 2

    def progress(done, total):
        print(f"  probed {done}/{total}", file=sys.stderr, flush=True)

    try:
        report = run_index(Path(args.data_dir), Path(args.out), probe=not args.no_probe, workers=args.workers,
                           reviews=Path(args.reviews) if args.reviews else None, progress=progress,
                           synonyms=Path(args.synonyms) if args.synonyms else None, yap_vocab=Path(args.yap_vocab) if args.yap_vocab else None)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0
