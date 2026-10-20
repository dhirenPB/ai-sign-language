"""python run_islrtc_pose.py --manifest data/islrtc/manifest.json --data-dir <ISLRTC root> --labels today world ... --out outputs/pose_poc --model <holistic_landmarker.task>"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from .quality import assess


def select_assets(manifest: Dict[str, object], labels: List[str], ids: List[str]) -> List[Dict[str, object]]:
    """Resolve labels / ids to manifest assets. A label with several signs or no clean entry is skipped and reported, never guessed."""
    from islrtc_index.parse import normalize_label

    assets = manifest["assets"]
    by_id = {a["asset_id"]: a for a in assets}
    chosen, problems = [], []
    for i in ids:
        (chosen.append(by_id[i]) if i in by_id else problems.append(f"unknown asset id {i}"))
    for label in labels:
        key = normalize_label(label)
        hits = [a for a in assets if a["label_norm"] == key and a["status"] in ("source_label", "verified", "needs_review")]
        if len(hits) == 1 and hits[0]["status"] != "needs_review":
            chosen.append(hits[0])
        else:
            problems.append(f"label {label!r}: {len(hits)} candidate(s); skipped (ambiguous or needs review)")
    for p in problems:
        print("warning:", p, file=sys.stderr)
    return chosen


def reassess(folder: Path) -> int:
    """Recompute quality for stored landmark documents; keeps the landmarks untouched."""
    summary = []
    for path in sorted(folder.glob("*.landmarks.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["quality"] = assess(doc)
        path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        src = doc["source"]
        summary.append({"asset_id": src["asset_id"], "label": src["label"], "rel_path": src["rel_path"], **doc["quality"]})
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"re-scored {len(summary)} clip(s): {sum(1 for r in summary if r['usable_for_retargeting'])} usable")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="islrtc-pose", description="Extract MediaPipe Holistic landmarks from a few ISLRTC clips (offline experiment).")
    ap.add_argument("--manifest", default="data/islrtc/manifest.json")
    ap.add_argument("--data-dir", default=os.environ.get("ISLRTC_DATA_DIR", ""))
    ap.add_argument("--model", default=os.environ.get("MEDIAPIPE_HOLISTIC_MODEL", ""), help="holistic_landmarker.task (see docs/islrtc_pose_poc.md)")
    ap.add_argument("--out", default="outputs/pose_poc")
    ap.add_argument("--labels", nargs="*", default=[], help="dictionary labels to extract (must be unambiguous)")
    ap.add_argument("--assets", nargs="*", default=[], help="asset ids to extract")
    ap.add_argument("--max-frames", type=int, default=None, help="cap analysed frames per clip (default: whole clip)")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--reassess", default="", help="re-score the *.landmarks.json files in this folder with the current quality rules (no MediaPipe run)")
    ap.add_argument("--no-face", action="store_true", help="do not store the 478 face points (blendshapes are still kept)")
    args = ap.parse_args(argv)
    if args.reassess:
        return reassess(Path(args.reassess))
    if not args.data_dir or not args.model:
        print("error: --data-dir (or ISLRTC_DATA_DIR) and --model (or MEDIAPIPE_HOLISTIC_MODEL) are required", file=sys.stderr)
        return 2
    data_dir, model, out = Path(args.data_dir), Path(args.model), Path(args.out)
    if not model.is_file():
        print(f"error: model file not found: {model}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    assets = select_assets(manifest, args.labels, args.assets)
    if not assets:
        print("error: nothing to process", file=sys.stderr)
        return 2

    from .extract import decode_frames, extract_clip, probe_video, write_json
    from .overlay import write_overlay

    summary = []
    for a in assets:
        video = data_dir / a["rel_path"]
        meta = {"asset_id": a["asset_id"], "label": a["label"], "rel_path": a["rel_path"], "manifest_status": a["status"]}
        row: Dict[str, object] = {"asset_id": a["asset_id"], "label": a["label"], "rel_path": a["rel_path"]}
        try:
            doc = extract_clip(video, model, meta, max_frames=args.max_frames, keep_face=not args.no_face)
            quality = assess(doc)
            doc["quality"] = quality
            write_json(doc, out / f"{a['asset_id']}.landmarks.json")
            if not args.no_overlay and doc["frames"]:
                info = probe_video(video)
                write_overlay(decode_frames(video, info, max_frames=args.max_frames), doc["frames"], info.out_width, info.out_height,
                              doc["video"]["fps"] or 25.0, out / f"{a['asset_id']}.overlay.mp4", a["label"])
            row.update(quality)
            print(f"{a['label']:<22} frames={quality['frames']:<4} hands L/R={quality['detection_rate']['left_hand']}/{quality['detection_rate']['right_hand']} "
                  f"usable={quality['usable_for_retargeting']} issues={len(quality['issues'])}", flush=True)
        except Exception as exc:  # one bad clip must not stop the batch
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{a['label']:<22} FAILED: {row['error']}", file=sys.stderr, flush=True)
        summary.append(row)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0
