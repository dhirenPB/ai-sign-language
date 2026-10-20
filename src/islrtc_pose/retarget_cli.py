"""python run_islrtc_retarget.py --landmarks outputs/pose_poc --glb <ybot.glb> --out <folder> --labels eat professor"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .retarget import Rig, retarget


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="islrtc-retarget", description="EXPERIMENT: landmarks -> Y Bot bone rotations (arms, hands, finger flexion).")
    ap.add_argument("--landmarks", default="outputs/pose_poc")
    ap.add_argument("--glb", required=True, help="path to yap-render public/models/ybot.glb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="source labels to process (default: all clips whose quality says usable)")
    args = ap.parse_args(argv)
    rig = Rig(Path(args.glb))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {x.lower() for x in args.labels}
    done = []
    for path in sorted(Path(args.landmarks).glob("*.landmarks.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        label = doc["source"]["label"]
        if wanted and label.lower() not in wanted:
            continue
        if not wanted and not doc.get("quality", {}).get("usable_for_retargeting"):
            continue
        track = retarget(doc, rig)
        (out / f"{doc['source']['asset_id']}.bones.json").write_text(json.dumps(track, separators=(",", ":")), encoding="utf-8")
        pe = track["position_error"]
        print(f"{label:<11} frames={len(track['frames']):<4} wrist err (frac. arm length) R={pe['right']['wrist_err']} L={pe['left']['wrist_err']}")
        done.append(label)
    if not done:
        print("nothing retargeted", file=sys.stderr)
        return 1
    return 0
