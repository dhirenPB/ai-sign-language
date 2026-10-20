"""Side-by-side comparison sheets: source clip frame (with its landmarks) next to the Y Bot rendered from the retargeted track.

Needs: the yap-render app serving /lab/retarget (see docs/islrtc_pose_poc.md), Chrome/Edge for headless screenshots, ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

CHROME_CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                     r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def find_chrome(explicit: str = "") -> str:
    for c in ([explicit] if explicit else []) + CHROME_CANDIDATES:
        if c and Path(c).is_file():
            return c
    raise FileNotFoundError("Chrome/Edge not found; pass --chrome")


def screenshot(chrome: str, url: str, out: Path) -> None:
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=720,720", "--virtual-time-budget=20000",
                    f"--screenshot={out}", url], capture_output=True, timeout=120, check=True)


def source_frame(video: Path, t: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=-2:720", str(out)], check=True)


def main(argv=None) -> int:
    import cv2
    import numpy as np

    ap = argparse.ArgumentParser(prog="islrtc-compare")
    ap.add_argument("--landmarks", required=True, help="<asset id>.landmarks.json")
    ap.add_argument("--video", required=True, help="the source clip")
    ap.add_argument("--base-url", default="http://localhost:3057")
    ap.add_argument("--frames", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chrome", default="")
    args = ap.parse_args(argv)
    doc = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    asset = doc["source"]["asset_id"]
    chrome = find_chrome(args.chrome)
    rows: List[object] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in args.frames:
            t = doc["frames"][min(i, len(doc["frames"]) - 1)]["t"]
            src, av = Path(tmp) / f"s{i}.png", Path(tmp) / f"a{i}.png"
            source_frame(Path(args.video), t, src)
            screenshot(chrome, f"{args.base_url}/lab/retarget?clip={asset}&frame={i}", av)
            s, a = cv2.imread(str(src)), cv2.imread(str(av))
            if s is None or a is None:
                print(f"frame {i}: could not read images", file=sys.stderr)
                continue
            a = cv2.resize(a, (720, 720))
            s = s[:, 280:1000] if s.shape[1] >= 1000 else s  # centre crop to the signer
            s = cv2.resize(s, (720, 720))
            row = np.concatenate([s, a], axis=1)
            cv2.putText(row, f"{doc['source']['label']}  frame {i}  t={t:.2f}s   [source | avatar]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rows.append(row)
    if not rows:
        return 1
    cv2.imwrite(args.out, np.concatenate(rows, axis=0))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
