"""Overlay video for inspecting landmark tracking: left = source frame with landmarks, right = the landmarks alone (skeleton view)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
LEFT_COLOR, RIGHT_COLOR, POSE_COLOR, FACE_COLOR = (255, 128, 0), (0, 200, 255), (60, 220, 60), (200, 200, 200)  # RGB


def _draw(img, f: Dict[str, object], cv2) -> None:
    h, w = img.shape[:2]
    px = lambda p: (int(p[0] * w), int(p[1] * h))  # noqa: E731
    pose = f.get("pose")
    if pose:
        for a, b in POSE_EDGES:
            if pose[a][3] > 0.3 and pose[b][3] > 0.3:
                cv2.line(img, px(pose[a]), px(pose[b]), POSE_COLOR, 2, cv2.LINE_AA)
    for key, color in (("left_hand", LEFT_COLOR), ("right_hand", RIGHT_COLOR)):
        hand = f.get(key)
        if hand:
            for a, b in HAND_EDGES:
                cv2.line(img, px(hand[a]), px(hand[b]), color, 2, cv2.LINE_AA)
            for p in hand:
                cv2.circle(img, px(p), 3, color, -1, cv2.LINE_AA)
    face = f.get("face")
    if face:
        for p in face[::6]:
            cv2.circle(img, px(p), 1, FACE_COLOR, -1)


def write_overlay(rgb_frames, frames: List[Dict[str, object]], width: int, height: int, fps: float, out: Path, label: str, ffmpeg: str = "ffmpeg") -> None:
    """`rgb_frames` is any iterable of RGB frames (a streaming decode); it is consumed one frame at a time."""
    import cv2
    import numpy as np

    if not frames:
        raise ValueError("no frames to draw")
    h, w = height, width
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w * 2}x{h}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for rgb, f in zip(rgb_frames, frames):
            left = np.ascontiguousarray(rgb).copy()
            right = np.full_like(left, 245)
            _draw(left, f, cv2)
            _draw(right, f, cv2)
            cv2.putText(left, f"{label}  t={f['t']:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(left, "orange=left  cyan=right (MediaPipe labels, signer-side)", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            proc.stdin.write(np.concatenate([left, right], axis=1).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed while writing the overlay video")
