"""Offline landmark extraction for ISLRTC dictionary clips (proof of concept, MediaPipe Holistic Landmarker).

Landmarks are *observations of a 2D video*: normalised image coordinates plus MediaPipe's estimated depth. They are not bone rotations and
not an ISL transcription. See docs/islrtc_pose_poc.md for the JSON format and the feasibility discussion.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

FORMAT_VERSION = "1.0"
MAX_HEIGHT = 720


@dataclass
class VideoInfo:
    width: int
    height: int
    out_width: int
    out_height: int
    fps: Optional[float]
    timestamps: List[float]
    codec: str


def probe_video(path: Path, ffprobe: str = "ffprobe") -> VideoInfo:
    """Dimensions plus the real presentation time of every frame (no constant-frame-rate assumption)."""
    meta = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
                           "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
    if meta.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {meta.stderr.strip()[:200]}")
    stream = (json.loads(meta.stdout).get("streams") or [None])[0]
    if not stream or not stream.get("width"):
        raise RuntimeError("no video stream")
    frames = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time",
                             "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=120)
    stamps = [float(x) for x in frames.stdout.split() if x.replace(".", "", 1).replace("-", "", 1).isdigit()]
    num, _, den = str(stream.get("avg_frame_rate", "")).partition("/")
    fps = round(float(num) / float(den), 3) if num.replace(".", "").isdigit() and den.isdigit() and float(den) > 0 else None
    w, h = int(stream["width"]), int(stream["height"])
    scale = min(1.0, MAX_HEIGHT / h)
    ow, oh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    return VideoInfo(w, h, ow, oh, fps, stamps, stream.get("codec_name", "?"))


def decode_frames(path: Path, info: VideoInfo, ffmpeg: str = "ffmpeg", max_frames: Optional[int] = None):
    """Yield RGB frames (numpy uint8, out_height x out_width x 3) via an FFmpeg pipe; HEVC is decoded by FFmpeg, not by the browser or OpenCV."""
    import numpy as np

    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-an", "-fps_mode", "passthrough", "-vf", f"scale={info.out_width}:{info.out_height}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = info.out_width * info.out_height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        n = 0
        while max_frames is None or n < max_frames:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(info.out_height, info.out_width, 3)
            n += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _pts(landmarks, with_vis: bool) -> Optional[List[List[float]]]:
    if not landmarks:
        return None
    out = []
    for lm in landmarks:
        row = [round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]
        if with_vis:
            row.append(round(getattr(lm, "visibility", None) or 0.0, 3))
        out.append(row)
    return out


def model_fingerprint(model: Path) -> Dict[str, object]:
    h = hashlib.sha256()
    with open(model, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"file": model.name, "sha256": h.hexdigest(), "bytes": model.stat().st_size}


def make_landmarker(model: Path, min_conf: float = 0.5, blendshapes: bool = True):
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=min_conf, min_face_landmarks_confidence=min_conf, min_pose_detection_confidence=min_conf,
        min_pose_landmarks_confidence=min_conf, min_hand_landmarks_confidence=min_conf, output_face_blendshapes=blendshapes)
    return vision.HolisticLandmarker.create_from_options(options)


def extract_clip(video: Path, model: Path, meta: Dict[str, object], ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                 max_frames: Optional[int] = None, keep_face: bool = True) -> Dict[str, object]:
    """Run the landmarker over one clip and return the landmark document. Frames keep their real timestamps.

    Decoded frames are not kept (a 17 s clip at 50 fps is about 2.4 GB of RGB); the overlay re-decodes in a streaming pass."""
    import mediapipe as mp

    info = probe_video(video, ffprobe)
    landmarker = make_landmarker(model)
    frames_out: List[Dict[str, object]] = []
    started = time.time()
    last_ms = -1
    try:
        for i, rgb in enumerate(decode_frames(video, info, ffmpeg, max_frames)):
            t = info.timestamps[i] if i < len(info.timestamps) else (i / info.fps if info.fps else float(i))
            ms = max(int(round(t * 1000)), last_ms + 1)  # MediaPipe needs strictly increasing integer milliseconds
            last_ms = ms
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            blend = None
            if result.face_blendshapes:
                blend = {c.category_name: round(c.score, 3) for c in result.face_blendshapes if c.score >= 0.01}
            frames_out.append({
                "i": i, "t": round(t, 4),
                "pose": _pts(result.pose_landmarks, True), "pose_world": _pts(result.pose_world_landmarks, True),
                "left_hand": _pts(result.left_hand_landmarks, False), "right_hand": _pts(result.right_hand_landmarks, False),
                "left_hand_world": _pts(result.left_hand_world_landmarks, False), "right_hand_world": _pts(result.right_hand_world_landmarks, False),
                "face": _pts(result.face_landmarks, False) if keep_face else None, "face_blendshapes": blend,
            })
    finally:
        landmarker.close()
    import mediapipe

    doc = {
        "format": "islrtc-landmarks", "format_version": FORMAT_VERSION,
        "source": meta,
        "video": {"codec": info.codec, "width": info.width, "height": info.height, "analysed_width": info.out_width,
                  "analysed_height": info.out_height, "fps": info.fps, "frame_count_probed": len(info.timestamps),
                  "frame_count_analysed": len(frames_out)},
        "extractor": {"name": "mediapipe HolisticLandmarker (Tasks API), VIDEO mode", "mediapipe_version": mediapipe.__version__,
                      "model": model_fingerprint(model), "min_confidence": 0.5, "seconds": round(time.time() - started, 2)},
        "coordinate_notes": {
            "pose/hands/face": "normalised image coordinates: x right, y down, in [0,1] of the analysed frame; z is MediaPipe's relative depth estimate (smaller = nearer the camera), not metric",
            "pose_world / *_hand_world": "metric-scale 3D estimate in metres, origin near the hips (pose) or hand centre (hands), inferred from a single 2D view, so depth is approximate",
            "left_hand/right_hand": "as labelled by MediaPipe for the signer (not mirrored). Verify against the overlay video before relying on it",
            "pose": "33 points [x,y,z,visibility]; hands 21 points [x,y,z]; face 478 points [x,y,z]; null = not detected in that frame",
        },
        "frames": frames_out,
    }
    return doc


def write_json(doc: Dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
