"""Quality metrics for an extracted landmark document. Pure Python (no numpy), so the tests run without MediaPipe."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
LOW_VISIBILITY = 0.5
MAX_INTERPOLATABLE_GAP = 3  # frames; a longer mid-sign dropout cannot be bridged honestly
MIN_WINDOW_COVERAGE = 0.9


def _runs(flags: List[bool]) -> List[int]:
    """Lengths of consecutive False runs that have a True on both sides (a dropout inside a detected stretch, not leading/trailing absence)."""
    seen_true = False
    runs: List[int] = []
    gap = 0
    for f in flags:
        if f:
            if seen_true and gap:
                runs.append(gap)
            seen_true, gap = True, 0
        elif seen_true:
            gap += 1
    return runs


def _window(flags: List[bool]) -> Dict[str, object]:
    """Active window of a hand: first to last detected frame, and how much of that window is actually detected."""
    idx = [i for i, f in enumerate(flags) if f]
    if not idx:
        return {"start": None, "end": None, "frames": 0, "coverage": 0.0}
    span = idx[-1] - idx[0] + 1
    return {"start": idx[0], "end": idx[-1], "frames": span, "coverage": round(len(idx) / span, 3)}


def _wrist_path(frames: List[Dict[str, object]], key: str) -> Optional[float]:
    """Total 2D path length of the wrist (hand landmark 0) in normalised units, over frames where it was detected."""
    pts = [f[key][0][:2] for f in frames if f.get(key)]
    if len(pts) < 2:
        return None
    return round(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])), 3)


def _jitter(frames: List[Dict[str, object]], key: str) -> Optional[float]:
    """Mean magnitude of the second difference of the wrist position across consecutive detected frames; high = shaky tracking."""
    pts = [f[key][0][:2] for f in frames if f.get(key)]
    if len(pts) < 3:
        return None
    seconds = [math.dist((c[0] - b[0], c[1] - b[1]), (b[0] - a[0], b[1] - a[1])) for a, b, c in zip(pts, pts[1:], pts[2:])]
    return round(sum(seconds) / len(seconds), 4)


def assess(doc: Dict[str, object]) -> Dict[str, object]:
    frames: List[Dict[str, object]] = doc["frames"]  # type: ignore[assignment]
    n = len(frames)
    if n == 0:
        return {"frames": 0, "issues": ["no frames analysed"], "usable_for_retargeting": False}
    rate = lambda key: round(sum(1 for f in frames if f.get(key)) / n, 3)  # noqa: E731
    left_flags = [bool(f.get("left_hand")) for f in frames]
    right_flags = [bool(f.get("right_hand")) for f in frames]
    both = round(sum(1 for a, b in zip(left_flags, right_flags) if a and b) / n, 3)
    low_vis = 0
    for f in frames:
        pose = f.get("pose")
        if pose and (pose[POSE_LEFT_WRIST][3] < LOW_VISIBILITY or pose[POSE_RIGHT_WRIST][3] < LOW_VISIBILITY):
            low_vis += 1
    left_path, right_path = _wrist_path(frames, "left_hand"), _wrist_path(frames, "right_hand")
    paths = {"left": left_path or 0.0, "right": right_path or 0.0}
    dominant = "none" if not any(paths.values()) else ("both" if min(paths.values()) > 0.5 * max(paths.values()) else max(paths, key=paths.get))
    stats = {
        "frames": n, "duration_s": round(frames[-1]["t"] - frames[0]["t"], 3) if n > 1 else 0.0,
        "detection_rate": {"pose": rate("pose"), "left_hand": rate("left_hand"), "right_hand": rate("right_hand"), "face": rate("face"),
                           "both_hands": both},
        "hand_dropout_runs": {"left": _runs(left_flags), "right": _runs(right_flags)},
        "active_window": {"left": _window(left_flags), "right": _window(right_flags)},
        "wrist_low_visibility_frames": low_vis,
        "wrist_path_length": {"left": left_path, "right": right_path},
        "wrist_jitter": {"left": _jitter(frames, "left_hand"), "right": _jitter(frames, "right_hand")},
        "moving_hands_measured": dominant,
    }
    issues: List[str] = []
    if stats["detection_rate"]["pose"] < 0.9:
        issues.append(f"pose missing in {round((1 - stats['detection_rate']['pose']) * 100)}% of frames")
    if max(left_flags + right_flags, default=False) is False:
        issues.append("no hand detected in any frame")
    for side, runs in stats["hand_dropout_runs"].items():
        if runs:
            issues.append(f"{side} hand lost mid-sign: {len(runs)} dropout(s), longest {max(runs)} frames")
    if low_vis / n > 0.25:
        issues.append(f"wrist low visibility (<{LOW_VISIBILITY}) in {round(low_vis / n * 100)}% of frames (occlusion or out of frame)")
    for side in ("left", "right"):
        j = stats["wrist_jitter"][side]
        if j is not None and j > 0.02:
            issues.append(f"{side} wrist tracking is jittery (mean 2nd difference {j})")
    stats["issues"] = issues
    # Deliberately strict, but fair to short clips where a hand is simply at rest (out of frame) before and after the sign:
    # a moving hand must be tracked in >= 90% of its own active window, with no gap too long to interpolate.
    ok = stats["detection_rate"]["pose"] >= 0.9
    moving = [s for s in ("left", "right") if paths[s] > 0.2]
    ok = ok and bool(moving)
    for side in moving:
        win = stats["active_window"][side]
        longest = max(stats["hand_dropout_runs"][side], default=0)
        ok = ok and win["coverage"] >= MIN_WINDOW_COVERAGE and longest <= MAX_INTERPOLATABLE_GAP
    stats["usable_for_retargeting"] = bool(ok)
    return stats
