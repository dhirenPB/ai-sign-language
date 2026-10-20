"""EXPERIMENT: retarget MediaPipe landmarks onto the Yap & Render Y Bot rig (arms, hand orientation, finger flexion only).

What this is: a direction-based solver. Each limb bone is rotated so that it points where the landmarks say the limb points; the hand is rotated to
match the palm frame; each finger joint gets a flexion angle from the landmark angles. Everything is computed from a single 2D camera view with
estimated depth, so it is an approximation and is only "successful" if the rendered avatar is inspected and compared with the source clip.

Not handled (left at rest): shoulders/clavicles, spine and head, thumbs, finger spread (abduction), wrist/forearm twist beyond what the palm frame
implies, and every non-manual feature (the rig has no face bones or blendshapes). Bone lengths differ from the signer's, so hand *positions*
(e.g. touching the mouth) are only approximate; the report measures that.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FORMAT_VERSION = "1.0"
FINGERS = {"Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}
MAX_FLEX = 1.9  # radians (about 109 degrees) per joint
SIDES = {"Right": {"shoulder": 12, "elbow": 14, "wrist": 16, "pinky": 18, "index": 20, "hand_key": "right_hand", "sign": 1.0},
         "Left": {"shoulder": 11, "elbow": 13, "wrist": 15, "pinky": 17, "index": 19, "hand_key": "left_hand", "sign": -1.0}}


# ---------------------------------------------------------------- maths
def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m: np.ndarray) -> List[float]:
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix taking unit vector a to unit vector b."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # opposite: any perpendicular axis
        axis = unit(np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def basis(forward: np.ndarray, side: np.ndarray) -> np.ndarray:
    e1 = unit(forward)
    e3 = unit(side - np.dot(side, e1) * e1)
    e2 = np.cross(e3, e1)
    return np.column_stack([e1, e2, e3])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0)))


# ---------------------------------------------------------------- rig
class Rig:
    """Node hierarchy of the avatar GLB (names as authored, e.g. 'mixamorig:RightArm')."""

    def __init__(self, glb: Path):
        data = glb.read_bytes()
        clen = struct.unpack("<I", data[12:16])[0]
        gltf = json.loads(data[20:20 + clen])
        self.nodes = gltf["nodes"]
        self.parent: Dict[int, Optional[int]] = {i: None for i in range(len(self.nodes))}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.index = {n["name"]: i for i, n in enumerate(self.nodes) if "name" in n}
        self.roots = gltf["scenes"][gltf.get("scene", 0)]["nodes"]
        self.rest_q = {i: list(n.get("rotation", [0, 0, 0, 1])) for i, n in enumerate(self.nodes)}
        self.rest_world = self.fk({})

    def local_matrix(self, i: int, q) -> np.ndarray:
        n = self.nodes[i]
        m = np.eye(4)
        m[:3, :3] = quat_to_mat(q) * np.array(n.get("scale", [1, 1, 1]))
        m[:3, 3] = n.get("translation", [0, 0, 0])
        return m

    def fk(self, overrides: Dict[str, List[float]]) -> Dict[int, np.ndarray]:
        """World matrices of all nodes; `overrides` maps a node name to a replacement local quaternion."""
        by_index = {self.index[k]: v for k, v in overrides.items() if k in self.index}
        world: Dict[int, np.ndarray] = {}

        def visit(i: int, parent: np.ndarray) -> None:
            world[i] = parent @ self.local_matrix(i, by_index.get(i, self.rest_q[i]))
            for c in self.nodes[i].get("children", []):
                visit(c, world[i])

        for r in self.roots:
            visit(r, np.eye(4))
        return world

    def pos(self, world, name: str) -> np.ndarray:
        return world[self.index[name]][:3, 3]

    def rot(self, world, name: str) -> np.ndarray:
        m = world[self.index[name]][:3, :3]
        return m / np.linalg.norm(m, axis=0)  # remove the 0.01 armature scale


def n(side: str, part: str) -> str:
    return f"mixamorig:{side}{part}"


# ---------------------------------------------------------------- solver
def _to_avatar(p) -> np.ndarray:
    """MediaPipe world axes (x image-right, y down, z smaller = nearer the camera) to avatar world (x right, y up, +z toward the camera)."""
    a = np.asarray(p, dtype=float)[..., :3]
    return a * np.array([1.0, -1.0, -1.0])


def _smooth(seq: List[Optional[np.ndarray]], window: int = 3) -> List[Optional[np.ndarray]]:
    """Centred moving average over available entries (None stays None). Window 3 keeps timing; it only tames frame-to-frame jitter."""
    out: List[Optional[np.ndarray]] = []
    h = window // 2
    for i, v in enumerate(seq):
        if v is None:
            out.append(None)
            continue
        group = [x for x in seq[max(0, i - h): i + h + 1] if x is not None]
        out.append(np.mean(group, axis=0))
    return out


def retarget(doc: Dict[str, object], rig: Rig) -> Dict[str, object]:
    frames = doc["frames"]
    rw = rig.rest_world
    rest = {side: {
        "arm": rig.rot(rw, n(side, "Arm")), "fore": rig.rot(rw, n(side, "ForeArm")), "hand": rig.rot(rw, n(side, "Hand")),
        "up0": unit(rig.pos(rw, n(side, "ForeArm")) - rig.pos(rw, n(side, "Arm"))),
        "fore0": unit(rig.pos(rw, n(side, "Hand")) - rig.pos(rw, n(side, "ForeArm"))),
        "hf0": unit(rig.pos(rw, n(side, "HandMiddle1")) - rig.pos(rw, n(side, "Hand"))),
        "hs0": unit(rig.pos(rw, n(side, "HandIndex1")) - rig.pos(rw, n(side, "HandPinky1"))),
        "parent_rot": rig.rot(rw, n(side, "Shoulder")),
    } for side in SIDES}

    # Per-frame raw targets (avatar axes), then 3-frame smoothing.
    raw: Dict[str, Dict[str, List[Optional[np.ndarray]]]] = {s: {"up": [], "fore": [], "hf": [], "hs": [], "angles": []} for s in SIDES}
    pose_ok: List[bool] = []
    for f in frames:
        pw = f.get("pose_world")
        pose_ok.append(bool(pw))
        P = _to_avatar(pw) if pw else None
        for side, cfg in SIDES.items():
            r = raw[side]
            if P is None:
                for k in ("up", "fore", "hf", "hs", "angles"):
                    r[k].append(None)
                continue
            r["up"].append(unit(P[cfg["elbow"]] - P[cfg["shoulder"]]))
            r["fore"].append(unit(P[cfg["wrist"]] - P[cfg["elbow"]]))
            hw = f.get(cfg["hand_key"].replace("_hand", "_hand_world"))
            if hw:
                H = _to_avatar(hw)
                r["hf"].append(unit(H[9] - H[0]))
                r["hs"].append(unit(H[5] - H[17]))
                ang = {}
                for name, mcp in FINGERS.items():
                    p = unit(H[mcp] - H[0])
                    s1, s2, s3 = unit(H[mcp + 1] - H[mcp]), unit(H[mcp + 2] - H[mcp + 1]), unit(H[mcp + 3] - H[mcp + 2])
                    ang[name] = np.array([min(angle_between(p, s1), MAX_FLEX), min(angle_between(s1, s2), MAX_FLEX), min(angle_between(s2, s3), MAX_FLEX)])
                r["angles"].append(ang)
            else:  # palm frame from the pose's wrist/index/pinky when the hand model found nothing
                mid = (P[cfg["index"]] + P[cfg["pinky"]]) / 2
                r["hf"].append(unit(mid - P[cfg["wrist"]]))
                r["hs"].append(unit(P[cfg["index"]] - P[cfg["pinky"]]))
                r["angles"].append(None)
    for side in SIDES:
        for k in ("up", "fore", "hf", "hs"):
            raw[side][k] = _smooth(raw[side][k])
        # finger angles: smooth each finger separately
        for name in FINGERS:
            seq = [a[name] if a else None for a in raw[side]["angles"]]
            sm = _smooth(seq)
            for i, v in enumerate(sm):
                if raw[side]["angles"][i] is not None:
                    raw[side]["angles"][i][name] = v

    out_frames = []
    metrics = {s: {"wrist_err": [], "elbow_err": []} for s in SIDES}
    shoulder_w_avatar = np.linalg.norm(rig.pos(rw, n("Right", "Arm")) - rig.pos(rw, n("Left", "Arm")))
    for i, f in enumerate(frames):
        # Every driven bone is written every frame (rest when unknown), so no pose from an earlier frame lingers during playback.
        bones: Dict[str, List[float]] = {}
        for side in SIDES:
            for part in ("Arm", "ForeArm", "Hand") + tuple(f"Hand{fn}{j}" for fn in FINGERS for j in (1, 2, 3)):
                bones[n(side, part)] = list(rig.rest_q[rig.index[n(side, part)]])
        flags = {"pose": pose_ok[i]}
        if pose_ok[i]:
            pw = _to_avatar(f["pose_world"])
            signer_w = np.linalg.norm(pw[12] - pw[11])
            for side, cfg in SIDES.items():
                r, rs = raw[side], rest[side]
                R_arm = align(rs["up0"], r["up"][i]) @ rs["arm"]
                bones[n(side, "Arm")] = mat_to_quat(rs["parent_rot"].T @ R_arm)
                R_fore = align(rs["fore0"], r["fore"][i]) @ rs["fore"]
                bones[n(side, "ForeArm")] = mat_to_quat(R_arm.T @ R_fore)
                ang = r["angles"][i]
                flags[cfg["hand_key"]] = ang is not None
                if ang is not None:
                    # Palm orientation only when the hand model actually saw the hand; otherwise the wrist stays relaxed (rest, relative to the forearm).
                    R_hand = (basis(r["hf"][i], r["hs"][i]) @ basis(rs["hf0"], rs["hs0"]).T) @ rs["hand"]
                    bones[n(side, "Hand")] = mat_to_quat(R_fore.T @ R_hand)
                if ang is not None:
                    for name in FINGERS:
                        for j in range(3):
                            bone = n(side, f"Hand{name}{j + 1}")
                            rq = quat_to_mat(rig.rest_q[rig.index[bone]])
                            bones[bone] = mat_to_quat(rq @ rot_z(cfg["sign"] * float(ang[name][j])))
                # Self-check with forward kinematics: where do the avatar's elbow and wrist end up compared with the (scaled) source?
                world = rig.fk(bones)
                scale = shoulder_w_avatar / max(signer_w, 1e-6)
                a_sh = rig.pos(world, n(side, "Arm"))
                arm_len = np.linalg.norm(rig.pos(rw, n(side, "Hand")) - rig.pos(rw, n(side, "Arm")))
                for key, lm, bone in (("elbow_err", cfg["elbow"], "ForeArm"), ("wrist_err", cfg["wrist"], "Hand")):
                    want = (pw[lm] - pw[cfg["shoulder"]]) * scale
                    got = rig.pos(world, n(side, bone)) - a_sh
                    metrics[side][key].append(float(np.linalg.norm(want - got) / arm_len))
        out_frames.append({"t": f["t"], "flags": flags, "bones": {k: [round(x, 5) for x in v] for k, v in bones.items()}})

    summary = {}
    for side in SIDES:
        e = metrics[side]
        summary[side.lower()] = {k: ({"mean": round(float(np.mean(v)), 3), "max": round(float(np.max(v)), 3)} if v else None) for k, v in e.items()}
    return {
        "format": "islrtc-bone-track", "format_version": FORMAT_VERSION, "experimental": True,
        "rig": "yap-render public/models/ybot.glb", "source": doc["source"], "video": doc["video"],
        "method": "direction-based arm solve + palm-frame hand + finger flexion about local Z (right +, left -); 3-frame smoothing; thumbs, shoulders, spine, head, face untouched",
        "bone_quaternion_convention": "local rotation [x,y,z,w] to assign to bone.quaternion (three.js); bone names without the colon, as three.js sanitises them",
        "position_error_note": "elbow/wrist position error of the avatar vs the shoulder-width-scaled source, as a fraction of the avatar's arm length (not a measure of sign correctness)",
        "position_error": summary,
        "frames": [{"t": fr["t"], "flags": fr["flags"], "bones": {k.replace(":", ""): v for k, v in fr["bones"].items()}} for fr in out_frames],
    }
