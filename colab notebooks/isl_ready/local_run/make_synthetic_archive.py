"""
Build a small SYNTHETIC stand-in for the iSign data so the notebook can be executed locally on a CPU.

It reproduces the structure the notebook depends on, not the real data:
  * iSign_v1.1.csv with uids like '<12 hex>-<n>' (unpadded sequence numbers, occasional gaps) and text,
  * a ZIP of real pose-format .pose files (576 MediaPipe-holistic landmarks, header with the five components)
    cut into four unevenly sized parts, exactly like the real pose_parts,
  * invalid landmarks of every kind (low confidence, masked, zero-filled), out-of-bounds coordinates,
    pixel-scale x/y and small hand z, a few deliberately bad sequences, unmatched CSV rows and pose files.
Usage: python make_synthetic_archive.py <output root>   (creates <root>/drive/MyDrive/ISL_Project/iSign/...)
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

ROOT = Path(sys.argv[1])
N_VIDEOS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rng = np.random.default_rng(7)

COMPONENTS = [("POSE_LANDMARKS", 33), ("FACE_LANDMARKS", 468), ("LEFT_HAND_LANDMARKS", 21),
              ("RIGHT_HAND_LANDMARKS", 21), ("POSE_WORLD_LANDMARKS", 33)]
def make_header(width, height):
    return PoseHeader(
        version=0.1, dimensions=PoseHeaderDimensions(width=width, height=height, depth=0),
        components=[PoseHeaderComponent(name, [f"{name}_{i}" for i in range(n)], [], [(255, 255, 255)], "XYZC") for name, n in COMPONENTS])


# like the real data: most files are 300 x 300, a few keep another image size (coordinates in that frame's pixels)
OTHER_SIZES = [(648, 720), (450, 396), (696, 720), (480, 614)]
PHRASES = ["Page 111", "Make it shorter.", "One day, Akbar drew a line on the floor and ordered.", "No one knew what to do.",
           "Each minister looked at the line and was puzzled.", "Say aloud", "Team Time!", "Reading is fun.",
           "What did Akbar order one day?", "which letters cannot be heard when we say these words.", "Short. Shorter. Bright.",
           "Discuss with your partner and draw pictures to show the following. 1.", "New words Order", "Rub.", "Puzzle"]


def make_pose(n_frames, kind="normal", size=(300, 300)):
    T = n_frames
    t = np.linspace(0, 1, T)[:, None]
    data = np.zeros((T, 1, 576, 3), np.float32)
    conf = np.ones((T, 1, 576), np.float32)
    mask = np.zeros(data.shape, bool)
    # body: standing signer, pixel units, slight drift
    k = np.arange(33)
    data[:, 0, :33, 0] = 158 + 45 * np.sin(k * 0.7)[None] + 8 * np.sin(2 * np.pi * t)
    data[:, 0, :33, 1] = 60 + 9 * k[None] + 6 * np.cos(2 * np.pi * t)
    data[:, 0, :33, 2] = rng.normal(-1.2, 0.8, size=(T, 33))
    # a tail of landmarks far outside the nominal image (legs below the frame, wide arms)
    data[:, 0, 30:33, 1] += rng.uniform(150, 500, size=(1, 3))        # feet far below the frame (~4% of landmarks)
    lw = np.concatenate([110 + 40 * np.sin(4 * np.pi * t), 150 + 30 * np.cos(3 * np.pi * t)], axis=1)
    rw = np.concatenate([205 + 40 * np.cos(4 * np.pi * t), 150 + 30 * np.sin(3 * np.pi * t)], axis=1)
    data[:, 0, 15, :2], data[:, 0, 16, :2] = lw, rw
    for base, wrist in ((501, lw), (522, rw)):                 # hands sit on the wrists
        data[:, 0, base:base + 21, :2] = wrist[:, None, :] + rng.normal(0, 6, size=(T, 21, 2))
        data[:, 0, base:base + 21, 2] = rng.normal(-0.03, 0.03, size=(T, 21))
        data[:, 0, base, 2] = 0.0                              # hand z is wrist-relative
    data[:, 0, 33:501, :2] = 150 + rng.normal(0, 20, size=(T, 468, 2))
    data[:, 0, 543:, :] = rng.normal(0, 0.3, size=(T, 33, 3))
    # missing data of every kind
    low = rng.random((T, 1, 576)) < 0.12
    conf[low] = rng.uniform(0.0, 0.45, size=int(low.sum()))
    gone = rng.random((T, 1, 576)) < 0.06                      # not detected: zero-filled, confidence 0, masked
    data[gone] = 0.0
    conf[gone] = 0.0
    mask[np.broadcast_to(gone[..., None], data.shape)] = True
    if kind == "no_hands":
        conf[:, 0, 501:543] = 0.0
        data[:, 0, 501:543] = 0.0
    if kind == "low_confidence":
        conf[:] = 0.1
    data[..., 0] *= size[0] / 300.0                          # same scene, another image size (zeros stay zero)
    data[..., 1] *= size[1] / 300.0
    return Pose(make_header(*size), NumPyPoseBody(25, np.ma.array(data, mask=mask), conf))


def pose_bytes(pose):
    buf = io.BytesIO()
    pose.write(buf)
    return buf.getvalue()


entries, rows = {}, []
for v in range(N_VIDEOS):
    vid = "".join(rng.choice(list("0123456789abcdef"), 12))
    size = OTHER_SIZES[int(rng.integers(len(OTHER_SIZES)))] if rng.random() < 0.08 else (300, 300)
    seq = 1
    for _ in range(int(rng.integers(2, 7))):
        uid = f"{vid}-{seq}"
        seq += int(rng.choice([1, 1, 1, 2]))                  # occasional gaps like the real data
        T = int(rng.integers(25, 90))
        kind = "normal"
        roll = rng.random()
        if roll < 0.02:
            kind = "no_hands"
        elif roll < 0.04:
            kind = "low_confidence"
        elif roll < 0.05:
            T = 700                                            # longer than MAX_FRAMES
        elif roll < 0.06:
            T = 1                                              # too short
        entries[f"poses/{uid}.pose"] = pose_bytes(make_pose(T, kind, size))
        rows.append((uid, str(rng.choice(PHRASES)) + (" " if rng.random() < 0.1 else "")))
for i in range(5):                                            # CSV rows without a pose file
    rows.append((f"{'ab12cd34ef56'}-{i + 1}", "orphan row"))
for i in range(3):                                            # pose files without a CSV row
    entries[f"poses/orphanpose{i}-1.pose"] = pose_bytes(make_pose(30))

base = ROOT / "drive" / "MyDrive" / "ISL_Project" / "iSign"
(base / "pose_parts").mkdir(parents=True, exist_ok=True)
with open(base / "iSign_v1.1.csv", "w", encoding="utf-8", newline="") as f:
    f.write("uid,text\n")
    for uid, text in rows:
        f.write(f'{uid},"{text}"\n')
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
    for name in rng.permutation(list(entries)):                # archive order differs from CSV order
        z.writestr(str(name), entries[str(name)])
blob = buf.getvalue()
cuts = [0, int(len(blob) * 0.22), int(len(blob) * 0.41), int(len(blob) * 0.80), len(blob)]
for i, s in enumerate(("aa", "ab", "ac", "ad")):
    (base / "pose_parts" / f"iSign-poses_v1.1_part_{s}").write_bytes(blob[cuts[i]:cuts[i + 1]])
print(f"{len(entries)} pose files, {len(rows)} CSV rows, {len(blob) / 1e6:.0f} MB archive in 4 parts -> {base}")
