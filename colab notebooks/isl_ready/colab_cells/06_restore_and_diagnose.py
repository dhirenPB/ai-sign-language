# RESTORE + DIAGNOSE (v3) — rebuild the session state, create the NEW normalisation (per-sequence image size), re-run the diagnostics,
# and rebuild the pilot. No remount, no re-clone, no re-patch.
import copy, gc, importlib, itertools, json, os, random, re, shutil, subprocess, sys, time, zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd

# ---------- settings (same as CELL 1; Drive is already mounted) ----------
DATA_DIR = Path("/content/drive/MyDrive/ISL_Project/iSign")
CSV_PATH = DATA_DIR / "iSign_v1.1.csv"
POSE_DIR = DATA_DIR / "pose_parts"
POSE_PART_NAMES = [f"iSign-poses_v1.1_part_{s}" for s in ("aa", "ab", "ac", "ad")]
CODE_DIR = Path("/content/isl_code")
REPO_URL = "https://github.com/BenSaunders27/ProgressiveTransformersSLP.git"
REPO_DIR = Path("/content/ProgressiveTransformersSLP")
WORK_DIR = Path("/content/isl_work")
DRIVE_OUT = Path("/content/drive/MyDrive/ISL_Project/pre_training_ready")
PROC_PARENT = DRIVE_OUT / "processed"
SEED = 27
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
MIN_MATCH_RATE = 0.9
SHARD_SIZE = 256
CONVERT_MAX_SHARDS = {"train": 4, "val": 1, "test": 1}
FULL_CONVERT = False
STORAGE_DTYPE = "float16"
QUALITY = {"min_frames": 2, "max_frames": 600, "min_valid_fraction": 0.15,
           "max_empty_frame_fraction": 0.5, "require_hands": True}
PE_MAX_LEN = 5000
CONF_THRESHOLD = 0.5
ZERO_FILL_IS_MISSING = True
Z_MODE = "keep"
DIAG_N_SEQUENCES = 300
PILOT_N = 5
BATCH_SIZE = 8
PILOT_BATCH_SIZE = 2
NUM_WORKERS = 2
MIN_FREE_GB = 8
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
REQUIRE_GPU_FOR_TRAINING = True
OVERFIT_N, OVERFIT_STEPS, OVERFIT_MAX_FRAMES, OVERFIT_BATCH = 32, 300, 250, 8
RUN_FULL_TRAINING = False
RUN_TEST_EVAL = False
LOCAL_SCAN_ROOTS = ["/content"]        # where check 1 looks for a combined multi-GB archive (it must not exist)
DISK_GUARD_PATH = "/"                  # the disk whose free space the conversion watches
for _k, _v in globals().get("ISL_OVERRIDES", {}).items():       # another front end (for example the PC runner) may change paths and sizes
    globals()[_k] = _v
for _d in (WORK_DIR, DRIVE_OUT, PROC_PARENT):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- modules and libraries ----------
import torch
try:
    from pose_format import Pose
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pose-format"], check=True)
    from pose_format import Pose
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for _m in ("isign_pose", "isign_report", "repo_patches"):
    sys.modules.pop(_m, None)
importlib.invalidate_caches()
import isign_pose as ip
import isign_report as rep
import repo_patches as rp
assert ip.DEFAULT_NORMALIZATION["xy"]["mode"] == "image_size", "old isign_pose loaded: run the UPLOAD cell first"
CHECKS = rep.Checks()
assert (REPO_DIR / "_originals").exists(), "the repository clone has not been patched in this VM; tell me"


def disk_report():
    def sh(cmd):
        return subprocess.run(cmd + " 2>/dev/null", shell=True, capture_output=True, text=True).stdout.strip()
    print(sh("df -h / | tail -n 2"))
    print(f"\nfree on /: {shutil.disk_usage('/').free / 1e9:.1f} GB\n")
    for path in ("/content", "/root/.cache", "/root/.config/Google/DriveFS", "/tmp"):
        print(sh(f"du -xsh {path}") or f"{path}: not present")


# ---------- archive access (check 1) ----------
archive = virtual_file = None
pose_names, pose_by_stem, KEEP_IDX = [], {}, ip.EXPECTED_KEEP_IDX
with CHECKS("1. Dataset archive access (no combined archive)") as chk:
    parts = [POSE_DIR / n for n in POSE_PART_NAMES]
    absent = [str(p) for p in parts if not p.exists()]
    assert not absent, f"pose parts missing on Drive: {absent}"
    virtual_file = ip.VirtualConcatFile(parts)
    assert virtual_file.total_size == sum(p.stat().st_size for p in parts)
    archive = zipfile.ZipFile(virtual_file, "r")
    pose_names = [n for n in archive.namelist() if n.lower().endswith(".pose")]
    assert pose_names, "no .pose entries in the archive"
    pose_by_stem = {PurePosixPath(n).stem: n for n in pose_names}
    first_pose = Pose.read(archive.read(pose_names[0]))
    KEEP_IDX, layout_note = ip.resolve_landmark_layout(first_pose)
    print("Landmark layout:", layout_note)
    large = []
    for scan_root in LOCAL_SCAN_ROOTS:
        for root, dirs, files in os.walk(scan_root):
            dirs[:] = [d for d in dirs if d != "drive"]
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if os.path.getsize(fp) > 1e9:
                        large.append(fp)
                except OSError:
                    pass
    assert not large, f"large local files found (a combined archive?): {large}"
    chk.note(f"{len(pose_names):,} pose entries from {len(parts)} parts ({virtual_file.total_size / 1e9:.1f} GB), read in place")
print(f"Archive entries: {len(pose_names):,} | local disk free: {shutil.disk_usage('/').free / 1e9:.0f} GB")

# ---------- matching (check 2) ----------
df = pd.read_csv(CSV_PATH)
assert {"uid", "text"} <= set(df.columns), f"Unexpected CSV columns: {df.columns.tolist()}"
df["uid"] = df["uid"].astype(str)
n_rows = len(df)
duplicate_uids = df.loc[df["uid"].duplicated(), "uid"].tolist()
df = df.drop_duplicates("uid", keep="first").reset_index(drop=True)
df["text_norm"] = df["text"].astype(str).map(ip.normalize_text)
df["video_id"] = df["uid"].map(ip.video_id_of)
stem_set = set(pose_by_stem)
has_pose = df["uid"].isin(stem_set)
empty_text = df.loc[df["text_norm"] == "", "uid"].tolist()
unmatched_csv = df.loc[~has_pose, "uid"].tolist()
pose_without_csv = sorted(stem_set - set(df["uid"]))
matched = df[has_pose & (df["text_norm"] != "")].reset_index(drop=True)
LOG_DIR = DRIVE_OUT / "logs"
LOG_DIR.mkdir(exist_ok=True)
for fname, items in (("unmatched_csv_uids.txt", unmatched_csv), ("pose_files_without_csv_row.txt", pose_without_csv),
                     ("empty_text_uids.txt", empty_text), ("duplicate_uids.txt", duplicate_uids)):
    (LOG_DIR / fname).write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")
print(f"CSV rows: {n_rows:,} | matched to a pose file: {len(matched):,} ({len(matched) / max(len(df), 1):.1%}) | "
      f"CSV uids without a pose file: {len(unmatched_csv):,} | pose files without a CSV row: {len(pose_without_csv):,}")
with CHECKS("2. UID-to-pose matching") as chk:
    rate = len(matched) / max(len(df), 1)
    assert rate >= MIN_MATCH_RATE, f"only {rate:.1%} of CSV rows have a pose file (< {MIN_MATCH_RATE:.0%})"
    assert matched["uid"].is_unique
    rng = np.random.RandomState(SEED)
    for i in rng.choice(len(matched), min(10, len(matched)), replace=False):
        p = Pose.read(archive.read(pose_by_stem[matched.loc[i, "uid"]]))
        assert p.body.data.shape[0] > 0, f"{matched.loc[i, 'uid']}: pose has no frames"
    chk.note(f"{len(matched):,}/{len(df):,} matched ({rate:.1%}); {len(unmatched_csv)} CSV uids and "
             f"{len(pose_without_csv)} pose files unmatched (logged); 10 random matched poses read OK")

# ---------- split by video (check 8) ----------
uids_by_split, videos_by_split = ip.split_by_video(matched["uid"].tolist(), SEED, SPLIT_FRACTIONS, SPLITS)
text_of = dict(zip(matched["uid"], matched["text_norm"]))
manifests = {s: pd.DataFrame({"uid": u, "video_id": [ip.video_id_of(x) for x in u], "text": [text_of[x] for x in u]})
             for s, u in uids_by_split.items()}
rows_of = {s: list(zip(m["uid"], m["text"])) for s, m in manifests.items()}
SPLIT_DIR = DRIVE_OUT / "splits"
SPLIT_DIR.mkdir(exist_ok=True)
for s, m in manifests.items():                       # a saved manifest must equal the recomputed split; the first run writes them
    manifest_path = SPLIT_DIR / f"manifest_{s}.csv"
    if manifest_path.exists():
        saved = pd.read_csv(manifest_path)
        assert saved["uid"].tolist() == m["uid"].tolist(), f"{s}: recomputed split differs from the saved manifest"
    else:
        m.to_csv(manifest_path, index=False)
split_table = pd.DataFrame(ip.split_report(uids_by_split, videos_by_split, SPLIT_FRACTIONS, len(matched)))
split_table["share_of_sequences"] = split_table["share_of_sequences"].map("{:.1%}".format)
split_table["target_share"] = split_table["target_share"].map("{:.0%}".format)
display(split_table)
with CHECKS("8. Train/val/test separation by video ID") as chk:
    assert not ip.split_overlaps(videos_by_split), "a video occurs in more than one split"
    vid_sets = {s: set(m["video_id"]) for s, m in manifests.items()}
    uid_sets = {s: set(m["uid"]) for s, m in manifests.items()}
    for a, b in itertools.combinations(SPLITS, 2):
        assert not (vid_sets[a] & vid_sets[b]), f"{a}/{b} share videos"
        assert not (uid_sets[a] & uid_sets[b]), f"{a}/{b} share uids"
    assert sum(len(u) for u in uid_sets.values()) == len(matched)
    again, _ = ip.split_by_video(matched["uid"].tolist(), SEED, SPLIT_FRACTIONS, SPLITS)
    assert again == uids_by_split, "split is not reproducible for a fixed seed"
    for row in ip.split_report(uids_by_split, videos_by_split, SPLIT_FRACTIONS, len(matched)):
        assert row["sequences"] > 0, f"{row['split']} is empty"
        assert abs(row["share_of_sequences"] - row["target_share"]) < 0.05, f"{row['split']} share is far from the target"
    chk.note("; ".join(f"{s}: {len(uid_sets[s]):,} seq / {len(vid_sets[s]):,} videos" for s in SPLITS))

# ---------- NEW normalisation: x/y per sequence with the image size from each pose header ----------
NORM = ip.make_normalization(Z_MODE, None, CONF_THRESHOLD, ZERO_FILL_IS_MISSING, STORAGE_DTYPE)
assert NORM["xy"]["mode"] == "image_size" and Z_MODE == "keep"
NORM_HASH = ip.params_hash(NORM)
print("\nnew normalisation hash:", NORM_HASH, "|", NORM["xy"]["formula"])

# ---------- diagnostics on a random training sample, read in archive order ----------
def read_pose_by_uid(uid):
    name = pose_by_stem.get(uid)
    return None if name is None else Pose.read(archive.read(name))

offset_of = {u: archive.getinfo(pose_by_stem[u]).header_offset for u in matched["uid"]}
convert_rows_of = {s: sorted(rows, key=lambda r: offset_of[r[0]]) for s, rows in rows_of.items()}
REUSE_DIAGNOSTICS = True      # a saved report made with the same normalisation is reused (saves ~10 minutes of reading and Drive cache)
diag_path = DRIVE_OUT / "diagnostics" / "diagnostics_report.json"
saved_report = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else None
if REUSE_DIAGNOSTICS and saved_report is not None and saved_report.get("normalization_hash") == NORM_HASH:
    DIAG_REPORT = saved_report
    flags = [(f["severity"], f["message"]) for f in saved_report["flags"]]
    print("reusing the saved diagnostics (same normalisation hash); set REUSE_DIAGNOSTICS = False to re-read the sample")
else:
    train_manifest = manifests["train"]
    rng = np.random.RandomState(SEED)
    pick = rng.choice(len(train_manifest), min(DIAG_N_SEQUENCES, len(train_manifest)), replace=False)
    pick = sorted(pick, key=lambda i: offset_of[train_manifest.loc[int(i), "uid"]])
    diag_raws, diag_unreadable, t0 = [], 0, time.time()
    for n, i in enumerate(pick, 1):
        try:
            raw = ip.extract_raw(read_pose_by_uid(train_manifest.loc[int(i), "uid"]), KEEP_IDX)
        except Exception:
            raw = None
        if raw is None:
            diag_unreadable += 1
            continue
        diag_raws.append(raw)
        if n % 50 == 0:
            print(f"  read {n}/{len(pick)} sequences ({time.time() - t0:.0f}s)")
    assert len(diag_raws) >= 30, "too few readable sequences for meaningful diagnostics"
    DIAG_REPORT, flags = ip.diagnose_sample(diag_raws, NORM)
report = DIAG_REPORT
bd = report["landmark_breakdown"]
print(f"\n{report['n_sequences']} sequences, {report['n_frames']:,} frames | valid landmarks {bd['valid_fraction']:.1%} | non-finite {bd['nonfinite']} | "
      f"zero-filled but confident {bd['zero_filled_but_confident']}")
print("image sizes (width, height, sequences):", [(s["width"], s["height"], s["sequences"]) for s in report["image_sizes"]])
print("% of valid x/y outside their own image:", {c: None if r["pct_xy_outside_nominal_bounds"] is None else round(r["pct_xy_outside_nominal_bounds"], 2) for c, r in report["components"].items()})
print("z spread / x spread:", {c: None if r["z_iqr_over_x_iqr"] is None else round(r["z_iqr_over_x_iqr"], 3) for c, r in report["components"].items()})
print("wrist pairing (median, normalised):", {k: None if v["median_norm"] is None else round(v["median_norm"], 3) for k, v in report["wrist_pairing"].items()})
print("frame-to-frame displacement percentiles:", {k: round(v, 3) for k, v in report["temporal"]["frame_to_frame_xy_displacement"].items()},
      "| sequences with a jump:", report["temporal"]["sequences_with_a_jump"])
print("\nlandmark groups (share of the target variance):")
display(pd.DataFrame(report["landmark_groups"]).round(1))
print("flags:")
for sev, msg in flags:
    print(f"  [{sev}] {msg}")
if not flags:
    print("  none")

NORM_JSON = json.dumps(NORM, indent=1, sort_keys=True)
(WORK_DIR / "normalization.json").write_text(NORM_JSON, encoding="utf-8")
(DRIVE_OUT / f"normalization_{NORM_HASH}.json").write_text(NORM_JSON, encoding="utf-8")
(DRIVE_OUT / "diagnostics").mkdir(exist_ok=True)
(DRIVE_OUT / "diagnostics" / "diagnostics_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
PROC_ROOT = PROC_PARENT / f"norm_{NORM_HASH}_archive_order"
PROC_ROOT.mkdir(exist_ok=True)

# ---------- the isolated pilot, rebuilt with the new normalisation ----------
PILOT_DIR = WORK_DIR / "pilot"
if PILOT_DIR.exists():
    shutil.rmtree(PILOT_DIR)
pilot_rows = list(zip(matched["uid"].head(PILOT_N * 3), matched["text_norm"].head(PILOT_N * 3)))
print("\npilot:", ip.convert_manifest(pilot_rows, PILOT_DIR, read_pose_by_uid, NORM, QUALITY, shard_size=len(pilot_rows), log=lambda m: None))
print(f"\nconverted data will go to {PROC_ROOT}\n")
CHECKS.print_table()
