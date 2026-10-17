"""
iSign pose preprocessing, video-level splitting and shard storage.

Numpy-only on purpose (no torch) so that the logic can be tested without a GPU environment.

Frame convention -- do not change silently:
  * 75 landmarks = 33 body + 21 left hand + 21 right hand
    = indices 0:33 and 501:543 of the 576-landmark pose-format layout
  * 225 coordinates per frame, landmark-major: [x0, y0, z0, x1, y1, z1, ...]
  * 1 frame counter appended as column 225  ->  frame width 226
  * counter = frame_index / n_frames   (first frame 0.0, last frame (n - 1) / n)
  * x, y: per sequence, (x - width/2) / (width/2) and (y - height/2) / (height/2) with the image size from the pose
    header (most iSign files are 300 x 300, i.e. (v - 150) / 150; a few are not);  z: kept in source units unless the
    normalization says otherwise
  * invalid landmarks are stored as 0.0 *and* flagged in a separate validity array; the loss uses
    the flags, never the values
"""
import bisect
import hashlib
import io
import json
import math
import os
import shutil
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- representation
N_BODY, N_HAND = 33, 21
N_LANDMARKS = N_BODY + 2 * N_HAND            # 75
N_COORDS = N_LANDMARKS * 3                   # 225
FRAME_WIDTH = N_COORDS + 1                   # 226 (coordinates + counter)
EXPECTED_KEEP_IDX = np.r_[0:33, 501:522, 522:543]   # source landmark indices that are kept

# component slices inside the kept 75-landmark layout
COMPONENTS = OrderedDict([("body", (0, 33)), ("left_hand", (33, 54)), ("right_hand", (54, 75))])
BODY_L_WRIST, BODY_R_WRIST = 15, 16          # MediaPipe pose indices
LH_WRIST, RH_WRIST = 33, 54                  # hand wrists inside the kept layout

DEFAULT_NORMALIZATION = {
    "version": 1,
    "landmarks": "pose-format indices 0:33 (body) + 501:522 (left hand) + 522:543 (right hand)",
    "n_landmarks": N_LANDMARKS,
    "coord_layout": "landmark-major [x0, y0, z0, x1, y1, z1, ...]",
    "frame_width": FRAME_WIDTH,
    "counter": "frame_index / n_frames in the last column (first frame 0.0)",
    # image_size: every pose file carries its own image size in the header (most are 300 x 300, some are not), so
    # x -> (x - width/2) / (width/2) and y -> (y - height/2) / (height/2). For 300 x 300 this is exactly (v - 150) / 150.
    "xy": {"mode": "image_size", "default_size": 300.0,
           "formula": "x: (v - width/2) / (width/2); y: (v - height/2) / (height/2); width and height from each pose header"},
    "z": {"mode": "keep",
          "scale": {"body": 1.0, "left_hand": 1.0, "right_hand": 1.0},
          "formula": "value / scale[component]  (scale is 1.0 in 'keep' mode)"},
    "confidence_threshold": 0.5,
    "zero_fill_is_missing": True,
    "clamp": None,
    "missing_fill": 0.0,
    "storage_dtype": "float16",
}

DEFAULT_QUALITY = {
    "min_frames": 2,
    "max_frames": 600,
    "min_valid_fraction": 0.15,        # share of valid landmarks over the whole sequence
    "max_empty_frame_fraction": 0.5,   # frames with no valid landmark at all
    "require_hands": True,             # at least one valid hand landmark somewhere
}


def make_normalization(z_mode="keep", z_scale=None, confidence_threshold=0.5,
                       zero_fill_is_missing=True, storage_dtype="float16", xy_mode="image_size"):
    params = json.loads(json.dumps(DEFAULT_NORMALIZATION))
    if xy_mode == "constant":          # the earlier assumption: every frame is 300 x 300
        params["xy"] = {"mode": "constant", "center": 150.0, "half_range": 150.0, "formula": "(v - center) / half_range"}
    elif xy_mode != "image_size":
        raise ValueError("xy_mode must be 'image_size' or 'constant'")
    if z_mode not in ("keep", "scale"):
        raise ValueError("z_mode must be 'keep' or 'scale'")
    params["z"]["mode"] = z_mode
    if z_mode == "scale":
        if not z_scale:
            raise ValueError("z_mode='scale' needs z_scale={'body': .., 'left_hand': .., 'right_hand': ..}")
        for name in COMPONENTS:
            params["z"]["scale"][name] = float(z_scale[name])
            if not params["z"]["scale"][name] > 0:
                raise ValueError("z scales must be positive")
    params["confidence_threshold"] = float(confidence_threshold)
    params["zero_fill_is_missing"] = bool(zero_fill_is_missing)
    params["storage_dtype"] = str(np.dtype(storage_dtype))
    return params


def params_hash(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def header_dims(pose, default=300.0):
    """(width, height) from the pose header; falls back to default x default if missing or not positive."""
    try:
        d = pose.header.dimensions
        w, h = float(d.width), float(d.height)
        if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return (float(default), float(default))


def xy_geometry(params, dims=None):
    """(cx, cy, hx, hy): centre and half range used for x and y of one sequence."""
    xy = params["xy"]
    if xy["mode"] == "constant":
        return xy["center"], xy["center"], xy["half_range"], xy["half_range"]
    w, h = dims if dims is not None else (xy["default_size"], xy["default_size"])
    return w / 2.0, h / 2.0, w / 2.0, h / 2.0


def z_scale_vector(params):
    """Per-landmark z divisor, shape [75]."""
    vec = np.ones(N_LANDMARKS, dtype=np.float32)
    if params["z"]["mode"] == "scale":
        for name, (a, b) in COMPONENTS.items():
            vec[a:b] = params["z"]["scale"][name]
    return vec


# ----------------------------------------------------------------------------- archive access
class VirtualConcatFile(io.BufferedIOBase):
    """Seekable, read-only view of several files joined end to end (nothing is copied or combined)."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.files = [open(p, "rb") for p in self.paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.starts, total = [], 0
        for size in self.sizes:
            self.starts.append(total)
            total += size
        self.total_size = total
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.total_size + offset
        else:
            raise ValueError("Invalid whence")
        if new_pos < 0:
            raise ValueError("Negative seek position")
        self.pos = new_pos
        return self.pos

    def read(self, size=-1):
        if self.pos >= self.total_size:
            return b""
        if size is None or size < 0:
            size = self.total_size - self.pos
        size = min(size, self.total_size - self.pos)
        chunks = []
        while size > 0:
            idx = bisect.bisect_right(self.starts, self.pos) - 1
            local_pos = self.pos - self.starts[idx]
            take = min(size, self.sizes[idx] - local_pos)
            f = self.files[idx]
            f.seek(local_pos)
            chunk = f.read(take)
            if not chunk:
                break
            chunks.append(chunk)
            self.pos += len(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self):
        for f in self.files:
            f.close()
        super().close()


# ----------------------------------------------------------------------------- pose -> arrays
def resolve_landmark_layout(pose):
    """
    Return (keep_idx, note). The kept landmarks are fixed (EXPECTED_KEEP_IDX); the file header is only
    used to *verify* that they are the body / left-hand / right-hand components. If the header says
    something different the function raises instead of silently selecting other landmarks.
    """
    note = "header not inspected; using the fixed indices"
    try:
        ranges, offset = {}, 0
        for comp in pose.header.components:
            n = len(comp.points)
            ranges[comp.name] = (offset, offset + n)
            offset += n
        body, left, right = ranges["POSE_LANDMARKS"], ranges["LEFT_HAND_LANDMARKS"], ranges["RIGHT_HAND_LANDMARKS"]
        sizes = (body[1] - body[0], left[1] - left[0], right[1] - right[0])
        total = pose.body.data.shape[2]
        if sizes == (N_BODY, N_HAND, N_HAND) and offset == total:
            idx = np.r_[body[0]:body[1], left[0]:left[1], right[0]:right[1]]
            if not np.array_equal(idx, EXPECTED_KEEP_IDX):
                raise ValueError(
                    "The pose header places body/hands at different indices than the expected "
                    f"0:33 + 501:543 (got body {body}, left {left}, right {right}). Investigate before continuing.")
            return EXPECTED_KEEP_IDX.copy(), f"verified from header: body {body}, left hand {left}, right hand {right}"
        note = f"header sizes {sizes} / total {total} not as expected; using the fixed indices"
    except ValueError:
        raise
    except Exception as exc:  # header missing or unusual
        note = f"header unreadable ({exc!r}); using the fixed indices"
    return EXPECTED_KEEP_IDX.copy(), note


def extract_raw(pose, keep_idx=EXPECTED_KEEP_IDX):
    """
    Raw (un-normalised) values of the kept landmarks for the first person, with every kind of
    'missing' kept apart. Returns None when the pose has no frames.
    """
    data, conf = pose.body.data, pose.body.confidence
    d, dmask = np.ma.getdata(data), np.ma.getmaskarray(data)
    c, cmask = np.ma.getdata(conf), np.ma.getmaskarray(conf)
    if d.ndim != 4 or d.shape[0] == 0 or d.shape[1] == 0:
        return None
    xyz = d[:, 0][:, keep_idx, :].astype(np.float32)           # [T, 75, 3]
    cf = c[:, 0][:, keep_idx].astype(np.float32)               # [T, 75]
    masked = dmask[:, 0][:, keep_idx, :].any(axis=-1) | cmask[:, 0][:, keep_idx]
    nonfinite = ~np.isfinite(xyz).all(axis=-1) | ~np.isfinite(cf)
    zero_filled = (xyz == 0).all(axis=-1)
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    cf = np.nan_to_num(cf, nan=0.0, posinf=0.0, neginf=0.0)
    return {"xyz": xyz, "conf": cf, "masked": masked, "nonfinite": nonfinite, "zero_filled": zero_filled,
            "dims": header_dims(pose)}


def validity_categories(raw, params):
    """Boolean arrays [T, 75] for each reason a landmark can be unusable."""
    return {
        "nonfinite": raw["nonfinite"],
        "masked": raw["masked"],
        "low_confidence": raw["conf"] < params["confidence_threshold"],
        "zero_filled": raw["zero_filled"],
    }


def landmark_validity(raw, params):
    """True where a landmark may be used as a target. Rule is explicit and configurable."""
    cats = validity_categories(raw, params)
    invalid = cats["nonfinite"] | cats["masked"] | cats["low_confidence"]
    if params["zero_fill_is_missing"]:
        invalid = invalid | cats["zero_filled"]
    return ~invalid


def normalize_xyz(xyz, params, dims=None):
    """Raw [..., 75, 3] -> normalised. dims = (width, height) of the sequence's image (default: the default size)."""
    out = np.array(xyz, dtype=np.float32, copy=True)
    cx, cy, hx, hy = xy_geometry(params, dims)
    out[..., 0] = (out[..., 0] - cx) / hx
    out[..., 1] = (out[..., 1] - cy) / hy
    out[..., 2] = out[..., 2] / z_scale_vector(params)
    return out


def denormalize_xyz(xyz, params, dims=None):
    """
    Inverse of normalize_xyz (for visualisation / inference output). Accepts [..., 75, 3].
    Generated poses have no image of their own: pass the canvas you want to draw on (default 300 x 300).
    """
    out = np.array(xyz, dtype=np.float32, copy=True)
    cx, cy, hx, hy = xy_geometry(params, dims)
    out[..., 0] = out[..., 0] * hx + cx
    out[..., 1] = out[..., 1] * hy + cy
    out[..., 2] = out[..., 2] * z_scale_vector(params)
    return out


def normalized_xy(raw, params):
    """Normalised x/y [T, 75, 2] of one extract_raw() result, using that sequence's own image size."""
    cx, cy, hx, hy = xy_geometry(params, raw.get("dims"))
    xy = np.array(raw["xyz"][..., :2], dtype=np.float32, copy=True)
    xy[..., 0] = (xy[..., 0] - cx) / hx
    xy[..., 1] = (xy[..., 1] - cy) / hy
    return xy


def raw_to_record(raw, params):
    """Normalise (per-sequence image size), then fill invalid landmarks with params['missing_fill']; keep the flags."""
    valid = landmark_validity(raw, params)
    xyz = normalize_xyz(raw["xyz"], params, raw.get("dims"))
    xyz[~valid] = params["missing_fill"]
    return {"coords": xyz.reshape(len(xyz), N_COORDS), "valid": valid, "dims": raw.get("dims")}


def add_counter(coords):
    """[T, 225] -> [T, 226] with the frame counter frame_index / n_frames in the last column."""
    n = coords.shape[0]
    counter = np.arange(n, dtype=np.float32) / np.float32(n)
    return np.concatenate([np.asarray(coords, dtype=np.float32), counter[:, None]], axis=1)


def quality_reason(record, quality):
    """None if the record is usable, otherwise a short reason for excluding it."""
    coords, valid = record["coords"], record["valid"]
    n = len(coords)
    if n < quality["min_frames"]:
        return f"too short ({n} frames)"
    if n > quality["max_frames"]:
        return f"too long ({n} frames)"
    if not np.isfinite(coords).all():
        return "non-finite values after normalisation"
    if valid.mean() < quality["min_valid_fraction"]:
        return f"too few valid landmarks ({valid.mean():.0%})"
    if (~valid.any(axis=1)).mean() > quality["max_empty_frame_fraction"]:
        return "too many frames without any valid landmark"
    if quality["require_hands"] and not valid[:, COMPONENTS["left_hand"][0]:].any():
        return "no valid hand landmark in any frame"
    return None


# ----------------------------------------------------------------------------- splitting
def video_id_of(uid):
    """iSign UIDs look like '<video_id>-<sequence_number>'."""
    return str(uid).rsplit("-", 1)[0]


def normalize_text(text):
    """Lower-case and collapse whitespace; the stored .text file has exactly one such line per sequence."""
    return " ".join(str(text).split()).lower()


def split_by_video(uids, seed, fractions=(0.8, 0.1, 0.1), names=("train", "val", "test")):
    """
    Assign whole videos to splits so that no video appears in two of them. Videos are shuffled with
    a fixed seed and added greedily until each split reaches its share of *sequences*.
    Returns (uids_by_split, videos_by_split).
    """
    if abs(sum(fractions) - 1.0) > 1e-9 or len(fractions) != len(names):
        raise ValueError("fractions must sum to 1 and match names")
    groups = OrderedDict()
    for u in uids:
        groups.setdefault(video_id_of(u), []).append(u)
    videos = sorted(groups)
    order = np.random.RandomState(seed).permutation(len(videos))
    cuts = np.cumsum(fractions) * len(uids)
    uids_by, videos_by = OrderedDict((n, []) for n in names), OrderedDict((n, []) for n in names)
    seen = 0
    for k in order:
        v = videos[int(k)]
        s = 0
        while s < len(names) - 1 and seen >= cuts[s]:
            s += 1
        uids_by[names[s]].extend(groups[v])
        videos_by[names[s]].append(v)
        seen += len(groups[v])
    return uids_by, videos_by


def split_overlaps(videos_by):
    """Videos (and, separately, nothing else) that occur in more than one split."""
    seen, overlap = {}, set()
    for name, vids in videos_by.items():
        for v in vids:
            if v in seen and seen[v] != name:
                overlap.add(v)
            seen[v] = name
    return sorted(overlap)


def split_report(uids_by, videos_by, fractions, total_matched):
    rows = []
    for (name, uids), frac in zip(uids_by.items(), fractions):
        rows.append({"split": name, "sequences": len(uids), "videos": len(videos_by[name]),
                     "share_of_sequences": len(uids) / max(total_matched, 1), "target_share": frac})
    return rows


# ----------------------------------------------------------------------------- shard storage
def shard_base(out_dir, k):
    return str(Path(out_dir) / f"shard_{k:04d}")


class ShardWriter:
    """Collects converted sequences and writes one shard (text / files / skels / valid / idx / json)."""

    def __init__(self, params):
        self.params = params
        self.dtype = np.dtype(params["storage_dtype"])
        self.uids, self.texts, self.coords, self.valid, self.excluded, self.dims = [], [], [], [], [], []

    def add(self, uid, text, coords, valid, dims=None):
        text = normalize_text(text)
        if not text:
            raise ValueError("empty text")
        self.uids.append(str(uid))
        self.texts.append(text)
        self.coords.append(np.asarray(coords, dtype=np.float32))
        self.valid.append(np.asarray(valid, dtype=bool))
        self.dims.append([float(dims[0]), float(dims[1])] if dims is not None else None)   # image size used for x/y

    def exclude(self, uid, reason):
        self.excluded.append({"uid": str(uid), "reason": str(reason)})

    def __len__(self):
        return len(self.uids)

    def write(self, out_dir, k, manifest_start, manifest_end, row_uids=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        base = shard_base(out_dir, k)
        lengths = [len(c) for c in self.coords]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        all_c = np.concatenate(self.coords) if self.coords else np.zeros((0, N_COORDS), np.float32)
        all_v = np.concatenate(self.valid) if self.valid else np.zeros((0, N_LANDMARKS), bool)
        stored = all_c.astype(self.dtype)
        # quantisation error: absolute for |value| < 1, relative otherwise
        qerr = float((np.abs(stored.astype(np.float32) - all_c) / np.maximum(np.abs(all_c), 1.0)).max()) if all_c.size else 0.0

        def atomic(ext, writer):
            tmp = f"{base}.{ext}.tmp"
            try:
                writer(tmp)
                os.replace(tmp, f"{base}.{ext}")
            except BaseException:      # e.g. disk full: do not leave half-written files behind
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise

        def write_text(path, text):
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

        def write_idx(path):
            with open(path, "wb") as f:
                np.save(f, offsets)

        atomic("text", lambda p: write_text(p, "".join(t + "\n" for t in self.texts)))
        atomic("files", lambda p: write_text(p, "".join(u + "\n" for u in self.uids)))
        atomic("skels", lambda p: stored.tofile(p))
        atomic("valid", lambda p: all_v.astype(np.uint8).tofile(p))
        atomic("idx.npy", write_idx)
        meta = {
            "shard": int(k), "n_sequences": len(self.uids), "n_frames": int(offsets[-1]),
            "dtype": str(self.dtype), "manifest_start": int(manifest_start), "manifest_end": int(manifest_end),
            "max_quant_error": qerr, "normalization_hash": params_hash(self.params),
            "lengths": [int(x) for x in lengths],
            "valid_fraction": [float(v.mean()) for v in self.valid],
            "dims": self.dims,
            "excluded": self.excluded,
            "row_first_uid": row_uids[0] if row_uids else None,      # identity of the manifest rows this shard covers
            "row_last_uid": row_uids[1] if row_uids else None,
        }
        atomic("json", lambda p: write_text(p, json.dumps(meta)))   # written last: marks the shard complete
        return meta


def convert_manifest(rows, out_dir, read_pose, params, quality, shard_size,
                     max_shards=None, keep_idx=EXPECTED_KEEP_IDX, log=print,
                     min_free_gb=None, disk_path="/"):
    """
    Convert (uid, text) rows into shards. Resumable: shards whose .json exists are skipped, after checking
    that they were made from the same rows in the same order (a different order raises).
    Every excluded sequence is recorded with a reason in its shard's metadata.
    read_pose(uid) must return a pose object (or raise).
    If min_free_gb is given, conversion stops cleanly (stopped='low disk') before a shard when the free
    space on disk_path is below it; rerun later to continue.
    """
    n_shards = math.ceil(len(rows) / shard_size) if rows else 0
    todo = n_shards if max_shards is None else min(int(max_shards), n_shards)
    done = skipped_existing = 0
    stopped = None
    for k in range(todo):
        a, b = k * shard_size, min(len(rows), (k + 1) * shard_size)
        meta_path = Path(shard_base(out_dir, k) + ".json")
        if meta_path.exists():
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("row_first_uid") != rows[a][0] or old.get("row_last_uid") != rows[b - 1][0]:
                raise ValueError(f"{meta_path.name} was made from different rows (or a different row order) than the ones "
                                 "given now; use a new output folder or delete the old shards")
            skipped_existing += 1
            continue
        if min_free_gb is not None:
            free_gb = shutil.disk_usage(disk_path).free / 1e9
            if free_gb < min_free_gb:
                stopped = f"low disk ({free_gb:.1f} GB free on {disk_path}, limit {min_free_gb} GB)"
                log(f"  stopping before shard {k:04d}: {stopped}. Free some space and rerun; finished shards are kept.")
                break
        started = time.time()
        writer = ShardWriter(params)
        for uid, text in rows[a:b]:
            try:
                pose = read_pose(uid)
            except Exception as exc:   # corrupt archive entry, missing file, ...
                writer.exclude(uid, f"unreadable pose ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            if pose is None:
                writer.exclude(uid, "no pose entry")
                continue
            try:
                raw = extract_raw(pose, keep_idx)
                if raw is None:
                    writer.exclude(uid, "pose has no frames")
                    continue
                record = raw_to_record(raw, params)
            except Exception as exc:
                writer.exclude(uid, f"pose structure error ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            reason = quality_reason(record, quality)
            if reason:
                writer.exclude(uid, reason)
                continue
            writer.add(uid, text, record["coords"], record["valid"], dims=record.get("dims"))
        meta = writer.write(out_dir, k, a, b, row_uids=(rows[a][0], rows[b - 1][0]))
        done += 1
        secs = time.time() - started
        log(f"  shard {k:04d}: kept {meta['n_sequences']}/{b - a} sequences, {meta['n_frames']} frames, "
            f"quantisation error {meta['max_quant_error']:.1e}, {secs:.0f}s ({secs / max(b - a, 1):.2f}s per sequence)")
    return {"shards_total": n_shards, "converted_now": done, "already_present": skipped_existing, "stopped": stopped}


class ShardStore:
    """Read access to the shards of one split (memory-mapped, opened lazily, safe to fork)."""

    def __init__(self, shard_dir, expected_hash=None, max_shards=None):
        self.dir = Path(shard_dir)
        self.metas = []
        k = 0
        while max_shards is None or k < max_shards:
            meta_path = Path(shard_base(self.dir, k) + ".json")
            if not meta_path.exists():
                break
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if expected_hash is not None and meta["normalization_hash"] != expected_hash:
                raise ValueError(f"{meta_path.name}: normalization hash {meta['normalization_hash']} "
                                 f"differs from the expected {expected_hash}")
            self.metas.append(meta)
            k += 1
        self.uids, self.texts, self.locs = [], [], []
        lengths, vfrac, dims = [], [], []
        for pos, meta in enumerate(self.metas):
            base = shard_base(self.dir, meta["shard"])
            files = Path(base + ".files").read_text(encoding="utf-8").splitlines()
            texts = Path(base + ".text").read_text(encoding="utf-8").splitlines()
            if not (len(files) == len(texts) == meta["n_sequences"] == len(meta["lengths"])):
                raise ValueError(f"shard {meta['shard']}: files/text/meta counts differ "
                                 f"({len(files)}/{len(texts)}/{meta['n_sequences']})")
            self.uids += files
            self.texts += texts
            self.locs += [(pos, i) for i in range(len(files))]
            lengths += meta["lengths"]
            vfrac += meta["valid_fraction"]
            dims += meta.get("dims") or [None] * len(files)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.valid_fraction = np.asarray(vfrac, dtype=np.float64)
        self.dims = [tuple(d) if d is not None else None for d in dims]       # (width, height) used for each sequence
        self._open = {}

    def __len__(self):
        return len(self.uids)

    def __getstate__(self):
        """Pickle without the open memory maps (they would be copied into every spawned worker)."""
        state = self.__dict__.copy()
        state["_open"] = {}
        return state

    def exclusions(self):
        return [e for m in self.metas for e in m["excluded"]]

    def close(self):
        """Release the memory maps (needed before deleting the files on Windows)."""
        for coords, valid, _ in self._open.values():
            for arr in (coords, valid):
                mm = getattr(arr, "_mmap", None)
                if mm is not None:
                    mm.close()
        self._open.clear()

    def _arrays(self, pos):
        if pos not in self._open:
            meta = self.metas[pos]
            base = shard_base(self.dir, meta["shard"])
            n = meta["n_frames"]
            coords = np.memmap(base + ".skels", dtype=np.dtype(meta["dtype"]), mode="r", shape=(n, N_COORDS))
            valid = np.memmap(base + ".valid", dtype=np.uint8, mode="r", shape=(n, N_LANDMARKS))
            offsets = np.load(base + ".idx.npy")
            self._open[pos] = (coords, valid, offsets)
        return self._open[pos]

    def get(self, i):
        """(uid, text, coords float32 [T, 225], valid bool [T, 75])"""
        pos, j = self.locs[i]
        coords, valid, offsets = self._arrays(pos)
        a, b = int(offsets[j]), int(offsets[j + 1])
        return (self.uids[i], self.texts[i],
                np.asarray(coords[a:b], dtype=np.float32), np.asarray(valid[a:b]).astype(bool))


# ----------------------------------------------------------------------------- diagnostics
def _pcts(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    a = np.asarray(a)
    if a.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(v) for q, v in zip(qs, np.percentile(a, qs))}


def _iqr(a):
    a = np.asarray(a)
    return float(np.subtract(*np.percentile(a, [75, 25]))) if a.size else None


def _pair_distance(raws, valid_list, i, j, params):
    """Distance between two landmarks in NORMALISED x/y units (each sequence with its own image size)."""
    out = []
    for r, v in zip(raws, valid_list):
        m = v[:, i] & v[:, j]
        if m.any():
            xy = normalized_xy(r, params)[m]
            out.append(np.linalg.norm(xy[:, i, :] - xy[:, j, :], axis=1))
    return np.concatenate(out) if out else np.zeros(0)


LANDMARK_GROUPS = OrderedDict([
    ("face 0-10", range(0, 11)), ("shoulders 11-12", range(11, 13)), ("elbows 13-14", range(13, 15)),
    ("wrists 15-16", range(15, 17)), ("pose hand points 17-22", range(17, 23)), ("hips 23-24", range(23, 25)),
    ("knees 25-26", range(25, 27)), ("ankles/feet 27-32", range(27, 33)), ("left hand", range(33, 54)),
    ("right hand", range(54, 75))])


def estimate_z_scales(raws, params):
    """Per-component z scale from a TRAINING-split sample: 99th percentile of |z| over valid landmarks."""
    valid_list = [landmark_validity(r, params) for r in raws]
    scales = {}
    for name, (a, b) in COMPONENTS.items():
        z = np.concatenate([np.abs(r["xyz"][:, a:b, 2][v[:, a:b]]) for r, v in zip(raws, valid_list)])
        scales[name] = float(max(np.percentile(z, 99), 1e-6)) if z.size else 1.0
    return scales


def diagnose_sample(raws, params, jump_threshold=0.5, z_ratio_range=(0.1, 10.0),
                    out_of_bounds_warn=2.0, out_of_bounds_fail=20.0):
    """
    Distribution / validity / coherence report over a list of extract_raw() results.
    Returns (report, flags) where flags is a list of (severity, message); severity is 'WARN' or 'FAIL'.
    """
    zvec = z_scale_vector(params)
    geoms = [xy_geometry(params, r.get("dims")) for r in raws]          # (cx, cy, hx, hy) of each sequence
    nxy = [normalized_xy(r, params) for r in raws]                      # normalised x/y with each sequence's own image size
    valid_list = [landmark_validity(r, params) for r in raws]
    flags = []

    # ---- landmark-level breakdown (categories overlap; 'valid' is what remains)
    thr = params["confidence_threshold"]
    total = sum(v.size for v in valid_list)
    breakdown = {
        "total_landmarks": int(total),
        "valid": int(sum(v.sum() for v in valid_list)),
        "nonfinite": int(sum(r["nonfinite"].sum() for r in raws)),
        "masked": int(sum(r["masked"].sum() for r in raws)),
        "low_confidence": int(sum((r["conf"] < thr).sum() for r in raws)),
        "zero_filled": int(sum(r["zero_filled"].sum() for r in raws)),
        "zero_filled_but_confident": int(sum((r["zero_filled"] & (r["conf"] >= thr)).sum() for r in raws)),
        "confident_and_not_zero_filled": int(sum(((r["conf"] >= thr) & ~r["zero_filled"]).sum() for r in raws)),
    }
    breakdown["valid_fraction"] = breakdown["valid"] / max(total, 1)
    all_conf = np.concatenate([r["conf"].ravel() for r in raws]) if raws else np.zeros(0)
    conf_report = {"min": float(all_conf.min()) if all_conf.size else None,
                   "max": float(all_conf.max()) if all_conf.size else None, **_pcts(all_conf, (5, 25, 50, 75, 95))}

    if breakdown["nonfinite"]:
        flags.append(("WARN", f"{breakdown['nonfinite']} landmarks had NaN/inf values (treated as invalid)"))
    if breakdown["zero_filled_but_confident"]:
        flags.append(("WARN", f"{breakdown['zero_filled_but_confident']} landmarks are exactly (0,0,0) but have confidence "
                              f">= {thr}; they are treated as missing because zero_fill_is_missing={params['zero_fill_is_missing']}"))
    if conf_report["min"] is not None and (conf_report["min"] < 0 or conf_report["max"] > 1.0 + 1e-6):
        flags.append(("FAIL", f"confidence outside [0, 1]: {conf_report['min']:.3f}..{conf_report['max']:.3f}"))
    if breakdown["valid_fraction"] < 0.2:
        flags.append(("FAIL", f"only {breakdown['valid_fraction']:.0%} of landmarks are valid"))

    # ---- per component
    components = {}
    for name, (a, b) in COMPONENTS.items():
        xs, ys, zs, xns, yns, outs = [], [], [], [], [], []
        for r, v, n in zip(raws, valid_list, nxy):
            m = v[:, a:b]
            xyz = r["xyz"][:, a:b, :]
            xs.append(xyz[..., 0][m])
            ys.append(xyz[..., 1][m])
            zs.append(xyz[..., 2][m])
            nx, ny = n[:, a:b, 0][m], n[:, a:b, 1][m]
            xns.append(nx)
            yns.append(ny)
            outs.append((np.abs(nx) > 1) | (np.abs(ny) > 1))             # beyond this sequence's own image
        x, y, z, xn, yn, out_flags = (np.concatenate(t) if t else np.zeros(0) for t in (xs, ys, zs, xns, yns, outs))
        zn = z / float(zvec[a])
        outside = out_flags.mean() * 100 if x.size else None
        ratio = (_iqr(zn) / _iqr(xn)) if x.size and _iqr(xn) else None
        components[name] = {
            "n_valid_landmarks": int(x.size),
            "x_raw": _pcts(x), "y_raw": _pcts(y), "z_raw": _pcts(z),
            "x_norm": _pcts(xn), "y_norm": _pcts(yn), "z_effective": _pcts(zn),
            "pct_xy_outside_nominal_bounds": None if outside is None else float(outside),
            "z_iqr_over_x_iqr": None if ratio is None else float(ratio),
        }
        if outside is not None:
            if outside > out_of_bounds_fail:
                flags.append(("FAIL", f"{name}: {outside:.1f}% of valid x/y lie outside their image; "
                                      "the x/y normalisation assumption is wrong"))
            elif outside > out_of_bounds_warn:
                flags.append(("WARN", f"{name}: {outside:.1f}% of valid x/y lie outside their image "
                                      "(landmarks the pose estimator placed beyond the frame; kept as is, not clamped)"))
        if ratio is not None and ratio > z_ratio_range[1]:
            flags.append(("FAIL" if params["z"]["mode"] == "keep" else "WARN",
                          f"{name}: z spread is {ratio:.3g}x the x spread (limit {z_ratio_range[1]}): z would dominate a "
                          f"squared-error loss; z mode is '{params['z']['mode']}'"))
        elif ratio is not None and ratio < z_ratio_range[0]:
            flags.append(("WARN", f"{name}: z spread is only {ratio:.3g}x the x spread (below {z_ratio_range[0]}): z carries little "
                                  "weight in the loss; that is a modelling choice, not a corrupted scale"))

    # ---- hands vs body wrists: same coordinate system and consistent left/right?
    pairs = {"LL": (BODY_L_WRIST, LH_WRIST), "RR": (BODY_R_WRIST, RH_WRIST),
             "LR": (BODY_L_WRIST, RH_WRIST), "RL": (BODY_R_WRIST, LH_WRIST)}
    wrist = {}
    for key, (i, j) in pairs.items():
        d = _pair_distance(raws, valid_list, i, j, params)
        wrist[key] = {"n": int(d.size), "median_norm": float(np.median(d)) if d.size else None, **_pcts(d, (25, 75, 95))}
    if all(wrist[k]["median_norm"] is not None for k in pairs):
        if not (wrist["LL"]["median_norm"] < wrist["LR"]["median_norm"] and wrist["RR"]["median_norm"] < wrist["RL"]["median_norm"]):
            flags.append(("FAIL", "hand wrists sit closer to the OPPOSITE body wrist than to their own: left/right ordering or "
                                  f"coordinate systems are inconsistent ({ {k: round(v['median_norm'], 3) for k, v in wrist.items()} })"))
        for key in ("LL", "RR"):
            if wrist[key]["median_norm"] > 0.25:
                flags.append(("WARN", f"{key}: hand wrist and body wrist are a median {wrist[key]['median_norm']:.2f} "
                                      "(normalised units) apart; the components may not share a coordinate system"))
    else:
        flags.append(("WARN", "not enough frames with both a valid hand wrist and body wrist to compare coordinate systems"))
    hand_z = {}
    for key, idx in (("left_wrist_z_abs_median", LH_WRIST), ("right_wrist_z_abs_median", RH_WRIST)):
        vals = np.concatenate([np.abs(r["xyz"][:, idx, 2][v[:, idx]]) for r, v in zip(raws, valid_list)]) if raws else np.zeros(0)
        hand_z[key] = float(np.median(vals)) if vals.size else None

    # ---- temporal behaviour
    jumps, seq_with_jump, empty_fracs = [], 0, []
    for r, v, xy in zip(raws, valid_list, nxy):
        both = v[1:] & v[:-1]
        if both.any():
            d = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)[both]
            jumps.append(d)
            seq_with_jump += int((d > jump_threshold).any())
        empty_fracs.append(float((~v.any(axis=1)).mean()))
    jumps = np.concatenate(jumps) if jumps else np.zeros(0)
    temporal = {
        "frame_to_frame_xy_displacement": _pcts(jumps, (50, 90, 99, 99.9, 100)),
        "pct_displacements_over_threshold": float((jumps > jump_threshold).mean() * 100) if jumps.size else None,
        "jump_threshold_norm": jump_threshold,
        "sequences_with_a_jump": int(seq_with_jump),
        "frames_without_any_valid_landmark_pct": float(np.mean(empty_fracs) * 100) if empty_fracs else None,
    }
    if temporal["pct_displacements_over_threshold"] and temporal["pct_displacements_over_threshold"] > 1.0:
        flags.append(("WARN", f"{temporal['pct_displacements_over_threshold']:.2f}% of frame-to-frame displacements exceed "
                              f"{jump_threshold} (normalised units): implausible discontinuities"))

    # ---- image sizes seen, and which landmark groups carry the target variance
    size_counts = Counter(r.get("dims") for r in raws)
    image_sizes = [{"width": None if s is None else s[0], "height": None if s is None else s[1], "sequences": n}
                   for s, n in size_counts.most_common()]
    per_var, per_rate, per_out = np.zeros(N_LANDMARKS), np.zeros(N_LANDMARKS), np.zeros(N_LANDMARKS)
    for j in range(N_LANDMARKS):
        sel = [v[:, j] for v in valid_list]
        xv = np.concatenate([n[:, j, 0][s] for n, s in zip(nxy, sel)]) if raws else np.zeros(0)
        yv = np.concatenate([n[:, j, 1][s] for n, s in zip(nxy, sel)]) if raws else np.zeros(0)
        zv = np.concatenate([r["xyz"][:, j, 2][s] / zvec[j] for r, s in zip(raws, sel)]) if raws else np.zeros(0)
        per_rate[j] = float(np.concatenate(sel).mean()) if raws else 0.0
        if xv.size > 1:
            per_out[j] = 100.0 * float(((np.abs(xv) > 1) | (np.abs(yv) > 1)).mean())
            per_var[j] = float(xv.var() + yv.var() + zv.var()) * per_rate[j]
    total_var = per_var.sum() or 1.0
    landmark_groups = [{"group": g, "landmarks": len(idx), "valid_pct": 100 * float(per_rate[list(idx)].mean()),
                        "outside_image_pct": float(per_out[list(idx)].mean()),
                        "variance_share_pct": 100 * float(per_var[list(idx)].sum() / total_var)}
                       for g, idx in LANDMARK_GROUPS.items()]

    report = {
        "n_sequences": len(raws), "n_frames": int(sum(len(r["xyz"]) for r in raws)),
        "image_sizes": image_sizes, "landmark_groups": landmark_groups,
        "landmark_breakdown": breakdown, "confidence": conf_report,
        "components": components, "wrist_pairing": wrist, "hand_wrist_z": hand_z, "temporal": temporal,
        "normalization_hash": params_hash(params),
        "flags": [{"severity": s, "message": m} for s, m in flags],
    }
    return report, flags


# ----------------------------------------------------------------------------- plots
POSE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15),
              (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
              (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
              (30, 32), (27, 31), (28, 32)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def all_edges():
    """Edges in the kept 75-landmark layout."""
    e = [(a, b) for a, b in POSE_EDGES]
    e += [(33 + a, 33 + b) for a, b in HAND_EDGES]
    e += [(54 + a, 54 + b) for a, b in HAND_EDGES]
    return e


def plot_distributions(raws, params, path):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    xy_p = params["xy"]
    edges_xy = (xy_p["center"] - xy_p["half_range"], xy_p["center"] + xy_p["half_range"]) if xy_p["mode"] == "constant" \
        else (0.0, xy_p["default_size"])          # the common 300 x 300 frame; other sizes are listed in the report
    valid_list = [landmark_validity(r, params) for r in raws]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for row, (name, (a, b)) in enumerate(COMPONENTS.items()):
        for col, axis in enumerate("xyz"):
            vals = np.concatenate([r["xyz"][:, a:b, col][v[:, a:b]] for r, v in zip(raws, valid_list)])
            ax = axes[row, col]
            if vals.size:
                ax.hist(vals, bins=100, color="#3a6ea5")
            if axis in "xy":
                for edge in edges_xy:
                    ax.axvline(edge, color="crimson", linestyle="--", linewidth=1)
            ax.set_title(f"{name} {axis} (raw, valid only)", fontsize=9)
    fig.suptitle("Raw coordinate distributions (red dashes = bounds of the common 300 x 300 frame)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_skeletons(raws, params, path, n_sequences=3, frames_per_sequence=4):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    edges = all_edges()
    chosen = [r for r in raws if len(r["xyz"]) >= frames_per_sequence][:n_sequences]
    fig, axes = plt.subplots(len(chosen), frames_per_sequence, figsize=(3.2 * frames_per_sequence, 3.4 * len(chosen)), squeeze=False)
    colors = {"body": "#222222", "left_hand": "#d1495b", "right_hand": "#2e86ab"}
    for row, r in enumerate(chosen):
        v = landmark_validity(r, params)
        xy = normalized_xy(r, params)
        for col, t in enumerate(np.linspace(0, len(xy) - 1, frames_per_sequence).astype(int)):
            ax = axes[row, col]
            for a, b in edges:
                if v[t, a] and v[t, b]:
                    comp = "body" if a < 33 else ("left_hand" if a < 54 else "right_hand")
                    ax.plot([xy[t, a, 0], xy[t, b, 0]], [xy[t, a, 1], xy[t, b, 1]], color=colors[comp], linewidth=1)
            for name, (a, b) in COMPONENTS.items():
                m = v[t, a:b]
                ax.scatter(xy[t, a:b, 0][m], xy[t, a:b, 1][m], s=6, color=colors[name])
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(1.2, -1.2)
            ax.set_aspect("equal")
            ax.set_title(f"seq {row} frame {t}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Skeleton reconstruction (normalised x/y, valid landmarks only; y axis points down)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_wrist_tracks(raws, params, path, index=0):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    r = raws[index]
    v = landmark_validity(r, params)
    xy = normalized_xy(r, params)
    t = np.arange(len(xy))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for col, (label, body_i, hand_i) in enumerate((("left", BODY_L_WRIST, LH_WRIST), ("right", BODY_R_WRIST, RH_WRIST))):
        for row, axis in enumerate("xy"):
            ax = axes[row, col]
            for idx, name, style in ((body_i, "body wrist", "-"), (hand_i, "hand wrist", "--")):
                series = np.where(v[:, idx], xy[:, idx, row], np.nan)
                ax.plot(t, series, style, label=name)
            ax.set_title(f"{label} wrist, {axis} (normalised)", fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle("Body wrist vs hand wrist over time (should move together)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
