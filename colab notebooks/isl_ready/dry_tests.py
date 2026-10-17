"""
Dry tests that need neither torch nor a GPU:
  1. numpy logic (preprocessing, validity rules, normalisation, splitting, shard storage, diagnostics),
  2. replay of repo_patches on the real repository source,
  3. static checks (syntax, undefined names, cross-module imports) of every module and of the notebook.

Usage: python dry_tests.py <folder with the original repository .py files> <scratch folder>
These tests do NOT execute any torch code; the torch / CUDA checks run inside the notebook on Colab.
"""
import ast
import builtins
import difflib
import json
import os
import re
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import isign_pose as ip          # noqa: E402
import isign_report as rep       # noqa: E402

REPO_SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
SCRATCH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.mkdtemp())
failures = []


def LP(p):
    p = str(Path(p).resolve())
    return Path(p if p.startswith("\\\\?\\") or os.name != "nt" else "\\\\?\\" + p)


def section(title):
    print(f"\n=== {title}")


def check(name, fn):
    try:
        fn()
        print(f"  ok   {name}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        failures.append(name)
        print(f"  FAIL {name}: {exc!r}")


# ----------------------------------------------------------------------------- fake poses
def fake_pose(T=10, with_header=True, seed=0, mutate=None, dims=None):
    rng = np.random.default_rng(seed)
    comps = [("POSE_LANDMARKS", 33), ("FACE_LANDMARKS", 468), ("LEFT_HAND_LANDMARKS", 21),
             ("RIGHT_HAND_LANDMARKS", 21), ("POSE_WORLD_LANDMARKS", 33)]
    data = rng.uniform(40, 260, size=(T, 1, 576, 3)).astype(np.float32)
    data[..., 2] = rng.uniform(-30, 30, size=(T, 1, 576))
    conf = rng.uniform(0.7, 1.0, size=(T, 1, 576)).astype(np.float32)
    mask = np.zeros(data.shape, bool)
    if mutate:
        mutate(data, conf, mask)
    header = (types.SimpleNamespace(components=[types.SimpleNamespace(name=n, points=list(range(k))) for n, k in comps])
              if with_header else types.SimpleNamespace())
    if dims is not None:
        header.dimensions = types.SimpleNamespace(width=dims[0], height=dims[1], depth=0)
    body = types.SimpleNamespace(data=np.ma.array(data, mask=mask), confidence=np.ma.array(conf))
    return types.SimpleNamespace(header=header, body=body)


# ----------------------------------------------------------------------------- 1. numpy logic
def t_virtual_zip():
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(1)
    payloads = {f"seq_{i}.pose": rng.bytes(5000 + 777 * i) for i in range(6)}
    zpath = tmp / "all.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in payloads.items():
            z.writestr(n, b)
    blob = zpath.read_bytes()
    zpath.unlink()
    cuts = [0, len(blob) // 3 - 5, 2 * len(blob) // 3 + 11, len(blob)]
    parts = []
    for i in range(3):
        p = tmp / f"part_{i}"
        p.write_bytes(blob[cuts[i]:cuts[i + 1]])
        parts.append(p)
    before = sorted(os.listdir(tmp))
    with ip.VirtualConcatFile(parts) as vf:
        assert vf.total_size == len(blob)
        with zipfile.ZipFile(vf) as z:
            assert sorted(z.namelist()) == sorted(payloads)
            for n, b in payloads.items():
                assert z.read(n) == b, n
    assert sorted(os.listdir(tmp)) == before, "a combined file was created"
    shutil.rmtree(tmp)


def t_params():
    p1, p2 = ip.make_normalization(), ip.make_normalization()
    assert ip.params_hash(p1) == ip.params_hash(p2)
    p3 = ip.make_normalization(z_mode="scale", z_scale={"body": 20.0, "left_hand": 2.0, "right_hand": 2.0})
    assert ip.params_hash(p3) != ip.params_hash(p1)
    assert ip.make_normalization(storage_dtype="float32")["storage_dtype"] == "float32"
    assert np.allclose(ip.z_scale_vector(p1), 1.0)
    v = ip.z_scale_vector(p3)
    assert v.shape == (75,) and v[0] == 20.0 and v[33] == 2.0 and v[74] == 2.0
    try:
        ip.make_normalization(z_mode="scale")
        raise AssertionError("scale mode without scales must raise")
    except ValueError:
        pass


def t_layout():
    idx, note = ip.resolve_landmark_layout(fake_pose())
    assert np.array_equal(idx, ip.EXPECTED_KEEP_IDX) and "verified" in note, note
    idx, note = ip.resolve_landmark_layout(fake_pose(with_header=False))
    assert np.array_equal(idx, ip.EXPECTED_KEEP_IDX) and "fixed" in note, note
    bad = fake_pose()
    bad.header.components = list(reversed(bad.header.components))   # order differs -> must raise, not silently change
    try:
        ip.resolve_landmark_layout(bad)
        raise AssertionError("a different header layout must raise")
    except ValueError:
        pass
    assert len(ip.EXPECTED_KEEP_IDX) == 75 and ip.FRAME_WIDTH == 226


def t_validity_and_record():
    T = 6
    # kept-landmark k -> source index: body k, left hand 501 + (k - 33), right hand 522 + (k - 54)
    src = lambda k: k if k < 33 else (501 + k - 33 if k < 54 else 522 + k - 54)  # noqa: E731
    cases = {}

    def mutate(data, conf, mask):
        # kept 0: low confidence, plausible coordinates
        conf[:, 0, src(0)] = 0.3
        # kept 1: exact (0,0,0) but confident
        data[:, 0, src(1), :] = 0.0
        # kept 2: masked entry
        mask[2, 0, src(2), 0] = True
        # kept 3: NaN coordinate
        data[1, 0, src(3), 1] = np.nan
        # kept 4: valid landmark whose NORMALISED coordinates are exactly zero (x=y=150, z=0.0 raw is not all-zero)
        data[:, 0, src(4), :] = np.array([150.0, 150.0, 0.0], dtype=np.float32)
        # kept 40: outside the nominal image bounds but confident
        data[:, 0, src(40), :] = np.array([400.0, -30.0, 5.0], dtype=np.float32)

    pose = fake_pose(T=T, mutate=mutate)
    raw = ip.extract_raw(pose)
    assert raw["xyz"].shape == (T, 75, 3) and raw["conf"].shape == (T, 75)
    params = ip.make_normalization()
    cats = ip.validity_categories(raw, params)
    assert cats["low_confidence"][:, 0].all() and not cats["low_confidence"][:, 5].any()
    assert cats["zero_filled"][:, 1].all()
    assert cats["masked"][2, 2] and not cats["masked"][3, 2]
    assert cats["nonfinite"][1, 3] and not cats["nonfinite"][0, 3]
    valid = ip.landmark_validity(raw, params)
    assert not valid[:, 0].any() and not valid[:, 1].any()
    assert not valid[2, 2] and valid[3, 2]
    assert not valid[1, 3] and valid[0, 3]
    assert valid[:, 4].all(), "a confident landmark must stay valid even if its normalised value is zero"
    # rule switch: zero-filled but confident becomes valid when the rule is turned off
    relaxed = ip.landmark_validity(raw, ip.make_normalization(zero_fill_is_missing=False))
    assert relaxed[:, 1].all()

    rec = ip.raw_to_record(raw, params)
    coords = rec["coords"].reshape(T, 75, 3)
    assert rec["coords"].shape == (T, 225) and rec["valid"].shape == (T, 75)
    assert np.all(coords[~rec["valid"]] == 0.0), "invalid landmarks must be stored as the fill value"
    assert np.all(coords[:, 4] == 0.0) and rec["valid"][:, 4].all(), "valid zero stays distinguishable via the flag"
    # no clamping: x = 400 -> (400-150)/150
    assert np.allclose(coords[:, 40, 0], (400 - 150) / 150) and coords[0, 40, 0] > 1.0
    assert np.allclose(coords[:, 40, 1], (-30 - 150) / 150)
    # landmark-major layout: coordinate index 3*k + c
    flat = rec["coords"]
    assert np.allclose(flat[:, 3 * 40 + 0], coords[:, 40, 0]) and np.allclose(flat[:, 3 * 40 + 2], coords[:, 40, 2])
    frames = ip.add_counter(rec["coords"])
    assert frames.shape == (T, 226) and np.allclose(frames[:, -1], np.arange(T) / T) and frames[0, -1] == 0 and frames[-1, -1] < 1


def t_normalisation_roundtrip():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-50, 400, size=(5, 75, 3)).astype(np.float32)
    for params in (ip.make_normalization(),
                   ip.make_normalization(z_mode="scale", z_scale={"body": 12.0, "left_hand": 1.5, "right_hand": 1.5})):
        back = ip.denormalize_xyz(ip.normalize_xyz(xyz, params), params)
        assert np.allclose(back, xyz, atol=1e-4)
    p = ip.make_normalization(z_mode="scale", z_scale={"body": 10.0, "left_hand": 2.0, "right_hand": 2.0})
    n = ip.normalize_xyz(xyz, p)
    assert np.allclose(n[:, 0, 2], xyz[:, 0, 2] / 10.0) and np.allclose(n[:, 40, 2], xyz[:, 40, 2] / 2.0)


def t_quality():
    q = dict(ip.DEFAULT_QUALITY)
    ok = {"coords": np.zeros((10, 225), np.float32), "valid": np.ones((10, 75), bool)}
    assert ip.quality_reason(ok, q) is None
    assert "too short" in ip.quality_reason({"coords": ok["coords"][:1], "valid": ok["valid"][:1]}, q)
    long_rec = {"coords": np.zeros((601, 225), np.float32), "valid": np.ones((601, 75), bool)}
    assert "too long" in ip.quality_reason(long_rec, q)
    v = np.zeros((10, 75), bool)
    v[:, :5] = True
    assert "too few valid" in ip.quality_reason({"coords": ok["coords"], "valid": v}, q)
    v = np.ones((10, 75), bool)
    v[:, 33:] = False
    assert "no valid hand" in ip.quality_reason({"coords": ok["coords"], "valid": v}, q)
    v = np.ones((10, 75), bool)
    v[:6] = False
    assert "without any valid" in ip.quality_reason({"coords": ok["coords"], "valid": v}, q)
    bad = {"coords": np.full((10, 225), np.nan, np.float32), "valid": np.ones((10, 75), bool)}
    assert "non-finite" in ip.quality_reason(bad, q)


def t_split():
    rng = np.random.default_rng(5)
    uids = []
    for v in range(400):
        for s in range(int(rng.integers(1, 30))):
            uids.append(f"video{v:04d}-{s:03d}")
    uids_by, vids_by = ip.split_by_video(uids, seed=27)
    assert not ip.split_overlaps(vids_by)
    allu = [u for us in uids_by.values() for u in us]
    assert sorted(allu) == sorted(uids) and len(set(allu)) == len(uids)
    for name, us in uids_by.items():
        assert {ip.video_id_of(u) for u in us} == set(vids_by[name])
        assert len(us) > 0
    rows = ip.split_report(uids_by, vids_by, (0.8, 0.1, 0.1), len(uids))
    for r in rows:
        assert abs(r["share_of_sequences"] - r["target_share"]) < 0.03, rows
    again, _ = ip.split_by_video(uids, seed=27)
    assert again == uids_by, "split must be deterministic for a fixed seed"
    other, _ = ip.split_by_video(uids, seed=28)
    assert other != uids_by
    # uid with extra dashes: only the last '-' separates the sequence number
    assert ip.video_id_of("abc-def-12") == "abc-def"
    # sequence order inside a video is preserved
    for us in uids_by.values():
        seen = {}
        for u in us:
            seen.setdefault(ip.video_id_of(u), []).append(u)
        for v, group in seen.items():
            assert group == sorted(group)


def t_shards():
    tmp = Path(tempfile.mkdtemp())
    params = ip.make_normalization()
    quality = dict(ip.DEFAULT_QUALITY)

    def read_pose(uid):
        if uid == "boom-0":
            raise IOError("corrupt entry")
        if uid == "none-0":
            return None
        if uid == "short-0":
            return fake_pose(T=1, seed=9)
        if uid == "lowvalid-0":
            def m(d, c, mask):
                c[...] = 0.1
            return fake_pose(T=8, seed=10, mutate=m)
        return fake_pose(T=5 + (hash(uid) % 7), seed=abs(hash(uid)) % 1000)

    rows = [(f"vid{i // 3}-{i % 3}", f"  Hello   WORLD number {i} ") for i in range(8)]
    rows += [("boom-0", "x"), ("none-0", "y"), ("short-0", "z"), ("lowvalid-0", "w")]
    out = tmp / "train"
    logs = []
    info = ip.convert_manifest(rows, out, read_pose, params, quality, shard_size=5, max_shards=2, log=logs.append)
    assert info == {"shards_total": 3, "converted_now": 2, "already_present": 0, "stopped": None}, info
    assert any("per sequence" in line for line in logs), "per-shard timing is not logged"
    store = ip.ShardStore(out, expected_hash=ip.params_hash(params))
    assert len(store) == 8 and len(store.metas) == 2     # shard 0: 5 kept; shard 1: 3 kept, 2 excluded
    first_mtime = Path(ip.shard_base(out, 0) + ".json").stat().st_mtime_ns
    # a different row order must be rejected, never silently mixed with the finished shards
    try:
        ip.convert_manifest(list(reversed(rows)), out, read_pose, params, quality, shard_size=5, log=logs.append)
        raise AssertionError("resuming with a different row order must raise")
    except ValueError:
        pass
    # low-disk guard stops cleanly before writing anything
    guard_dir = tmp / "guard"
    info = ip.convert_manifest(rows, guard_dir, read_pose, params, quality, shard_size=5, log=logs.append, min_free_gb=1e9)
    assert info["converted_now"] == 0 and "low disk" in info["stopped"], info
    assert not guard_dir.exists() or not list(guard_dir.glob("*")), "nothing may be written when the guard stops the run"
    info = ip.convert_manifest(rows, out, read_pose, params, quality, shard_size=5, log=logs.append)   # resume: finishes shard 2
    assert info == {"shards_total": 3, "converted_now": 1, "already_present": 2, "stopped": None}, info
    assert Path(ip.shard_base(out, 0) + ".json").stat().st_mtime_ns == first_mtime, "finished shards must not be rewritten"
    store = ip.ShardStore(out, expected_hash=ip.params_hash(params))
    reasons = {e["uid"]: e["reason"] for e in store.exclusions()}
    assert "unreadable" in reasons["boom-0"] and reasons["none-0"] == "no pose entry"
    assert "too short" in reasons["short-0"] and "too few valid" in reasons["lowvalid-0"], reasons
    assert len(store) == 8 and len(reasons) == 4 and len(store.metas) == 3 and store.metas[2]["n_sequences"] == 0   # empty shard is fine
    assert store.uids == [r[0] for r in rows[:8]]
    assert all(t == f"hello world number {i}" for i, t in enumerate(store.texts)), store.texts   # lower-cased, spaces collapsed
    for i, (uid, text) in enumerate(rows[:8]):
        u, t, coords, valid = store.get(i)
        fresh = ip.raw_to_record(ip.extract_raw(read_pose(uid)), params)
        assert u == uid and coords.shape == fresh["coords"].shape and valid.dtype == bool
        assert np.array_equal(valid, fresh["valid"]) and np.allclose(coords, fresh["coords"], rtol=2e-3, atol=2e-3)
        assert len(coords) == store.lengths[i]
    assert (store.valid_fraction > 0).all()
    try:
        ip.ShardStore(out, expected_hash="deadbeef00")
        raise AssertionError("a normalisation hash mismatch must raise")
    except ValueError:
        pass
    assert not list(Path(out).glob("*.tmp")), "temporary files left behind"
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)


def synthetic_raws(n=12, T=60, seed=0, swap_hands=False, z_body_sd=0.08, z_hand_sd=0.03, offset=0.0, dims=None):
    rng = np.random.default_rng(seed)
    raws = []
    for _ in range(n):
        xyz = np.zeros((T, 75, 3), np.float32)
        base = rng.uniform(110, 190, size=(1, 1, 2))
        walk = np.cumsum(rng.normal(0, 1.5, size=(T, 2)), axis=0)
        xyz[..., :2] = base + 0 * walk[:, None, :]
        for k in range(33):                                   # body: fixed pose + small motion
            xyz[:, k, :2] = 150 + 30 * np.sin(k) + offset + 0.3 * walk
            xyz[:, k, 2] = rng.normal(0, z_body_sd, size=T)
        lw = np.stack([100 + walk[:, 0], 160 + walk[:, 1]], axis=1) + offset
        rw = np.stack([200 - walk[:, 0], 160 + walk[:, 1]], axis=1) + offset
        xyz[:, ip.BODY_L_WRIST, :2], xyz[:, ip.BODY_R_WRIST, :2] = lw, rw
        hl, hr = (rw, lw) if swap_hands else (lw, rw)
        for k in range(21):
            xyz[:, 33 + k, :2] = hl + rng.normal(0, 3, size=(T, 2))
            xyz[:, 54 + k, :2] = hr + rng.normal(0, 3, size=(T, 2))
            xyz[:, 33 + k, 2] = rng.normal(0, z_hand_sd, size=T)
            xyz[:, 54 + k, 2] = rng.normal(0, z_hand_sd, size=T)
        conf = rng.uniform(0.6, 1.0, size=(T, 75)).astype(np.float32)
        raw = {"xyz": xyz, "conf": conf, "masked": np.zeros((T, 75), bool),
               "nonfinite": np.zeros((T, 75), bool), "zero_filled": np.zeros((T, 75), bool)}
        if dims is not None:                       # same scene filmed at another image size
            raw["xyz"][..., 0] *= dims[0] / 300.0
            raw["xyz"][..., 1] *= dims[1] / 300.0
            raw["dims"] = (float(dims[0]), float(dims[1]))
        raws.append(raw)
    return raws


def t_image_sizes():
    # header_dims: valid sizes, fall back to 300 x 300 otherwise
    assert ip.header_dims(fake_pose(dims=(648, 720))) == (648.0, 720.0)
    assert ip.header_dims(fake_pose()) == (300.0, 300.0)
    assert ip.header_dims(fake_pose(dims=(0, 0))) == (300.0, 300.0)
    params = ip.make_normalization()
    const = ip.make_normalization(xy_mode="constant")
    assert params["xy"]["mode"] == "image_size" and ip.params_hash(params) != ip.params_hash(const)
    # extract_raw carries the image size; normalisation uses it per sequence
    pose = fake_pose(T=6, dims=(600, 400))
    raw = ip.extract_raw(pose)
    assert raw["dims"] == (600.0, 400.0)
    rec = ip.raw_to_record(raw, params)
    lm = rec["coords"].reshape(6, 75, 3)
    v = rec["valid"]
    expect_x = (raw["xyz"][..., 0] - 300.0) / 300.0
    expect_y = (raw["xyz"][..., 1] - 200.0) / 200.0
    assert np.allclose(lm[..., 0][v], expect_x[v], atol=1e-5) and np.allclose(lm[..., 1][v], expect_y[v], atol=1e-5)
    # the centre of ANY image maps to 0 and its edges to +-1
    probe = {"xyz": np.zeros((1, 75, 3), np.float32), "dims": (648.0, 720.0)}
    probe["xyz"][0, :, 0], probe["xyz"][0, :, 1] = 324.0, 0.0
    n = ip.normalize_xyz(probe["xyz"], params, probe["dims"])
    assert np.allclose(n[0, :, 0], 0.0) and np.allclose(n[0, :, 1], -1.0)
    # a 300 x 300 sequence gives exactly the earlier (v - 150) / 150
    p300 = fake_pose(T=4, dims=(300, 300))
    a = ip.raw_to_record(ip.extract_raw(p300), params)["coords"]
    b = ip.raw_to_record(ip.extract_raw(p300), const)["coords"]
    assert np.allclose(a, b), "300 x 300 must be identical to the constant normalisation"
    # inverse, with the canvas given explicitly
    back = ip.denormalize_xyz(ip.normalize_xyz(raw["xyz"], params, raw["dims"]), params, raw["dims"])
    assert np.allclose(back, raw["xyz"], atol=1e-3)
    # normalized_xy matches normalize_xyz
    assert np.allclose(ip.normalized_xy(raw, params), ip.normalize_xyz(raw["xyz"], params, raw["dims"])[..., :2], atol=1e-6)
    # the per-sequence size survives the shard round trip
    tmp = Path(tempfile.mkdtemp())
    poses = {"a-1": fake_pose(T=8, dims=(648, 720), seed=1), "b-1": fake_pose(T=8, seed=2), "c-1": fake_pose(T=8, dims=(450, 396), seed=3)}
    ip.convert_manifest([(u, "text") for u in poses], tmp / "s", lambda u: poses[u], params, dict(ip.DEFAULT_QUALITY), 8, log=lambda m: None)
    st = ip.ShardStore(tmp / "s", expected_hash=ip.params_hash(params))
    assert st.dims == [(648.0, 720.0), (300.0, 300.0), (450.0, 396.0)], st.dims
    st.close()
    shutil.rmtree(tmp, ignore_errors=True)


def t_diagnostics():
    params = ip.make_normalization()
    report, flags = ip.diagnose_sample(synthetic_raws(), params)
    # mixed image sizes: fine with per-sequence normalisation, flagged under the old constant assumption
    mixed = synthetic_raws(n=6, seed=1) + synthetic_raws(n=2, seed=2, dims=(648, 720)) + synthetic_raws(n=2, seed=3, dims=(450, 396))
    rep_mixed, flags_mixed = ip.diagnose_sample(mixed, params)
    assert not [m for s, m in flags_mixed if s == "FAIL"], flags_mixed
    assert rep_mixed["components"]["body"]["pct_xy_outside_nominal_bounds"] == 0.0
    assert {(s["width"], s["height"]) for s in rep_mixed["image_sizes"]} == {(None, None), (648.0, 720.0), (450.0, 396.0)}
    _, flags_const = ip.diagnose_sample(mixed, ip.make_normalization(xy_mode="constant"))
    assert any("outside their image" in m for s, m in flags_const), "the constant assumption should be flagged on mixed sizes"
    assert [g["group"] for g in rep_mixed["landmark_groups"]][:2] == ["face 0-10", "shoulders 11-12"]
    assert abs(sum(g["variance_share_pct"] for g in rep_mixed["landmark_groups"]) - 100.0) < 1e-6
    # a tiny z spread is a WARN (little weight), never a FAIL
    _, flags_low = ip.diagnose_sample(synthetic_raws(z_hand_sd=0.0005), params)
    assert any(s == "WARN" and "little" in m for s, m in flags_low) and not any(s == "FAIL" for s, _ in flags_low), flags_low
    fails = [m for s, m in flags if s == "FAIL"]
    assert not fails, fails
    assert report["wrist_pairing"]["LL"]["median_norm"] < report["wrist_pairing"]["LR"]["median_norm"]
    assert report["components"]["body"]["pct_xy_outside_nominal_bounds"] == 0.0
    json.dumps(report)   # must be serialisable

    _, flags = ip.diagnose_sample(synthetic_raws(swap_hands=True), params)
    assert any(s == "FAIL" and "OPPOSITE" in m for s, m in flags), flags

    big_z = synthetic_raws(z_body_sd=40.0, z_hand_sd=40.0)      # z in pixel-like units, x/y normalised to +-1
    _, flags = ip.diagnose_sample(big_z, params)
    assert any(s == "FAIL" and "z spread" in m for s, m in flags), flags     # keep mode -> blocker
    scaled = ip.make_normalization(z_mode="scale", z_scale={"body": 400.0, "left_hand": 400.0, "right_hand": 400.0})
    _, flags = ip.diagnose_sample(big_z, scaled)
    assert not any(s == "FAIL" and "z spread" in m for s, m in flags), flags
    est_scale = ip.estimate_z_scales(big_z, params)
    assert all(20 < v < 200 for v in est_scale.values()), est_scale          # p99 of |N(0, 40)| ~ 103

    _, flags = ip.diagnose_sample(synthetic_raws(offset=700.0), params)
    assert any(s == "FAIL" and "outside" in m for s, m in flags), flags       # wrong nominal range -> blocker

    # exact zero-filled confident landmarks are reported separately
    raws = synthetic_raws(n=2)
    raws[0]["zero_filled"][:, 7] = True
    report, flags = ip.diagnose_sample(raws, params)
    assert report["landmark_breakdown"]["zero_filled_but_confident"] == 60
    assert any("exactly (0,0,0)" in m for _, m in flags)


def t_plots():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("       (matplotlib not installed locally: plotting code only syntax-checked)")
        return
    tmp = Path(tempfile.mkdtemp())
    params = ip.make_normalization()
    raws = synthetic_raws(n=4)
    ip.plot_distributions(raws, params, tmp / "d.png")
    ip.plot_skeletons(raws, params, tmp / "s.png")
    ip.plot_wrist_tracks(raws, params, tmp / "w.png")
    assert all((tmp / f).stat().st_size > 1000 for f in ("d.png", "s.png", "w.png"))
    shutil.rmtree(tmp)


def t_registry():
    chk = rep.Checks()
    with chk("a") as c:
        c.note("fine")
    with chk("b"):
        assert 1 == 2, "nope"
    with chk("c") as c:
        c.unverified("no GPU here")
    with chk("d"):
        raise RuntimeError("boom")
    assert [chk.status(n) for n in "abcd"] == [rep.PASS, rep.FAIL, rep.UNVERIFIED, rep.FAIL]
    assert chk.status("never-ran") == rep.UNVERIFIED
    decision, blockers = rep.decide(chk, ["a"])
    assert decision.startswith("READY") and not blockers
    decision, blockers = rep.decide(chk, ["a", "b", "c", "x"])
    assert decision == "NOT READY" and [b["check"] for b in blockers] == ["b", "c", "x"]
    md = rep.render_markdown("T", decision, blockers, chk, {"Section": {"k": 1}})
    assert "NOT READY" in md and "| FAIL | b |" in md


# ----------------------------------------------------------------------------- 2. patch replay
def patch_replay():
    import repo_patches as rp
    work = SCRATCH / "patched"
    if work.exists():
        shutil.rmtree(LP(work))
    LP(work).mkdir(parents=True)
    for f in REPO_SRC.glob("*.py"):
        shutil.copy(LP(f), LP(work) / f.name)
    originals = {f.name: LP(f).read_text(encoding="utf-8") for f in work.glob("*.py")}
    rp.apply_all(LP(work), 5000, log=lambda m: None)
    first = {f.name: LP(f).read_text(encoding="utf-8") for f in work.glob("*.py")}
    msgs = rp.apply_all(LP(work), 5000, log=lambda m: None)
    second = {f.name: LP(f).read_text(encoding="utf-8") for f in work.glob("*.py")}
    assert first == second and all("already patched" in m for m in msgs), "patches are not idempotent"
    backups = {f.name: LP(f).read_text(encoding="utf-8") for f in (work / "_originals").glob("*.py")}
    assert {n: originals[n] for n in backups} == backups and len(backups) == len(rp.PATCHES), "backups differ from the originals"
    for name in ("vocabulary.py", "helpers.py", "model.py", "batch.py", "loss.py", "search.py", "decoders.py", "encoders.py",
                 "embeddings.py", "initialization.py", "transformer_layers.py"):
        assert not re.search(r"^\s*(from|import)\s+torchtext", first[name], flags=re.M), name
    # expected content landed
    assert "def build_element_mask" in first["loss.py"] and "class XentLoss" in first["loss.py"]
    assert "self.coord_mask = lv.unsqueeze(-1)" in first["batch.py"] and "max_len: int = 5000" in first["transformer_layers.py"]
    assert "known = set(self.itos)" in first["vocabulary.py"] and "coord_mask=coord_mask" in first["model.py"]
    assert "torch.ones(" in first["search.py"]
    out_dir = HERE / "patched_repo_files"
    out_dir.mkdir(exist_ok=True)
    diffs = []
    for name, _ in rp.PATCHES:
        (out_dir / name).write_text(first[name], encoding="utf-8")
        diffs += list(difflib.unified_diff(originals[name].splitlines(), first[name].splitlines(),
                                           f"a/{name}", f"b/{name}", lineterm="", n=2))
    (HERE / "repo_patches.diff").write_text("\n".join(diffs) + "\n", encoding="utf-8")
    return first


# ----------------------------------------------------------------------------- 3. static checks
def bound_names(tree):
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
                names.add(arg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                names.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.Global):
            names.update(n.names)
    return names


def undefined_names(tree, extra=()):
    defined = set(dir(builtins)) | set(extra) | {"__name__", "__file__"} | bound_names(tree)
    return sorted({(n.id, n.lineno) for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined})


def toplevel_names(source):
    tree = ast.parse(source)
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for target in n.targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
    return names


def static_modules(patched):
    ours = {p.stem: p.read_text(encoding="utf-8") for p in HERE.glob("*.py")
            if p.stem in ("isign_pose", "isign_report", "isign_data", "isign_train", "repo_patches")}
    for name, src in ours.items():
        tree = ast.parse(src)
        bad = undefined_names(tree)
        assert not bad, f"{name}: undefined names {bad}"
    known = {**{k[:-3]: toplevel_names(v) for k, v in patched.items() if k.endswith(".py")},
             **{k: toplevel_names(v) for k, v in ours.items()}}
    for name in ("isign_data", "isign_train"):
        tree = ast.parse(ours[name])
        aliases = {}
        for n in tree.body:
            if isinstance(n, ast.ImportFrom) and n.module in known:
                for a in n.names:
                    assert a.name in known[n.module], f"{name}: 'from {n.module} import {a.name}' does not exist"
            if isinstance(n, ast.Import):
                for a in n.names:
                    aliases[a.asname or a.name] = a.name
        for n in ast.walk(tree):    # ip.<attr> must exist in isign_pose
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and aliases.get(n.value.id) == "isign_pose":
                assert n.attr in known["isign_pose"], f"{name}: isign_pose has no '{n.attr}'"
    # the helper functions the notebook calls on the patched loss exist with the expected signature
    loss_tree = ast.parse(patched["loss.py"])
    fn = next(n for n in loss_tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_element_mask")
    assert [a.arg for a in fn.args.args] == ["targets", "frame_mask", "coord_mask"]


def static_notebook():
    nb = json.loads((HERE.parent / "isl_text_to_pose_pre_training_ready.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    defined = set(dir(builtins)) | {"display", "get_ipython", "__name__", "__file__"}
    problems = []
    for i, src in enumerate(cells):
        first = src.split("\n", 1)[0]
        if first.startswith("%%writefile"):
            path = first.split()[1]
            module = Path(path).name
            body = src.split("\n", 1)[1]
            assert body.rstrip("\n") == (HERE / module).read_text(encoding="utf-8").rstrip("\n"), f"{module}: notebook copy differs from the file"
            continue
        code = "\n".join("pass  # magic" if l.lstrip().startswith(("%", "!")) else l for l in src.split("\n"))
        tree = ast.parse(code)
        defined |= bound_names(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined:
                problems.append((i, n.id, n.lineno))
    assert not problems, f"names used before any cell defines them: {problems[:10]}"
    print(f"       {len(cells)} code cells parse; no undefined names")

    # every module attribute / imported name the notebook uses must exist, and calls must match the signatures
    patched = {p.stem: p.read_text(encoding="utf-8") for p in (HERE / "patched_repo_files").glob("*.py")}
    known = {k: toplevel_names(v) for k, v in patched.items()}
    for p in HERE.glob("*.py"):
        known[p.stem] = toplevel_names(p.read_text(encoding="utf-8"))
    known.update({"constants": {"UNK_TOKEN", "PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "TARGET_PAD", "DEFAULT_UNK_ID"},
                  "model": toplevel_names((REPO_SRC / "model.py").read_text(encoding="utf-8")) if REPO_SRC else {"build_model"},
                  "batch": {"Batch"}, "vocabulary": {"Vocabulary"}})
    aliases = {"it": "isign_train", "idata": "isign_data", "ip": "isign_pose", "rep": "isign_report", "rp": "repo_patches"}
    signatures = {}
    for module in ("isign_train", "isign_data", "isign_pose", "isign_report", "repo_patches"):
        for n in ast.parse((HERE / f"{module}.py").read_text(encoding="utf-8")).body:
            if isinstance(n, ast.FunctionDef):
                a = n.args
                signatures[(module, n.name)] = ([x.arg for x in a.posonlyargs + a.args], {x.arg for x in a.kwonlyargs},
                                                bool(a.vararg), bool(a.kwarg), len(a.defaults))
    problems = []
    for i, src in enumerate(cells):
        if src.startswith("%%writefile"):
            continue
        tree = ast.parse("\n".join("pass" if l.lstrip().startswith(("%", "!")) else l for l in src.split("\n")))
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                module = aliases[n.value.id]
                if n.attr not in known[module]:
                    problems.append(f"cell {i}: {n.value.id}.{n.attr} does not exist in {module}")
            if isinstance(n, ast.ImportFrom) and n.module in known and n.module not in ("os", "sys"):
                for a in n.names:
                    if a.name not in known[n.module]:
                        problems.append(f"cell {i}: 'from {n.module} import {a.name}' does not exist")
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in aliases and (aliases[n.func.value.id], n.func.attr) in signatures):
                params, kwonly, has_var, has_kw, n_defaults = signatures[(aliases[n.func.value.id], n.func.attr)]
                if any(isinstance(x, ast.Starred) for x in n.args) or any(k.arg is None for k in n.keywords):
                    continue
                for k in n.keywords:
                    if k.arg not in params and k.arg not in kwonly and not has_kw:
                        problems.append(f"cell {i}: {n.func.attr}() has no parameter '{k.arg}'")
                if len(n.args) > len(params) and not has_var:
                    problems.append(f"cell {i}: {n.func.attr}() got {len(n.args)} positional arguments, takes {len(params)}")
                supplied = set(params[:len(n.args)]) | {k.arg for k in n.keywords}
                required = set(params[:len(params) - n_defaults])
                if required - supplied:
                    problems.append(f"cell {i}: {n.func.attr}() missing {sorted(required - supplied)}")
    assert not problems, "API mismatches:\n  " + "\n  ".join(problems)
    print("       notebook calls into the project modules match their definitions (names, keywords, argument counts)")


def main():
    section("numpy logic")
    for name, fn in [("zip split across parts read without combining", t_virtual_zip), ("normalisation params / hash", t_params),
                     ("landmark layout verified, never silently changed", t_layout),
                     ("validity rules, no clamping, valid zeros distinguishable", t_validity_and_record),
                     ("normalise / denormalise round trip", t_normalisation_roundtrip), ("quality filter reasons", t_quality),
                     ("video-level split", t_split), ("shards: write, resume, exclusions, read-back", t_shards),
                     ("per-sequence image size normalisation", t_image_sizes), ("diagnostics flags", t_diagnostics), ("plots", t_plots), ("test registry and decision", t_registry)]:
        check(name, fn)
    patched = {}
    if REPO_SRC:
        section("patch replay on the real repository source")
        try:
            patched = patch_replay()
            print("  ok   patches apply, are idempotent, back up the originals and compile")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures.append("patch replay")
            print(f"  FAIL patch replay: {exc!r}")
        section("static checks")
        if patched:
            check("modules: syntax, names, cross-module imports", lambda: static_modules(patched))
    if (HERE.parent / "isl_text_to_pose_pre_training_ready.ipynb").exists():
        check("notebook cells", static_notebook)
    print("\n" + ("ALL DRY TESTS PASSED" if not failures else f"FAILED: {failures}"))
    sys.exit(1 if failures else 0)


main()
