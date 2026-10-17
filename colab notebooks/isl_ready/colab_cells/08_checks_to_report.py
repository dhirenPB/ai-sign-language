# Notebook CELLS 12-25 as one script. Needs the state created by 06_restore_and_diagnose.py and 07_convert.py.



# ----------------------------------------------------------------------

# CELL 12 — Alignment of text, files and poses; coordinate / confidence validity of the stored data.
def stored_equals_fresh(uid, stored_coords, stored_valid):
    fresh = ip.raw_to_record(ip.extract_raw(read_pose_by_uid(uid), KEEP_IDX), NORM)
    return (fresh["coords"].shape == stored_coords.shape and np.array_equal(fresh["valid"], stored_valid)
            and np.allclose(fresh["coords"], stored_coords, rtol=2e-3, atol=2e-3))


with CHECKS("3. Text/pose/file count alignment") as chk:
    rng = np.random.RandomState(SEED)
    for s, st in STORES.items():
        n = len(st)
        assert n > 0, f"{s}: nothing converted"
        assert len(st.uids) == len(st.texts) == len(st.lengths) == len(st.valid_fraction) == n
        assert len(set(st.uids)) == n, f"{s}: duplicate uids"
        manifest_text = dict(rows_of[s])
        assert set(st.uids) <= set(manifest_text), f"{s}: converted uids that are not in the split manifest"
        bad_text = [u for u, t in zip(st.uids, st.texts) if manifest_text[u] != t]
        assert not bad_text, f"{s}: text differs from the manifest for {bad_text[:3]}"
        assert int(st.lengths.sum()) == sum(m["n_frames"] for m in st.metas), f"{s}: frame counts differ from shard metadata"
        for i in rng.choice(n, min(20, n), replace=False):        # re-read from the archive: uid really maps to this pose
            uid, _, coords, valid = st.get(int(i))
            assert stored_equals_fresh(uid, coords, valid), f"{s}: stored data for {uid} differs from the archive"
    videos_kept = {s: {ip.video_id_of(u) for u in st.uids} for s, st in STORES.items()}
    for a, b in itertools.combinations(SPLITS, 2):
        assert not (videos_kept[a] & videos_kept[b]), f"converted data: {a}/{b} share videos"
    chk.note("; ".join(f"{s}: {len(st)} seq" for s, st in STORES.items()) + "; 20 random sequences per split re-read from the archive and identical")

with CHECKS("4. Coordinate and confidence validity") as chk:
    fails = [m for s_, m in flags if s_ == "FAIL"]
    assert not fails, "diagnostic FAIL flags: " + " | ".join(fails)
    conf = DIAG_REPORT["confidence"]
    assert conf["min"] is not None and conf["min"] >= 0 and conf["max"] <= 1 + 1e-6, f"confidence out of [0,1]: {conf}"
    total_valid, total_lm, outside = 0, 0, 0
    for s, st in STORES.items():
        assert float(max(m["max_quant_error"] for m in st.metas)) < 2e-3, f"{s}: storage precision loss too large"
        for i in range(len(st)):
            _, _, coords, valid = st.get(i)
            assert np.isfinite(coords).all(), f"{s}[{i}]: non-finite stored coordinates"
            lm = coords.reshape(-1, 75, 3)
            assert np.all(lm[~valid] == 0.0), f"{s}[{i}]: invalid landmarks are not stored as the fill value"
            total_valid += int(valid.sum())
            total_lm += valid.size
            xy = lm[..., :2][valid]
            outside += int((np.abs(xy) > 1.0).sum())
    assert total_valid > 0
    chk.note(f"{total_valid / total_lm:.0%} of stored landmarks valid; {outside / max(2 * total_valid, 1):.2%} of valid x/y outside [-1, 1] (kept, not clamped); "
             f"warnings: {[m for s_, m in flags if s_ == 'WARN'] or 'none'}")

# CELL 13 — Import the patched repository modules and select the device.
def reset_imports():
    """Drop cached copies of repository / project modules so re-runs see the current files."""
    stems = {p.stem for d in (CODE_DIR, REPO_DIR) for p in d.glob("*.py") if not p.stem.startswith("__")}
    for s in stems - {"isign_pose", "isign_report", "repo_patches"}:
        sys.modules.pop(s, None)
    for d in (CODE_DIR, REPO_DIR):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    importlib.invalidate_caches()


reset_imports()
import torch
import isign_data as idata
import isign_train as it
from batch import Batch
from constants import EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from loss import RegLoss, build_element_mask
from model import build_model
from vocabulary import Vocabulary

DEVICE, DEVICE_INFO = it.select_device(require_cuda=False)
IS_CUDA = DEVICE.type == "cuda"
it.set_seed(SEED)
print(json.dumps(DEVICE_INFO, indent=1))
if not IS_CUDA:
    print("\nWARNING: no GPU in this runtime. Preprocessing and smoke tests run on the CPU, but the CUDA checks will be "
          "UNVERIFIED and the decision cannot be READY. Switch to a T4 runtime for the final run.")

# CELL 14 — Datasets, vocabulary, configuration, loaders.
import yaml

train_set = idata.ISignShardDataset(PROC_ROOT / "train", NORM, max_frames=QUALITY["max_frames"])
val_set = idata.ISignShardDataset(PROC_ROOT / "val", NORM, max_frames=QUALITY["max_frames"])
test_set = idata.ISignShardDataset(PROC_ROOT / "test", NORM, max_frames=QUALITY["max_frames"])
pilot_ds = idata.ISignShardDataset(PILOT_DIR, NORM)
pilot_set = torch.utils.data.Subset(pilot_ds, list(range(min(PILOT_N, len(pilot_ds)))))

src_vocab = idata.build_vocab(train_set.token_lists())          # training split only
pad_id = src_vocab.stoi[PAD_TOKEN]
src_vocab.to_file(str(WORK_DIR / "src_vocab.txt"))

cfg = yaml.safe_load((REPO_DIR / "Configs" / "Base.yaml").read_text())
cfg["data"].update({"src": "text", "train": str(PROC_ROOT / "train"), "dev": str(PROC_ROOT / "val"), "test": str(PROC_ROOT / "test")})
cfg["data"].pop("src_vocab", None)
cfg["model"]["trg_size"] = ip.N_COORDS                           # 225 coordinates (+1 counter added by build_model)
cfg["training"].update({
    "use_cuda": bool(IS_CUDA), "batch_size": BATCH_SIZE, "epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE,
    "early_stop_patience": EARLY_STOP_PATIENCE, "val_every_epochs": 1, "continue": False, "overwrite": True,
    "model_dir": str(DRIVE_OUT / "runs" / "main"), "random_seed": SEED,
})
cfg["normalization"] = NORM                                      # saved next to the model configuration
cfg_path = WORK_DIR / "config.yaml"
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
(DRIVE_OUT / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
cfg = yaml.safe_load(cfg_path.read_text())

train_loader = idata.make_loader(train_set, src_vocab, BATCH_SIZE, shuffle=True, seed=SEED, num_workers=NUM_WORKERS)
val_loader = idata.make_loader(val_set, src_vocab, BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader = idata.make_loader(test_set, src_vocab, BATCH_SIZE, num_workers=NUM_WORKERS)    # not used before model selection

print(f"train/val/test/pilot sequences: {len(train_set)}/{len(val_set)}/{len(test_set)}/{len(pilot_set)}")
print(f"source vocabulary (incl. 4 specials): {len(src_vocab)} | target width {cfg['model']['trg_size'] + 1} | batch size {BATCH_SIZE}")
print("config + normalisation saved to", cfg_path, "and", DRIVE_OUT / "config.yaml")

# CELL 15 — Build the model for 225-coordinate iSign targets and put it on the device.
it.set_seed(SEED)
model = build_model(cfg, src_vocab=src_vocab, trg_vocab=Vocabulary(tokens=[]))
it.prepare_model(model, DEVICE)
criterion = RegLoss(cfg)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model on {it.parameter_devices(model)} | {n_params / 1e6:.1f} M parameters | loss: {cfg['training']['loss']}")
assert model.in_trg_size == ip.FRAME_WIDTH and model.out_trg_size == ip.FRAME_WIDTH
assert model.decoder.output_layer.out_features == ip.FRAME_WIDTH and model.trg_embed.in_features == ip.FRAME_WIDTH
assert model.src_embed.lut.num_embeddings == len(src_vocab)
print("No pretrained weights are loaded: the repository ships none, so there is nothing that could mismatch the iSign sizes.")

# CELL 16 — Frame width / counter, decoder shift alignment, frame-level padding masks.
by_len = sorted(range(len(train_set)), key=lambda i: int(train_set.store.lengths[i]))
examples = [train_set[by_len[0]], train_set[by_len[len(by_len) // 2]], train_set[by_len[-1]]]      # shortest, median, longest -> real padding
raw = idata.collate_isign(examples, src_vocab)
batch = it.make_batch(raw, pad_id, model, DEVICE)

with CHECKS("5. Frame width and frame-counter consistency") as chk:
    for k in range(min(50, len(train_set))):
        item = train_set[k]
        frames, valid = item["trg"].numpy(), item["landmark_valid"].numpy()
        T = frames.shape[0]
        assert frames.shape == (T, 226) and valid.shape == (T, 75), f"seq {k}: bad shapes {frames.shape}/{valid.shape}"
        assert np.array_equal(frames[:, -1], (np.arange(T, dtype=np.float32) / np.float32(T))), f"seq {k}: counter is not frame_index / n_frames"
        assert frames[0, -1] == 0.0 and frames[-1, -1] < 1.0 and np.all(np.diff(frames[:, -1]) > 0)
    chk.note("226 = 225 coordinates + counter (frame_index / n_frames); checked on up to 50 sequences")

with CHECKS("5b. Decoder shift and target alignment") as chk:
    for b, ex in enumerate(examples):
        T = ex["trg"].shape[0]
        assert torch.equal(batch.trg_input[b, :T - 1].cpu(), ex["trg"][:-1]), f"seq {b}: decoder input is not frames[:-1]"
        assert torch.equal(batch.trg[b, :T - 1].cpu(), ex["trg"][1:]), f"seq {b}: target is not frames[1:]"
        assert torch.allclose(batch.trg[b, :T - 1, -1].cpu(), torch.arange(1, T, dtype=torch.float32) / T), "target counters misaligned"
        assert not batch.trg[b, T - 1:].any(), f"seq {b}: padded target frames are not zero"
    chk.note("input at step t is frame t; target at step t is frame t+1 (counter included); padding is zero")

with CHECKS("7. Frame-level padding masking") as chk:
    B, L = batch.trg.shape[0], batch.trg.shape[1]
    lengths = [ex["trg"].shape[0] for ex in examples]
    assert max(lengths) > min(lengths), "need sequences of different length to test padding"
    assert tuple(batch.trg_mask.shape) == (B, 1, L, L) and batch.trg_mask.dtype == torch.bool
    assert batch.frame_mask.sum(1).cpu().tolist() == [n - 1 for n in lengths], "frame mask != frame_length - 1"
    for b, n in enumerate(lengths):
        assert batch.trg_mask[b, 0, :, : n - 1].all() and not batch.trg_mask[b, 0, :, n - 1:].any(), f"seq {b}: decoder padding mask wrong"
    elem = build_element_mask(batch.trg, batch.frame_mask, batch.coord_mask)
    for b, n in enumerate(lengths):
        assert not elem[b, n - 1:].any(), f"seq {b}: padded frames can reach the loss"
    # padded frames must carry no gradient, whatever the predictions there are
    preds = torch.randn_like(batch.trg, requires_grad=True)
    loss_value = criterion(preds, batch.trg, frame_mask=batch.frame_mask, coord_mask=batch.coord_mask)
    loss_value.backward()
    for b, n in enumerate(lengths):
        assert torch.all(preds.grad[b, n - 1:] == 0), f"seq {b}: gradient on padded frames"
    chk.note(f"lengths {lengths}; decoder mask [B,1,T,T]; padded frames excluded from loss and gradient")
print("Batch tensors by device:", it.assert_batch_device(batch, DEVICE))

# CELL 17 — Landmark-level masking: unit tests of the loss plus alignment on a real batch.
def run_loss_unit_tests(loss_name):
    local_cfg = copy.deepcopy(cfg)
    local_cfg["training"]["loss"] = loss_name
    crit = RegLoss(local_cfg)
    dev = DEVICE
    err = (lambda d: d * d) if loss_name.lower() == "mse" else (lambda d: abs(d))

    # a) valid zero-valued target coordinates DO contribute
    p = torch.tensor([[[1.0, 2.0]]], device=dev, requires_grad=True)
    t = torch.tensor([[[0.0, 2.0]]], device=dev)
    ones = torch.ones(1, 1, 2, dtype=torch.bool, device=dev)
    loss = crit(p, t, coord_mask=ones)
    loss.backward()
    assert abs(loss.item() - (err(1.0) + err(0.0)) / 2) < 1e-6, f"{loss_name}: valid zero target not counted"
    assert p.grad[0, 0, 0].item() != 0.0, f"{loss_name}: valid zero target produced no gradient"

    # b) invalid coordinates do NOT contribute, whatever the prediction there
    for bad_value in (1.0, 1000.0):
        p = torch.tensor([[[bad_value, 2.0]]], device=dev, requires_grad=True)
        mask = torch.tensor([[[False, True]]], device=dev)
        loss = crit(p, t, coord_mask=mask)
        loss.backward()
        assert abs(loss.item()) < 1e-6, f"{loss_name}: invalid coordinate leaked into the loss"
        assert p.grad[0, 0, 0].item() == 0.0, f"{loss_name}: gradient at an invalid coordinate"
        assert torch.isfinite(p.grad).all()

    # c) padded frame excluded; d) frame and coordinate masks combine (AND)
    p = torch.tensor([[[1.0, 1.0], [5.0, 5.0], [7.0, 7.0]]], device=dev, requires_grad=True)
    t3 = torch.zeros(1, 3, 2, device=dev)
    frame = torch.tensor([[True, True, False]], device=dev)
    coord = torch.tensor([[[True, True], [False, True], [True, True]]], device=dev)
    loss = crit(p, t3, frame_mask=frame, coord_mask=coord)
    loss.backward()
    expected = (err(1.0) + err(1.0) + err(5.0)) / 3        # frame0: 2 coords, frame1: only coord 1, frame2: padded
    assert abs(loss.item() - expected) < 1e-5, f"{loss_name}: combined mask value {loss.item()} != {expected}"
    assert torch.all(p.grad[0, 2] == 0) and p.grad[0, 1, 0].item() == 0.0 and p.grad[0, 1, 1].item() != 0.0
    assert torch.isfinite(p.grad).all() and torch.isfinite(loss)

    # e) a mask one column narrower than the targets leaves the counter column valid (and padded frames still excluded)
    p = torch.ones(1, 2, 3, device=dev, requires_grad=True)
    t0 = torch.zeros(1, 2, 3, device=dev)
    cm = torch.tensor([[[False, False], [False, False]]], device=dev)
    loss = crit(p, t0, frame_mask=torch.tensor([[True, False]], device=dev), coord_mask=cm)
    loss.backward()
    assert abs(loss.item() - err(1.0)) < 1e-6 and p.grad[0, 0, 2].item() != 0.0 and torch.all(p.grad[0, 1] == 0)

    # f) nothing valid at all: exactly zero, finite, no gradient, no division by zero
    p = torch.randn(2, 3, 4, device=dev, requires_grad=True)
    loss = crit(p, torch.randn(2, 3, 4, device=dev), frame_mask=torch.zeros(2, 3, dtype=torch.bool, device=dev))
    loss.backward()
    assert loss.item() == 0.0 and torch.all(p.grad == 0) and torch.isfinite(loss)
    all_bad = torch.zeros(2, 3, 4, dtype=torch.bool, device=dev)
    assert int(build_element_mask(torch.zeros(2, 3, 4, device=dev), coord_mask=all_bad).sum()) == 0

    # g) wrong shapes are rejected loudly
    for kwargs in ({"frame_mask": torch.ones(2, 4, dtype=torch.bool, device=dev)},
                   {"coord_mask": torch.ones(2, 3, 9, dtype=torch.bool, device=dev)}):
        try:
            crit(torch.zeros(2, 3, 4, device=dev), torch.zeros(2, 3, 4, device=dev), **kwargs)
            raise AssertionError("a wrongly shaped mask should have been rejected")
        except ValueError:
            pass


with CHECKS("6. Landmark-level masking") as chk:
    for _name in ("MSE", "L1"):
        run_loss_unit_tests(_name)
    # the same flags on a real batch: coordinate triplets line up with landmark flags, shifted by one frame
    lv = raw.landmark_valid
    assert tuple(batch.coord_mask.shape) == (lv.shape[0], lv.shape[1] - 1, 225)
    assert int(batch.coord_mask.sum()) == 3 * int(lv[:, 1:].sum()), "coordinate mask count != 3 x landmark flags"
    checked_invalid = checked_valid = 0
    for b, ex in enumerate(examples):
        T = ex["trg"].shape[0]
        for t_idx in range(0, T - 1, max(1, (T - 1) // 6)):
            for j in range(75):
                flag = bool(lv[b, t_idx + 1, j])
                triplet = batch.coord_mask[b, t_idx, 3 * j: 3 * j + 3].cpu().tolist()
                assert triplet == [flag] * 3, f"seq {b} step {t_idx} landmark {j}: {triplet} vs flag {flag}"
                checked_valid += flag
                checked_invalid += not flag
    assert checked_invalid > 0 and checked_valid > 0, "this batch has no invalid (or no valid) landmarks to check"
    # invalid landmarks are stored as 0.0; valid zero-valued targets (if any) keep their flag
    zero_but_valid = int(((batch.trg[..., :225] == 0) & batch.coord_mask).sum())
    chk.note(f"MSE and L1 unit tests passed; {checked_valid} valid / {checked_invalid} invalid landmark triplets verified on a real batch; "
             f"{zero_but_valid} valid zero-valued target coordinates in this batch")

# CELL 18 — Forward pass, causality, finite loss and gradients.
with CHECKS("9. Model forward pass") as chk:
    model.eval()
    with torch.no_grad():
        pred, _ = model(src=batch.src, trg_input=batch.trg_input, src_mask=batch.src_mask,
                        src_lengths=batch.src_lengths, trg_mask=batch.trg_mask)
        assert pred.shape == batch.trg.shape, f"prediction {tuple(pred.shape)} vs target {tuple(batch.trg.shape)}"
        assert torch.isfinite(pred).all()
        # no look-ahead: changing future decoder inputs must not change earlier predictions
        k = batch.trg_input.shape[1] // 2
        altered = batch.trg_input.clone()
        altered[:, k:, :] += 1.0
        pred2, _ = model(src=batch.src, trg_input=altered, src_mask=batch.src_mask,
                         src_lengths=batch.src_lengths, trg_mask=batch.trg_mask)
        assert torch.allclose(pred[:, :k], pred2[:, :k], atol=1e-5), "earlier predictions changed: the decoder sees the future"
        assert not torch.allclose(pred[:, k:], pred2[:, k:], atol=1e-5), "later predictions ignore their inputs"
    # padding invariance: a sequence alone and padded inside a batch give the same predictions
    worst = 0.0
    for b, ex in enumerate(examples):
        raw_one = idata.collate_isign([ex], src_vocab)
        one = it.make_batch(raw_one, pad_id, model, DEVICE)
        with torch.no_grad():
            alone, _ = model(src=one.src, trg_input=one.trg_input, src_mask=one.src_mask, src_lengths=one.src_lengths, trg_mask=one.trg_mask)
        n = ex["trg"].shape[0] - 1
        worst = max(worst, (pred[b, :n] - alone[0, :n]).abs().max().item())
    assert worst < 1e-3, f"padded batch differs from the single-sequence run by {worst:.2e}"
    chk.note(f"output {tuple(pred.shape)}; causal; padding invariance max diff {worst:.1e}")

with CHECKS("10. Finite loss and gradients") as chk:
    # (a) the model's own loss uses both masks. Compared in eval mode: in training mode the repository's feed-forward
    #     layers keep a fixed dropout of 0.1 (the config's dropout 0.0 does not reach them), so two passes would differ.
    model.eval()
    with torch.no_grad():
        eval_loss, _ = model.get_loss_for_batch(batch, criterion)
        pred_eval = model(src=batch.src, trg_input=batch.trg_input, src_mask=batch.src_mask,
                          src_lengths=batch.src_lengths, trg_mask=batch.trg_mask)[0]
        manual = criterion(pred_eval, batch.trg, frame_mask=batch.frame_mask, coord_mask=batch.coord_mask)
        unmasked = criterion(pred_eval, batch.trg)
    assert abs(eval_loss.item() - manual.item()) < 1e-5 * max(1.0, abs(manual.item())), "get_loss_for_batch is not using both masks"
    assert abs(eval_loss.item() - unmasked.item()) > 1e-6, "masked and legacy losses are identical: the masks have no effect"
    # (b) training-mode forward / backward
    model.train()
    model.zero_grad(set_to_none=True)
    loss_value, _ = model.get_loss_for_batch(batch, criterion)
    assert torch.isfinite(loss_value), "non-finite loss"
    loss_value.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), "non-finite gradient"
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    assert norm > 0, "all gradients are zero"
    model.zero_grad(set_to_none=True)
    model.eval()
    chk.note(f"loss {loss_value.item():.4f}, gradient norm {norm:.3f}" + (f"; parameters without gradient: {missing}" if missing else ""))

# CELL 19 — Device placement and one full optimiser step (CUDA acceptance test).
with CHECKS("11. CUDA execution and optimizer update") as chk:
    print("device:", DEVICE, "| parameters on:", it.parameter_devices(model))
    assert it.parameter_devices(model) == [str(DEVICE)], f"model parameters on {it.parameter_devices(model)}, expected {DEVICE}"
    step_batch = it.make_batch(raw, pad_id, model, DEVICE)
    placed = it.assert_batch_device(step_batch, DEVICE)
    print("batch tensors:", placed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    before = [p.detach().clone() for p in model.parameters()]
    result = it.train_step(model, step_batch, criterion, optimizer, cfg["training"].get("clip_grad_norm"))
    if IS_CUDA:
        torch.cuda.synchronize()
    assert result["skipped"] is None, f"step skipped: {result['skipped']}"
    assert np.isfinite(result["loss"]) and np.isfinite(result["grad_norm"])
    changed = sum(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))
    assert changed > 0.9 * len(before), f"only {changed}/{len(before)} parameter tensors changed"
    assert it.parameter_devices(model) == [str(DEVICE)], "a parameter left the device"
    detail = f"loss {result['loss']:.4f}, grad norm {result['grad_norm']:.3f}, {changed}/{len(before)} parameter tensors updated"
    if IS_CUDA:
        detail += f", peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB at batch size {raw.trg.shape[0]}"
        chk.note(detail)
    else:
        chk.unverified("the step ran, but on the CPU: no CUDA device in this runtime. " + detail)
    del optimizer, before
    gc.collect()

# CELL 20 — The original pilot still passes through the same patched code (separate vocabulary and model).
with CHECKS("P. Pilot regression (isolated 5-sequence set)") as chk:
    pilot_vocab = idata.build_vocab(pilot_ds.token_lists())
    pilot_loader = idata.make_loader(pilot_set, pilot_vocab, PILOT_BATCH_SIZE)
    it.set_seed(SEED)
    pilot_model = it.prepare_model(build_model(cfg, src_vocab=pilot_vocab, trg_vocab=Vocabulary(tokens=[])), DEVICE)
    pilot_opt = torch.optim.Adam(pilot_model.parameters(), lr=LEARNING_RATE)
    stats = it.run_train_epoch(pilot_model, pilot_loader, criterion, pilot_opt, pilot_vocab.stoi[PAD_TOKEN], DEVICE, cfg["training"].get("clip_grad_norm"))
    assert stats["steps"] >= 1 and np.isfinite(stats["loss"]), stats
    chk.note(f"{len(pilot_set)} pilot sequences, one epoch: loss {stats['loss']:.4f}")
    del pilot_model, pilot_opt
    gc.collect()

# CELL 21 — Overfit experiment.
OVERFIT = {}
with CHECKS("13. Small-subset overfit experiment") as chk:
    ready = CHECKS.all_passed(["5. Frame width and frame-counter consistency", "5b. Decoder shift and target alignment", "6. Landmark-level masking",
                               "7. Frame-level padding masking", "9. Model forward pass", "10. Finite loss and gradients"])
    assert ready, "earlier checks failed: diagnose those before the overfit experiment"
    lengths_all, vfrac_all = train_set.store.lengths, train_set.store.valid_fraction
    eligible = [i for i in range(len(train_set)) if lengths_all[i] <= OVERFIT_MAX_FRAMES]
    eligible.sort(key=lambda i: (-vfrac_all[i], i))
    subset_idx = sorted(eligible[:OVERFIT_N])
    assert len(subset_idx) >= 8, f"only {len(subset_idx)} eligible sequences (<= {OVERFIT_MAX_FRAMES} frames); convert more shards or raise OVERFIT_MAX_FRAMES"
    subset = torch.utils.data.Subset(train_set, subset_idx)
    sub_loader = idata.make_loader(subset, src_vocab, OVERFIT_BATCH, shuffle=True, seed=SEED)
    sub_eval = idata.make_loader(subset, src_vocab, OVERFIT_BATCH)

    it.set_seed(SEED)
    of_model = it.prepare_model(build_model(cfg, src_vocab=src_vocab, trg_vocab=Vocabulary(tokens=[])), DEVICE)
    before = it.evaluate(of_model, sub_eval, criterion, pad_id, DEVICE)
    print(f"subset: {len(subset_idx)} sequences, mean valid fraction {vfrac_all[subset_idx].mean():.2f}, "
          f"frames {int(lengths_all[subset_idx].min())}-{int(lengths_all[subset_idx].max())}; loss before training {before['loss']:.4f}")
    history = it.overfit_steps(of_model, sub_loader, criterion, cfg, pad_id, DEVICE, OVERFIT_STEPS, log_every=25)
    after = it.evaluate(of_model, sub_eval, criterion, pad_id, DEVICE)

    done = [h for h in history if not h["skipped"]]
    skipped = len(history) - len(done)
    losses = np.array([h["loss"] for h in done])
    grads = np.array([h["grad_norm"] for h in done])
    k = max(5, len(losses) // 10)
    first, last = float(losses[:k].mean()), float(losses[-k:].mean())
    OVERFIT.update({"subset_size": len(subset_idx), "steps": len(history), "skipped_steps": skipped, "loss_first_window": first,
                    "loss_last_window": last, "eval_loss_before": before["loss"], "eval_loss_after": after["loss"],
                    "max_grad_norm": float(grads.max()), "window": k})
    print(json.dumps(OVERFIT, indent=1))
    assert skipped == 0, f"{skipped} steps were skipped (non-finite loss or gradient)"
    assert np.isfinite(losses).all() and np.isfinite(grads).all()
    assert grads.max() < 1e4, f"exploding gradients (max norm {grads.max():.1f})"
    assert last < first, f"loss did not decrease (first {first:.4f} -> last {last:.4f}): diagnose before any full training"
    chk.note(f"train loss {first:.4f} -> {last:.4f} (mean of first/last {k} steps); eval-mode loss on the subset {before['loss']:.4f} -> {after['loss']:.4f}; "
             f"max grad norm {grads.max():.2f}")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.2))
        plt.plot(losses)
        plt.xlabel("optimizer step")
        plt.ylabel("masked loss")
        plt.title("Overfit experiment (training subset only)")
        plt.tight_layout()
        plt.savefig(DRIVE_OUT / "diagnostics" / "overfit_loss.png", dpi=110)
        plt.show()
    except Exception as exc:
        print("plot skipped:", exc)

# CELL 22 — Checkpoint save / reload equivalence, and resume of the training loop.
with CHECKS("12. Checkpoint save/reload") as chk:
    source_model = of_model if "of_model" in globals() else model
    ckpt_dir = WORK_DIR / "ckpt_test"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "roundtrip.ckpt"
    it.save_checkpoint(ckpt_path, source_model, state={"epoch": 0}, normalization=NORM, cfg=cfg)
    reloaded = it.prepare_model(build_model(cfg, src_vocab=src_vocab, trg_vocab=Vocabulary(tokens=[])), DEVICE)
    payload = it.load_checkpoint(ckpt_path, DEVICE)
    reloaded.load_state_dict(payload["model_state"])
    assert payload["normalization"] == NORM, "normalisation stored in the checkpoint differs"
    assert payload["config"]["model"]["trg_size"] == 225
    source_model.eval()
    reloaded.eval()
    probe = it.make_batch(raw, pad_id, source_model, DEVICE)
    with torch.no_grad():
        a, _ = source_model(src=probe.src, trg_input=probe.trg_input, src_mask=probe.src_mask, src_lengths=probe.src_lengths, trg_mask=probe.trg_mask)
        b, _ = reloaded(src=probe.src, trg_input=probe.trg_input, src_mask=probe.src_mask, src_lengths=probe.src_lengths, trg_mask=probe.trg_mask)
    assert torch.allclose(a, b, atol=1e-6), f"reloaded model differs by {(a - b).abs().max().item():.2e}"
    chk.note(f"max prediction difference {(a - b).abs().max().item():.1e} ({'trained' if 'of_model' in globals() else 'untrained'} weights)")

with CHECKS("12b. Training-loop resume and test-set guard") as chk:
    assert "subset" in globals() and CHECKS.status("13. Small-subset overfit experiment") == rep.PASS, "needs the overfit subset"
    mcfg = copy.deepcopy(cfg)
    mcfg["training"].update({"early_stop_patience": 50, "patience": 50, "learning_rate_min": 0.0})
    mdir = WORK_DIR / "machinery_test"
    if mdir.exists():
        shutil.rmtree(mdir)
    # machinery only: the training subset doubles as 'validation' here, so these numbers mean nothing
    it.set_seed(SEED)
    m1 = it.prepare_model(build_model(cfg, src_vocab=src_vocab, trg_vocab=Vocabulary(tokens=[])), DEVICE)
    s1 = it.train(m1, sub_loader, sub_eval, criterion, mcfg, mdir, DEVICE, pad_id, NORM, max_epochs=2, resume=True, log=lambda m: None)
    assert s1["epoch"] == 2 and (mdir / "last.ckpt").exists() and (mdir / "best.ckpt").exists() and (mdir / "selection_done.json").exists()
    # simulate a Colab interruption: new process-like state, resume to 4 epochs
    it.set_seed(SEED + 1)
    m2 = it.prepare_model(build_model(cfg, src_vocab=src_vocab, trg_vocab=Vocabulary(tokens=[])), DEVICE)
    s2 = it.train(m2, sub_loader, sub_eval, criterion, mcfg, mdir, DEVICE, pad_id, NORM, max_epochs=4, resume=True, log=lambda m: None)
    assert s2["epoch"] == 4 and len(s2["history"]) == 4 and s2["history"][:2] == s1["history"][:2], "resume did not continue from the saved state"
    assert s2["best_epoch"] >= 1
    # the test set must be unreachable before model selection finished
    try:
        it.final_test_evaluation(m2, sub_eval, criterion, WORK_DIR / "never_trained", DEVICE, pad_id)
        raise AssertionError("test evaluation was allowed without finished model selection")
    except RuntimeError:
        pass
    chk.note("2 epochs, simulated interruption, resumed to 4 (history preserved); test evaluation refuses to run before selection is done")

with CHECKS("14. Free-running (greedy) evaluation") as chk:
    assert "of_model" in globals() and CHECKS.status("13. Small-subset overfit experiment") == rep.PASS, "needs the overfit model"
    teacher_forced = it.evaluate(of_model, sub_eval, criterion, pad_id, DEVICE, free_running=False, max_batches=1)
    free_running = it.evaluate(of_model, sub_eval, criterion, pad_id, DEVICE, free_running=True, max_batches=1)
    assert free_running["batches"] == 1 and np.isfinite(free_running["loss"]), free_running
    assert np.isfinite(teacher_forced["loss"])
    chk.note(f"one batch of the training subset: teacher-forced loss {teacher_forced['loss']:.4f}, free-running (ground-truth counters) "
             f"loss {free_running['loss']:.4f}; generated frames have the target shape")

# CELL 23 — Pass/fail report, final representation, split statistics, decision.
REQUIRED = [
    "1. Dataset archive access (no combined archive)", "2. UID-to-pose matching", "3. Text/pose/file count alignment",
    "4. Coordinate and confidence validity", "5. Frame width and frame-counter consistency", "6. Landmark-level masking",
    "7. Frame-level padding masking", "8. Train/val/test separation by video ID", "9. Model forward pass",
    "10. Finite loss and gradients", "11. CUDA execution and optimizer update", "12. Checkpoint save/reload",
    "13. Small-subset overfit experiment", "5b. Decoder shift and target alignment", "12b. Training-loop resume and test-set guard",
    "14. Free-running (greedy) evaluation", "P. Pilot regression (isolated 5-sequence set)",
]
CHECKS.print_table(REQUIRED)
DECISION, BLOCKERS = rep.decide(CHECKS, REQUIRED)

split_stats = {}
for s in SPLITS:
    m, st = manifests[s], STORES[s]
    split_stats[s] = {"planned_sequences": len(m), "planned_videos": int(m["video_id"].nunique()),
                      "planned_share": round(len(m) / len(matched), 4),
                      "converted_sequences": len(st), "converted_videos": len({ip.video_id_of(u) for u in st.uids}),
                      "excluded_sequences": len(st.exclusions())}
representation = {
    "landmarks": "33 body + 21 left hand + 21 right hand (pose-format indices 0:33 and 501:543); layout verified from the file header: " + layout_note,
    "frame": "225 coordinates (landmark-major x,y,z) + 1 frame counter = 226",
    "frame_counter": "frame_index / n_frames (first frame 0.0, last (n-1)/n); regenerated at load time",
    "decoder_alignment": "input step t = frame t, target step t = frame t+1 (counter included); input and target are shifted by one frame",
    "invalid_landmarks": "stored as 0.0 and flagged in a separate [T,75] validity array; the loss masks them (and padded frames)",
    "validity_rule": f"valid = finite and not masked and confidence >= {CONF_THRESHOLD}" + (" and not an exact raw (0,0,0)" if ZERO_FILL_IS_MISSING else ""),
    "normalization": NORM, "normalization_hash": NORM_HASH,
    "no_imputation": "missing landmarks are not interpolated; decoder inputs contain 0.0 at those positions",
}
sections = {
    "Final coordinate representation": representation,
    "Split statistics (by video, seed %d)" % SEED: split_stats,
    "Conversion": [{k: v for k, v in r.items()} for r in conv_rows],
    "Diagnostics flags": [{"severity": s_, "message": m_} for s_, m_ in flags],
    "Overfit outcome": OVERFIT,
    "Runtime": DEVICE_INFO,
    "Not verified / limitations": (
        "Unconstrained inference (text only, no ground-truth counters) is not part of this notebook; greedy decoding uses the "
        "ground-truth frame counters as in the original repository. Resume granularity is one epoch. The text-format .skels of the "
        "pilot was replaced by binary shards. Test-set performance has not been measured."),
}
report_md = rep.render_markdown("iSign text-to-pose: pre-training readiness report", DECISION, BLOCKERS, CHECKS, sections)
(WORK_DIR / "pre_training_report.md").write_text(report_md, encoding="utf-8")
(DRIVE_OUT / "pre_training_report.md").write_text(report_md, encoding="utf-8")
(DRIVE_OUT / "pre_training_report.json").write_text(json.dumps({"decision": DECISION, "blockers": BLOCKERS, "checks": CHECKS.results,
                                                                "sections": sections}, indent=1, default=str), encoding="utf-8")
print("\n" + "=" * 70)
print("DECISION:", DECISION)
for b in BLOCKERS:
    print(f"  - {b['check']}: {b['status']}  {b['detail']}")
print("=" * 70)
print("report saved to", DRIVE_OUT / "pre_training_report.md")

# CELL 24 — Full training (guarded).
if not RUN_FULL_TRAINING:
    print("RUN_FULL_TRAINING is False — not starting. Review the report above first.")
elif DECISION != "READY FOR CONTROLLED TRAINING":
    print("Decision is", DECISION, "— not starting. Resolve the blockers first.")
else:
    if REQUIRE_GPU_FOR_TRAINING and not IS_CUDA:
        raise RuntimeError("Training requires a GPU (Runtime > Change runtime type > T4 GPU).")
    it.set_seed(SEED)
    train_model = it.prepare_model(build_model(cfg, src_vocab=src_vocab, trg_vocab=Vocabulary(tokens=[])), DEVICE)
    final_state = it.train(train_model, train_loader, val_loader, criterion, cfg, DRIVE_OUT / "runs" / "main",
                           DEVICE, pad_id, NORM, max_epochs=MAX_EPOCHS, resume=True)
    print("stopped:", final_state["stop_reason"], "| best epoch", final_state["best_epoch"], "| best val", final_state["best_val"])

# CELL 25 — Test-set evaluation (guarded; only after training finished; only once).
if not RUN_TEST_EVAL:
    print("RUN_TEST_EVAL is False — the test split has not been touched.")
else:
    results = it.final_test_evaluation(train_model if "train_model" in globals() else model, test_loader, criterion,
                                       DRIVE_OUT / "runs" / "main", DEVICE, pad_id, free_running=True)
    print(json.dumps(results, indent=1))
