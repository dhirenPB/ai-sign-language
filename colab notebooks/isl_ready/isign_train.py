"""
Device handling, batching helpers, training / evaluation loops and checkpointing for the iSign
text-to-pose experiments. Written into the Progressive Transformers repository by the notebook;
import as `isign_train`. Uses the repository's Model / Batch / RegLoss (after the notebook's patches).
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from batch import Batch
from loss import build_element_mask

BATCH_TENSOR_ATTRS = ("src", "src_lengths", "src_mask", "trg_input", "trg", "trg_mask",
                      "frame_mask", "coord_mask", "frame_lengths")


# ----------------------------------------------------------------------------- reproducibility / device
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(require_cuda=False):
    """
    Return (device, info). With require_cuda=True a missing GPU raises a clear error; otherwise the
    CPU is used (fine for preprocessing and small smoke tests, not for training).
    """
    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": props.name, "gpu_memory_gb": round(props.total_memory / 1e9, 1),
                     "gpu_capability": f"{props.major}.{props.minor}"})
    else:
        if require_cuda:
            raise RuntimeError(
                "No CUDA device is available. In Colab choose Runtime > Change runtime type > T4 GPU, "
                "then restart and run again. (CPU is only acceptable for preprocessing and smoke tests.)")
        device = torch.device("cpu")
    info["device"] = str(device)
    return device, info


def prepare_model(model, device):
    """Put the model on `device` and make the repository's own use_cuda flag agree with it."""
    model.to(device)
    model.use_cuda = device.type == "cuda"
    return model


def parameter_devices(model):
    return sorted({str(p.device) for p in model.parameters()})


# ----------------------------------------------------------------------------- batches
def move_batch(batch, device):
    for name in BATCH_TENSOR_ATTRS:
        value = getattr(batch, name, None)
        if torch.is_tensor(value) and value.device != device:
            setattr(batch, name, value.to(device))
    return batch


def make_batch(raw, pad_index, model, device):
    batch = Batch(torch_batch=raw, pad_index=pad_index, model=model)
    return move_batch(batch, device)


def batch_devices(batch):
    return {name: str(getattr(batch, name).device) for name in BATCH_TENSOR_ATTRS
            if torch.is_tensor(getattr(batch, name, None))}


def assert_batch_device(batch, device):
    placed = batch_devices(batch)
    wrong = {k: v for k, v in placed.items()
             if torch.device(v).type != device.type or (device.index is not None and torch.device(v).index != device.index)}
    assert not wrong, f"batch tensors not on {device}: {wrong}"
    return placed


def count_valid_elements(batch):
    """Number of target elements that may contribute to the loss (padded frames and invalid landmarks excluded)."""
    mask = build_element_mask(batch.trg, frame_mask=batch.frame_mask, coord_mask=batch.coord_mask)
    return int(mask.sum().item())


# ----------------------------------------------------------------------------- one optimisation step
def train_step(model, batch, criterion, optimizer, clip_norm=None):
    """Forward, masked loss, backward, clip, step. Skips (and reports) non-finite losses / gradients."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model.get_loss_for_batch(batch, criterion)
    if not bool(torch.isfinite(loss).item()):
        optimizer.zero_grad(set_to_none=True)
        return {"loss": float("nan"), "grad_norm": float("nan"), "skipped": "non-finite loss"}
    loss.backward()
    if clip_norm:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm)))
    else:
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        grad_norm = float(torch.norm(torch.stack(norms))) if norms else 0.0
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return {"loss": float(loss.item()), "grad_norm": grad_norm, "skipped": "non-finite gradient"}
    optimizer.step()
    return {"loss": float(loss.item()), "grad_norm": grad_norm, "skipped": None}


def run_train_epoch(model, loader, criterion, optimizer, pad_index, device, clip_norm=None):
    stats = {"loss_sum": 0.0, "n_valid": 0, "steps": 0, "skipped_empty": 0, "skipped_nonfinite": 0,
             "max_grad_norm": 0.0}
    start = time.time()
    for raw in loader:
        batch = make_batch(raw, pad_index, model, device)
        n_valid = count_valid_elements(batch)
        if n_valid == 0:                      # nothing to learn from: skip explicitly, never divide by zero
            stats["skipped_empty"] += 1
            continue
        result = train_step(model, batch, criterion, optimizer, clip_norm)
        if result["skipped"]:
            stats["skipped_nonfinite"] += 1
            continue
        stats["loss_sum"] += result["loss"] * n_valid
        stats["n_valid"] += n_valid
        stats["steps"] += 1
        stats["max_grad_norm"] = max(stats["max_grad_norm"], result["grad_norm"])
    stats["loss"] = stats["loss_sum"] / stats["n_valid"] if stats["n_valid"] else float("nan")
    stats["seconds"] = time.time() - start
    return stats


@torch.no_grad()
def evaluate(model, loader, criterion, pad_index, device, free_running=False, max_batches=None):
    """
    Mean loss over valid elements (frames and landmarks), gradients disabled.
    free_running=True decodes greedily (ground-truth frame counters drive the timing, as in the
    original repository's validation) and scores the generated frames instead of the teacher-forced ones.
    """
    model.eval()
    total, n_valid_total, n_batches, n_seq, empty = 0.0, 0, 0, 0, 0
    for i, raw in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = make_batch(raw, pad_index, model, device)
        n_seq += batch.nseqs
        n_valid = count_valid_elements(batch)
        if n_valid == 0:
            empty += 1
            continue
        if free_running:
            generated, _ = model.run_batch(batch, max_output_length=None)
            loss = criterion(generated[:, 1:, :], batch.trg, frame_mask=batch.frame_mask, coord_mask=batch.coord_mask)
        else:
            loss, _ = model.get_loss_for_batch(batch, criterion)
        total += float(loss.item()) * n_valid
        n_valid_total += n_valid
        n_batches += 1
    return {"loss": total / n_valid_total if n_valid_total else float("nan"), "n_valid": n_valid_total,
            "batches": n_batches, "sequences": n_seq, "skipped_empty_batches": empty, "free_running": bool(free_running)}


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, optimizer=None, scheduler=None, state=None, normalization=None, cfg=None):
    payload = {"model_state": model.state_dict(), "state": state or {}, "normalization": normalization,
               "config": cfg, "torch_version": torch.__version__}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic: an interrupted save never leaves a half-written checkpoint


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)   # our own files only
    except TypeError:   # very old torch without weights_only
        return torch.load(path, map_location=device)


# ----------------------------------------------------------------------------- full training loop
def train(model, train_loader, val_loader, criterion, cfg, run_dir, device, pad_index,
          normalization=None, max_epochs=None, resume=True, log=print, time_budget_s=None):
    """
    Epoch-based training with validation, best-checkpoint selection on validation loss, LR plateau
    scheduling, early stopping and resume from run_dir/last.ckpt (granularity: completed epochs).
    time_budget_s: stop cleanly after the epoch during which this many seconds have passed (the run stays
    resumable and model selection is NOT marked finished). The test set is never touched here.
    """
    session_start = time.time()
    tr = cfg["training"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = int(max_epochs or tr["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]),
                                 weight_decay=float(tr.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(tr["decrease_factor"]), patience=int(tr["patience"]))
    lr_stop = float(tr.get("learning_rate_min", 0.0))
    early_patience = int(tr.get("early_stop_patience", 10))
    val_every = int(tr.get("val_every_epochs", 1))
    clip_norm = tr.get("clip_grad_norm")

    state = {"epoch": 0, "global_step": 0, "best_val": float("inf"), "best_epoch": 0,
             "bad_validations": 0, "history": [], "stop_reason": None}
    last_path, best_path = run_dir / "last.ckpt", run_dir / "best.ckpt"
    if resume and last_path.exists():
        ckpt = load_checkpoint(last_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        state = ckpt["state"]
        log(f"Resumed from {last_path.name}: {state['epoch']} epochs done, best val {state['best_val']:.5f} "
            f"(epoch {state['best_epoch']}), stop reason so far: {state['stop_reason']}")
        if state.get("stop_reason") in ("early_stopping", "lr_below_min"):
            log("This run had already stopped; nothing to do.")
            return state
        state["stop_reason"] = None

    sampler = getattr(train_loader, "sampler", None)
    if state["epoch"] < max_epochs:             # training continues, so an earlier "selection finished" marker is stale
        (run_dir / "selection_done.json").unlink(missing_ok=True)
    for epoch in range(state["epoch"] + 1, max_epochs + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        stats = run_train_epoch(model, train_loader, criterion, optimizer, pad_index, device, clip_norm)
        entry = {"epoch": epoch, "train_loss": stats["loss"], "lr": optimizer.param_groups[0]["lr"],
                 "steps": stats["steps"], "skipped_empty": stats["skipped_empty"],
                 "skipped_nonfinite": stats["skipped_nonfinite"], "max_grad_norm": stats["max_grad_norm"],
                 "seconds": round(stats["seconds"], 1)}
        state["global_step"] += stats["steps"]
        stop = None
        if epoch % val_every == 0:
            val = evaluate(model, val_loader, criterion, pad_index, device)
            entry["val_loss"] = val["loss"]
            improved = False
            if math.isfinite(val["loss"]):
                scheduler.step(val["loss"])
                improved = val["loss"] < state["best_val"] - 1e-8
            if improved:
                state.update(best_val=val["loss"], best_epoch=epoch, bad_validations=0)
                save_checkpoint(best_path, model, state=dict(state), normalization=normalization, cfg=cfg)
            else:
                state["bad_validations"] += 1
            if state["bad_validations"] >= early_patience:
                stop = "early_stopping"
            if optimizer.param_groups[0]["lr"] < lr_stop:
                stop = "lr_below_min"
        state["epoch"] = epoch
        state["history"].append(entry)
        state["stop_reason"] = stop or ("max_epochs" if epoch >= max_epochs else None)
        save_checkpoint(last_path, model, optimizer, scheduler, state, normalization, cfg)
        log(f"epoch {epoch:3d} | train {entry['train_loss']:.5f} | val {entry.get('val_loss', float('nan')):.5f} | "
            f"lr {entry['lr']:.2e} | steps {entry['steps']} | skipped empty/non-finite "
            f"{entry['skipped_empty']}/{entry['skipped_nonfinite']} | best epoch {state['best_epoch']}")
        if stop:
            break
        if time_budget_s is not None and time.time() - session_start >= time_budget_s and epoch < max_epochs:
            log(f"time budget of {time_budget_s / 3600:.1f} h used up after epoch {epoch}; stopping here. "
                "Run the same command again to resume.")
            state["paused"] = True
            break

    if state["stop_reason"] is None and state["epoch"] >= max_epochs:
        state["stop_reason"] = "max_epochs"
    (run_dir / "history.json").write_text(json.dumps(state["history"], indent=1), encoding="utf-8")
    if state["stop_reason"]:
        (run_dir / "selection_done.json").write_text(json.dumps({
            "best_epoch": state["best_epoch"], "best_val": state["best_val"], "stop_reason": state["stop_reason"],
            "epochs_run": state["epoch"], "normalization": normalization}), encoding="utf-8")
    return state


def final_test_evaluation(model, test_loader, criterion, run_dir, device, pad_index, free_running=True, force=False,
                          free_running_max_batches=None):
    """
    Score the BEST validation checkpoint on the test split. Allowed once, and only after training has finished.
    The teacher-forced score covers the whole split; the (slow) free-running score can be limited to the first
    free_running_max_batches batches.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "selection_done.json").exists():
        raise RuntimeError("Model selection has not finished (no selection_done.json); the test set must not be used yet.")
    out_path = run_dir / "test_results.json"
    if out_path.exists() and not force:
        raise RuntimeError(f"{out_path.name} already exists. Re-evaluating on the test set after looking at it "
                           "is model selection on the test set; pass force=True only if you really intend that.")
    ckpt = load_checkpoint(run_dir / "best.ckpt", device)
    model.load_state_dict(ckpt["model_state"])
    results = {"best_epoch": ckpt["state"].get("best_epoch"),
               "teacher_forced": evaluate(model, test_loader, criterion, pad_index, device, free_running=False)}
    if free_running:
        results["free_running"] = evaluate(model, test_loader, criterion, pad_index, device, free_running=True,
                                           max_batches=free_running_max_batches)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


# ----------------------------------------------------------------------------- overfit helper
def overfit_steps(model, loader, criterion, cfg, pad_index, device, steps, log_every=10, log=print):
    """A fixed number of optimiser steps on one small loader (no validation / test data involved)."""
    tr = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]),
                                 weight_decay=float(tr.get("weight_decay", 0.0)))
    sampler = getattr(loader, "sampler", None)
    epoch, history = 1, []
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:            # next pass over the small subset, reshuffled
            epoch += 1
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            raw = next(iterator)
        batch = make_batch(raw, pad_index, model, device)
        if count_valid_elements(batch) == 0:
            continue
        result = train_step(model, batch, criterion, optimizer, tr.get("clip_grad_norm"))
        history.append(result)
        if log_every and step % log_every == 0:
            log(f"step {step:4d} | loss {result['loss']:.5f} | grad norm {result['grad_norm']:.3f}")
    return history
