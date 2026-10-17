# TRAIN — either a quick throughput measurement (BENCHMARK_STEPS > 0) or the real training run (resumable).
# Needs 10_setup_training.py. Stop with Ctrl+C / interrupt at any time: the last finished epoch is saved and the run resumes.
import copy, time
import numpy as np
import torch

BENCHMARK_STEPS = _opt("BENCHMARK_STEPS", 0)
TIME_BUDGET_HOURS = _opt("TIME_BUDGET_HOURS", None)

if BENCHMARK_STEPS:
    # measured on a throw-away copy of the model: no checkpoint is written and the real model is not changed
    bench_model = copy.deepcopy(model)
    bench_opt = torch.optim.Adam(bench_model.parameters(), lr=LEARNING_RATE)
    it.set_seed(SEED)
    frames, t_start, done, warmup = 0, None, 0, 5
    if IS_CUDA:
        torch.cuda.reset_peak_memory_stats()
    for raw in train_loader:
        batch = it.make_batch(raw, pad_id, bench_model, DEVICE)
        if it.count_valid_elements(batch) == 0:
            continue
        result = it.train_step(bench_model, batch, criterion, bench_opt, cfg["training"].get("clip_grad_norm"))
        if IS_CUDA:
            torch.cuda.synchronize()
        done += 1
        if done == warmup:
            t_start, frames = time.time(), 0
        elif done > warmup:
            frames += int(raw.frame_lengths.sum())
        if done >= warmup + BENCHMARK_STEPS:
            break
    secs = time.time() - t_start
    fps = frames / secs
    mean_len = float(train_set.store.lengths.mean())
    full_frames = len(rows_of["train"]) * mean_len
    print(f"\n{BENCHMARK_STEPS} timed steps: {secs / BENCHMARK_STEPS:.3f} s/step | {fps:,.0f} frames/s | last loss {result['loss']:.4f}")
    if IS_CUDA:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB at batch size {BATCH_SIZE}")
    print(f"projected epoch: {train_frames / fps / 60:,.1f} min on the {len(train_set)} converted training sequences, "
          f"~{full_frames / fps / 3600:,.1f} h on all {len(rows_of['train']):,} training sequences (estimated from their mean length)")
    del bench_model, bench_opt
else:
    final_state = it.train(model, train_loader, val_loader, criterion, cfg, RUN_DIR, DEVICE, pad_id, NORM,
                           max_epochs=MAX_EPOCHS, resume=True,
                           time_budget_s=None if TIME_BUDGET_HOURS is None else float(TIME_BUDGET_HOURS) * 3600)
    print(f"\nstopped: {final_state['stop_reason'] or 'paused (time budget or interruption); run again to resume'} | "
          f"epochs done {final_state['epoch']} | best epoch {final_state['best_epoch']} | best val loss {final_state['best_val']:.5f}")
