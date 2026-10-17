# Test report

**Decision: NOT READY**, with one blocker: check 11 (CUDA execution and optimizer update) is UNVERIFIED because no GPU was available.
Everything else in the notebook has been **executed** on a CPU against a synthetic archive. Nothing has been verified against the real iSign
poses yet, apart from the diagnostics you ran on Colab (section C).

## A. Notebook executed locally on a CPU (synthetic archive, real code path)

`local_run/make_synthetic_archive.py` builds a stand-in with the real structure: real-format uids (`<12 hex>-<n>`), a ZIP of real pose-format
`.pose` files (576 landmarks, header with the five components) cut into four uneven parts, invalid landmarks of every kind, out-of-bounds coordinates,
bad sequences, unmatched CSV rows and pose files. `local_run/run_notebook_local.py` then runs **the notebook's own cells** in order (only paths, the Drive
mount and a few sizes are adapted). Environment: Python 3.11, torch 2.14 (CPU), numpy 2.4, pandas 3.0, pose-format 0.15.

| # | Check | Local result |
|---|---|---|
| 1 | Dataset archive access, no combined archive | PASS |
| 2 | UID-to-pose matching | PASS (570/575, 5 CSV rows and 3 pose files unmatched and logged) |
| 3 | Text / pose / file count alignment | PASS (20 random sequences per split re-read from the archive and identical) |
| 4 | Coordinate and confidence validity | PASS |
| 5 | Frame width and frame-counter consistency | PASS |
| 5b | Decoder shift and target alignment | PASS |
| 6 | Landmark-level masking | PASS (MSE and L1 unit tests; flags verified on a real batch) |
| 7 | Frame-level padding masking | PASS |
| 8 | Train / val / test separation by video | PASS |
| 9 | Model forward pass (causal, padding-invariant) | PASS (padding difference 2.9e-06) |
| 10 | Finite loss and gradients | PASS |
| 11 | CUDA execution and optimizer update | **UNVERIFIED**: the step ran on the CPU, 92/92 parameter tensors updated |
| 12 | Checkpoint save / reload | PASS (identical predictions) |
| 12b | Training-loop resume and test-set guard | PASS |
| 13 | Small-subset overfit experiment | PASS (train loss 1.006 → 0.122; this is synthetic data and proves nothing about iSign) |
| 14 | Free-running (greedy) evaluation | PASS |
| P | Pilot regression | PASS |

Also verified locally: a run with two DataLoader workers under the `spawn` start method; the Colab cells (`colab_cells/05`, `06`, `07`: sync modules → restore + diagnose → convert)
executed in a **fresh process** (as after a kernel restart) with stale old-normalisation files present, followed by the notebook from CELL 12 to the end; an earlier pass
over finished shards (resume, nothing reconverted, identical numbers). The synthetic archive now mixes image sizes (300 x 300 plus 648x720, 450x396, 696x720, 480x614) like the real one.

### Bugs found by actually running it (all fixed)
1. `EpochShuffleSampler` called `Sampler.__init__(None)`, which fails on current torch (`TypeError`). The call is removed (it does nothing in any torch version).
2. Check 10 compared two training-mode passes, but the repository's feed-forward layers keep a fixed dropout of 0.1 (the config's `dropout: 0.0` does not reach them), so the passes differ. The comparison now runs in eval mode; the backward check stays in training mode.
3. Hardening: the loader now passes a plain-dict vocabulary (the original contains a lambda and cannot be pickled for spawn / forkserver workers) and `ShardStore` no longer pickles open memory maps.

### Static / numpy dry tests (`dry_tests.py`, all pass)
Archive reading without combining, validity rules, normalisation round trip, video split, shard writing / resume / exclusions, diagnostics flags, patch replay on the real repository source,
syntax and name checks, and the consistency of every call from the notebook into the project modules.

## B. What is still unverified
* **CUDA / T4** (check 11, and peak GPU memory at your batch size).
* **Anything depending on real iSign poses.** The synthetic data cannot show real distributions, real failure modes, real speed or memory. The Google Drive connector cannot supply the real data: it returns whole files as text and the pose parts are 25–48 GB each.
* **Real-scale behaviour:** conversion speed from Drive, Drive-cache disk growth, `NUM_WORKERS = 2` under fork on Linux (tested here under spawn only), long sequences.

## C. What your real diagnostics (from Colab, 300 training sequences) already tell us
* 127,237 / 127,237 CSV rows match a pose file; split by video: train 101,791 seq / 4,728 videos, val 12,831 / 661, test 12,615 / 669 (80.0 / 10.1 / 9.9 %).
* 80.3 % of landmarks valid; no non-finite values; no exact-zero landmark with high confidence (the zero-fill rule is redundant on this data).
* Left / right ordering and coordinate-system compatibility confirmed: median wrist distance to the **own** body wrist 0.061 / 0.064, to the **opposite** wrist 0.861 / 0.829 (normalised units).
* **x/y range: explained and fixed.** 8.8–12.2 % of valid x/y fell outside the assumed [0, 300] range because the pose headers do not all say 300 x 300: 56 of 60 sampled files are 300 x 300, the rest are 648x720, 450x396, 696x720 and 480x614. x/y are now normalised per sequence with the header size (identical for 300 x 300). Whether the remaining out-of-image landmarks are only genuine off-frame estimates is re-checked by the new diagnostics.
* **z: decision taken, keep.** z is not pixel-scale (body z median −1.2, hand z about ±0.05); a small hand z only means little weight, so that is now a WARN and only z that dominates the loss is a FAIL.
* **Lower body: not a problem.** Hips are valid in 27 % of frames and carry 0.8 % of the target variance; knees and feet are valid in under 1 % and carry about nothing. The 75-landmark layout stays.

## D. Next steps (updated: the work moves to a Windows PC with a GPU)
The remaining verification happens on the PC: `python pc_run.py check` makes check 11 a real CUDA test and re-runs checks 3-14 on the real data;
see `SETUP_PC.md` and `project_gist.txt`. The Colab-oriented steps below are the earlier plan and still work.
1. On Colab, in the current session: upload the module files to Drive, then run `colab_cells/05_sync_modules.py` → `06_restore_and_diagnose.py` → `07_convert.py` (the new per-sequence normalisation changes the data format hash, so the diagnostics are re-run and the shards are rebuilt).
2. Read the new diagnostics (image sizes, out-of-image rates, landmark-group variance shares, flags). The local regression run with mixed image sizes passes (19 PASS, only check 11 UNVERIFIED).
3. Run the rest of the notebook (CELL 12 onward) on a **T4**; check 11 is then a real test and the decision can become READY.
