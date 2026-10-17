# Pre-training readiness: change log

Notebook: `../isl_text_to_pose_pre_training_ready.ipynb` (the earlier `isl_text_to_pose_pre_training.ipynb` and the original
`isl_text_to_pose_clean_pipeline.ipynb` are untouched and serve as backups).

The architecture is unchanged. The Progressive Transformers repository is cloned fresh on every run, patched in place by
`repo_patches.py` (every edit is asserted, the originals are copied to `<repo>/_originals/`), and the new code lives in
four project modules that the notebook writes to `/content/isl_code`.

## Files in this folder

| File | Purpose |
|---|---|
| `isign_pose.py` | numpy only: archive access (`VirtualConcatFile`), landmark extraction, validity rules, normalisation, video-level split, shard writer / reader, diagnostics and plots |
| `isign_data.py` | PyTorch dataset over the shards, vocabulary, collate (adds `frame_lengths` and `landmark_valid`), loader with a resumable shuffle |
| `isign_train.py` | device selection, batch placement checks, train step / epoch, evaluation (teacher-forced and free-running), checkpoints, `train()` with early stopping and resume, guarded `final_test_evaluation()` |
| `isign_report.py` | PASS / FAIL / UNVERIFIED registry and the READY / NOT READY decision |
| `repo_patches.py` | the patches to the cloned repository |
| `patched_repo_files/`, `repo_patches.diff` | the repository files as they look after patching, and the diff against the originals (for review) |
| `notebook_cells.txt`, `build_notebook.py` | source of the notebook and the generator that embeds the modules |
| `dry_tests.py` | tests that need neither torch nor a GPU (see `TEST_REPORT.md`) |

## Fix 1: landmark-level missing data

* **Before:** `xyz[scores < 0.5] = 0.0`; the loss masked padded frames (or every zero), so artificial zeros were trained on.
* **Now:** `extract_raw()` keeps confidence and four separate categories per landmark: non-finite, masked (pose-format mask), low confidence
  (`< CONF_THRESHOLD`), and zero-filled (exact raw `(0,0,0)`, the file format's fill value). `landmark_validity()` applies the explicit rule
  `finite and not masked and confidence >= threshold and not zero-filled` (the zero-fill term is switchable). The diagnostics count every
  category separately, including "zero-filled but confident", so the rule is checked against the data rather than assumed.
* Invalid landmarks are stored as `0.0` **plus** a `[T, 75]` validity array. A valid landmark whose normalised value happens to be exactly 0 stays valid.
* `Batch` now carries `frame_mask [B, T]` (padded frames), `coord_mask [B, T, 225]` (`landmark_valid` shifted by one frame like the target, each flag repeated for x, y, z),
  and the decoder padding mask `[B, 1, T, T]` built from true lengths.
* `RegLoss(preds, targets, frame_mask=None, coord_mask=None)`: `build_element_mask()` ANDs both masks (a 225-wide mask leaves the counter column valid),
  the loss is the mean over valid elements (`masked_fill` + denominator clamped to 1), an all-invalid batch gives exactly 0 with zero gradient, and wrong shapes raise.
  Without masks the original behaviour is kept for regression. `Model.get_loss_for_batch` passes both masks. `train_step` / `run_train_epoch` count valid elements first and skip
  empty batches explicitly (counted and logged).
* **No imputation / interpolation:** invalid coordinates are not filled in; decoder *inputs* contain `0.0` at those positions. The validity array identifies them everywhere.

## Fix 2: real splits

* `video_id` = UID text before the last `-`. `split_by_video()` shuffles whole videos with a fixed seed (27) and fills train, then val, then test up to 80 / 10 / 10 % of
  **sequences**; videos never straddle splits. The split is made on the matched sequences before quality filtering, so later exclusions cannot move a video.
* Unmatched CSV uids, pose files without a CSV row, empty texts and duplicate uids are written to `logs/` on Drive; every sequence excluded at conversion time is logged with its reason
  (`logs/excluded_sequences.csv`).
* Separate `.text`, `.files` and `.skels` per split. **Deviation:** `.skels` is binary (plus `.valid`, `.idx.npy`, `.json` per shard). As text, the full dataset would need tens of
  gigabytes. Frame counters are not stored: they are regenerated as `frame_index / n_frames`. Conversion is sharded and resumable.
* The 5-example pilot is built in its own local folder and only used for a regression check; it never touches the splits and is never evaluated.
* Review-scale conversion by default (4 + 1 + 1 shards of 256); `FULL_CONVERT = True` converts everything (hours, resumable, still reads the archive in place).

## Fix 3: coordinate normalisation

* Frame convention unchanged (75 landmarks, `0:33` + `501:543`, 225 coordinates landmark-major + counter = 226). The header is only used to *verify* the layout; a mismatch raises instead of silently choosing other landmarks.
* x, y: **normalised per sequence with the image size from each pose header** (see "Update after the real diagnostics" below); not clamped (out-of-range values are kept and counted). For a 300 x 300 file this is exactly the earlier `(v - 150) / 150`.
* z: **unchanged by default** (`Z_MODE = "keep"`), but now investigated. The diagnostics compare, per component, the z spread to the x spread, print wrist-z statistics (hand z is expected to be
  wrist-relative), and flag the result. If keeping z is not appropriate the check fails with a message; `Z_MODE = "scale"` then divides z by per-component scales estimated from the **training sample only**.
* Diagnostics over a random training-split sample: percentiles (0, 1, 5, 25, 50, 75, 95, 99, 100) per component and axis, % of x/y outside the nominal `[0, 300]` image bounds, NaN / inf counts,
  confidence distribution, frame-to-frame displacement percentiles and jump counts, frames without any valid landmark, and whether hand wrists sit closer to their own body wrist than to the opposite one
  (left / right and coordinate-system compatibility). Plots: distributions, skeleton reconstruction, wrist tracks.
* `normalization.json` (and its hash) is saved next to the config, stored in every checkpoint, used to name the shard folder (a different normalisation can never mix into existing shards)
  and verified by the dataset on load. `denormalize_xyz()` is the inverse for inference output.

## Fix 4: device

* `select_device()`, `prepare_model()` (keeps the repository's `use_cuda` flag consistent), `make_batch()` / `move_batch()` (moves every batch tensor, including source lengths, masks, frame lengths and validity masks) and
  `assert_batch_device()` (checks each one). PyTorch version, CUDA build, GPU name, memory and capability are logged. No GPU: preprocessing and smoke tests run on the CPU, the CUDA check is **UNVERIFIED**, training refuses to start.
* Check 11 does a complete forward / loss / backward / clip / optimizer step, requires finite loss and gradients and that more than 90 % of parameter tensors changed, and is only PASS on a real CUDA device.
* Batch size is configurable (default 8); peak GPU memory is reported.

## Fix 5: overfit experiment

Fresh model, fixed subset of up to 32 **training** sequences (highest valid-landmark fraction, at most 250 frames), 300 steps, validation / test data untouched. Reported as measured: mean loss of the first vs last 10 % of steps,
eval-mode loss before / after, max gradient norm. PASS needs finite values, no skipped steps, gradient norm < 1e4 and a lower final than initial window. There is no fixed reduction threshold. It also plots the loss curve.

## Fix 6: training and evaluation foundation

`isign_train.train()`: Adam, gradient clipping, validation with gradients off, plateau scheduler (the repository's settings), best checkpoint on validation loss, early stopping, stop when the LR falls below `learning_rate_min`,
atomic `last.ckpt` after every epoch, resume (granularity: completed epochs), per-epoch seeds so a resumed run shuffles like an uninterrupted one, `history.json`, `selection_done.json`.
`final_test_evaluation()` refuses to run before `selection_done.json` exists, evaluates the best checkpoint, and refuses a second run unless forced. Both training and test cells are off by default and gated on the decision.

Inspected before writing the loop: input step *t* is frame *t*, target step *t* is frame *t+1* (counter included), padding is zero; the decoder is causal and padding-invariant (both asserted in checks 5b and 9).
In the original repository free-running decoding is driven by the ground-truth counters; unconstrained text-only inference is not implemented here.

## Other repository patches

* `vocabulary.py` / `helpers.py`: `torchtext` imports removed (the earlier notebook only removed the annotation, so `vocabulary.py` would still have failed to import); TensorBoard import made optional; `add_tokens` uses a set (the list scan is quadratic for a large vocabulary).
* `transformer_layers.py`: positional-encoding length `200000 → 5000` (≈0.8 GB for the two tables).
* `search.py`: greedy decoding built its padding mask from coordinate *values*, which blanked out zero-filled landmarks; now all generated frames are visible.

## Update after the first Colab run (disk full during conversion)

Observed: CELL 11 took ~29 min for the pilot plus two shards (~3.4 s per sequence) and then failed with `OSError: [Errno 28] No space left on device` (Colab: 90.6 / 112.6 GB used).
Likely cause (not yet confirmed by a disk report): conversion followed the shuffled split order, i.e. random access across the ~150 GB archive on Drive, and Drive caches what it reads on the VM disk.

* Conversion now follows the **archive order** within each split (near-sequential reads). Split membership is unchanged. The review-scale subset is therefore the first videos of each split in archive order, not a random sample.
* The coordinate-diagnostics sample is still random but is read in archive order.
* `convert_manifest()` checks free local disk before every shard and stops cleanly below `MIN_FREE_GB` (default 8); it logs seconds per sequence; it refuses to resume if the existing shards were made from different rows or a different order (shards now record their first / last row); a failed write removes its temp file.
* New CELL 10b: `disk_report()` and an opt-in, unverified `clear_drivefs_cache(confirm=True)`.
* Shards go to `norm_<hash>_archive_order`; the earlier `norm_<hash>` folder is obsolete and can be deleted.
* `FULL_CONVERT` still reads the archive once per split (three passes); a single-pass converter is a possible later improvement.

## Update after running the notebook locally on a CPU

The notebook (and the Colab cells in `colab_cells/`) were executed end to end on a CPU against a synthetic archive (`local_run/`); see `TEST_REPORT.md`. Fixes that came out of it:

* `EpochShuffleSampler` no longer calls `Sampler.__init__` (a `TypeError` on current torch).
* Check 10 compares the model's loss with the manual masked loss in eval mode (the repository's feed-forward layers keep a fixed dropout of 0.1 in training mode) and also asserts that the masked and unmasked losses differ.
* `make_loader` passes a plain-dict vocabulary to the collate function, and `ShardStore` drops its open memory maps when pickled, so `NUM_WORKERS > 0` also works with spawn / forkserver start methods (verified with two workers under spawn).
* `colab_cells/04_inspect_all.py` adds per-landmark-group statistics (valid rate, out-of-bounds rate, share of target variance).

## Update after the real diagnostics (inspect cell on Colab, 60 training sequences)

Finding: the pose headers do **not** all say 300 x 300. 56 of 60 sampled files are 300 x 300 (depth 0), the others are 648 x 720, 450 x 396, 696 x 720 and 480 x 614, with coordinates in those frames' pixels. The fixed `(v - 150) / 150` was therefore wrong for about 7 % of the sequences (x up to ~740, y up to ~755 in raw units), which is what the "outside [0, 300]" counts were measuring.
The lower-body landmarks are not a problem: hips are valid in 27 % of frames and carry 0.8 % of the target variance, knees and feet are valid in under 1 % and carry ~0 %. The 75-landmark layout is therefore unchanged.

* **x/y are normalised per sequence**: `x -> (x - width/2) / (width/2)`, `y -> (y - height/2) / (height/2)`, width / height from the pose header (300 x 300 if missing). Identical to before for 300 x 300 files; every frame, whatever its size, maps to [-1, 1]. All frames are treated as the same squashed square, which is what the 300 x 300 files already are. z is unaffected by the image size (it is MediaPipe's normalised z) and is still kept in source units.
* New functions: `header_dims`, `xy_geometry`, `normalized_xy`; `normalize_xyz` / `denormalize_xyz` take `dims` (for generated poses pass the canvas you want to draw on, default 300 x 300); `extract_raw` carries `dims`; shards store the image size of each sequence (`ShardStore.dims`).
* The normalisation hash changes (new hash `50878210e1` for the default settings), so new shards go to a new folder and can never mix with the old ones; the old `normalization_0a40d852b8.json`, the old shard folders and the old pilot are obsolete.
* Diagnostics now work per sequence: out-of-bounds means outside that sequence's own image, wrist distances and displacements are in normalised units, and the report lists the image sizes seen and each landmark group's valid rate, out-of-image rate and share of the target variance.
* **z gate relaxed:** a small z spread (hand z about 0.09 x the x spread) is now a WARN ("little weight in the loss"); only z that would dominate a squared-error loss (ratio above 10) is a FAIL in keep mode. Real hand z is small but not pixel-scale, so keeping z is the decision taken here; `Z_MODE = "scale"` remains available.
* Colab cells: `colab_cells/00_upload.py` (file chooser; places the 5 modules and the two cell scripts and keeps a copy on Drive), `06_restore_and_diagnose.py`, `07_convert.py` replace the earlier restore / conversion cells (tested in a fresh local process with stale old-normalisation files present, including a browser-renamed `name (1).py` upload). `05_sync_modules.py` is the alternative when the files are already in Drive's `code/` folder. `upload_to_colab/` holds copies of the 7 files to select in the chooser; regenerate it when a module changes.

## Update: restarting on a fresh VM (for example after the GPU quota ran out)

* `colab_cells/09_fresh_vm_setup.py`: restores the project files from Drive's `code/` folder, installs pose-format, clones and patches the repository. Order on a new VM: mount Drive, upload any changed cell files, 09, 06, 07, 08. Tested locally by wiping the simulated VM and keeping only the simulated Drive.
* `06_restore_and_diagnose.py` reuses a saved diagnostics report made with the same normalisation hash (`REUSE_DIAGNOSTICS = True`) instead of re-reading 300 sequences (about 10 minutes and a lot of Drive cache). Found and fixed by the local test: a print line still used the sample list that is not read when reusing.
* `colab_cells/08_checks_to_report.py` = notebook cells 12-25 as one script (generated by `build_notebook.py`).

## Update: running on a Windows PC with a GPU (`pc_run.py`)

* `pc_run.py` drives the same cell scripts with PC paths (`setup`, `prepare`, `convert`, `check`, `train`, `test`); `download_isign.py` fetches the CSV and the four pose parts from Hugging Face (resumable, size-checked; written but not run against Hugging Face); `SETUP_PC.md` has the step-by-step; `requirements_pc.txt`.
* New cell scripts shared with Colab: `10_setup_training.py` (refuses to train unless the latest report says READY; saved vocabulary and a config-compatibility check make resuming safe), `11_train.py` (real run, or `BENCHMARK_STEPS` to measure frames/s and GPU memory), `12_test_eval.py` (once, after training finished; free-running score limited to a number of batches).
* `isign_train.train(time_budget_s=...)` stops cleanly after the epoch that passes the budget (resumable, selection not marked finished); a stale `selection_done.json` is removed when training continues, so the test stage cannot run on an unfinished model; `final_test_evaluation(free_running_max_batches=...)`.
* `make_loader` uses persistent workers; verified with two worker processes under Windows `spawn`.
* `06_restore_and_diagnose.py`: override hook (`ISL_OVERRIDES`), configurable scan roots and disk-guard path; writes the split manifests and unmatched-UID logs on a first run (it used to assume they existed: found by testing on a fresh output folder).
* Tested locally on CPU with the synthetic archive through every stage, including the READY guard, resume, time budget, test-once guard, and a 2-worker training run. Not tested: real data, a real GPU, the Hugging Face download.

## Not changed, by design

`data.py`, `training.py`, `prediction.py` (still `torchtext`; replaced by the modules above), model architecture, landmark selection, frame width, counter convention, normalisation constants.
