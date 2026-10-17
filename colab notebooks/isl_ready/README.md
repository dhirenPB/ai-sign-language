# iSign English -> sign pose: pre-training readiness

Gets the iSign dataset and the Progressive Transformers SLP model (unchanged architecture) to the point where a controlled training experiment can
start, and provides the training / evaluation tooling for it. Nothing here is trained yet and the real GPU step has not run.

**Read first:** [`project_gist.txt`](project_gist.txt) (everything done so far and where to pick up), [`setup.txt`](setup.txt) (what must be installed and true),
[`SETUP_PC.md`](SETUP_PC.md) (step by step on a Windows PC with a GPU).

## Frame representation
75 landmarks (33 body + 21 left hand + 21 right hand = pose-format indices `0:33` and `501:543`), 225 coordinates `[x0,y0,z0,x1,...]` + 1 frame
counter (`frame_index / n_frames`) = **226 values per frame**. x/y are normalised per sequence with the image size from each pose header; z is kept.
Unreliable landmarks are stored as 0.0 and flagged in a separate validity array that the loss uses together with the padded-frame mask.

## Quick start (Windows PC, NVIDIA GPU)
```powershell
python pc_run.py setup --data-dir D:\isign --out-dir D:\isl_out    # folders + patched model repository
python pc_run.py prepare                                          # matching, split by video, normalisation, diagnostics
python pc_run.py convert                                          # all shards, resumable
python pc_run.py check                                            # pre-training checks incl. the real CUDA step -> READY / NOT READY
python pc_run.py train --benchmark 50                             # speed and GPU memory
python pc_run.py train                                            # real run, resumable (only starts when the last check said READY)
python pc_run.py test                                             # once, after training has finished
```
The dataset (`iSign_v1.1.csv` + `pose_parts/`, about 170 GB) must be in `--data-dir`; `download_isign.py` fetches it from Hugging Face.

## What is in this folder
| Path | Purpose |
|---|---|
| `isign_pose.py`, `isign_data.py`, `isign_train.py`, `isign_report.py`, `repo_patches.py` | the modules (archive access, preprocessing, shards, dataset, training, test registry, patches to the model repository) |
| `pc_run.py`, `download_isign.py`, `requirements_pc.txt` | PC front end, dataset download, packages |
| `colab_cells/` | the cell scripts shared by Colab and `pc_run.py` |
| `../isl_text_to_pose_pre_training_ready.ipynb` | the whole pipeline as one notebook (built by `build_notebook.py` from `notebook_cells.txt`) |
| `dry_tests.py`, `local_run/` | tests that need no GPU: numpy logic, patch replay, static checks, a synthetic-archive CPU run |
| `CHANGELOG.md`, `TEST_REPORT.md` | every change and why; what was verified and what was not |

## Status
Verified on a CPU with a synthetic archive that has the real file structure: all required checks pass except the CUDA step (`UNVERIFIED` without a GPU),
so the decision is `NOT READY` until `check` runs on a GPU. On the real data (Colab): archive access, UID matching, split by video and the coordinate
diagnostics were done and a review-scale subset was converted. See `TEST_REPORT.md`.

## Rules
Keep the dataset and everything derived from it (the shards) private; never commit them. Do not start full training without a READY decision and the
owner's go-ahead; the code enforces the READY guard and a test-once guard. Do not put access tokens in files.
