# Running everything on a Windows PC with a GPU (Quadro T1000)

The same code that ran on Colab, driven by `pc_run.py`. Everything below is PowerShell. Tested by the assistant on a Windows laptop with a CPU only,
using a synthetic archive; **not yet run on the real data, on the T1000, or against Hugging Face.**

## 1. One-time setup

1. **Driver.** `nvidia-smi` must work. Note the driver version and the memory (4 GB or 8 GB). If it is missing or old (below about 525), install the current
   NVIDIA driver for the Quadro T1000 from nvidia.com. You do **not** need to install the CUDA Toolkit: the PyTorch wheel carries its own CUDA runtime.
2. **Python 3.11** from python.org (tick "Add python.exe to PATH").
3. **Virtual environment and PyTorch with CUDA**
   ```powershell
   py -3.11 -m venv C:\isl\venv
   C:\isl\venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   It must print `True` and the GPU name. If it prints `False`, the driver is too old for this wheel: update it, or use `cu121` / `cu118` in the URL.
4. **The other packages**, from the folder that contains `pc_run.py`:
   ```powershell
   pip install -r requirements_pc.txt
   ```
5. **Copy the project folder** (`isl_ready`: all `.py` files, `colab_cells/`, `requirements_pc.txt`; you do not need `upload_to_colab/`, `local_run/` or `patched_repo_files/`) to the PC.

## 2. The data (about 170 GB; 290 GB free is enough)

Put it in one folder with this layout:
```
D:\isign\iSign_v1.1.csv
D:\isign\pose_parts\iSign-poses_v1.1_part_aa  ...  part_ad
```
* **From Hugging Face (resumable, size-checked):** create an access token on huggingface.co (the dataset is gated, your account already has access), then
  ```powershell
  $env:HF_TOKEN = "hf_xxxxxxxx"      # your own token; never paste it into a chat or a file
  python download_isign.py --out D:\isign
  ```
* **Or from your Drive:** download the CSV and the four pose parts in the browser (or with Drive for desktop, "available offline") into the folders above.

The dataset licence applies: keep these files on your machine.

## 3. Run the stages, in order

Run these from the folder with `pc_run.py`, in the activated environment. Every stage can be stopped and run again.

| Step | Command | What it does | Rough time |
|---|---|---|---|
| 1 | `python pc_run.py setup --data-dir D:\isign --out-dir D:\isl_out` | remembers the folders, fetches and patches the model repository | 1 min |
| 2 | `python pc_run.py prepare` | archive access, UID matching, split by video, normalisation, diagnostics, pilot. **Read the flags it prints.** | minutes |
| 3 | `python pc_run.py convert` | converts every shard (about 127,000 sequences), resumable; stops cleanly if the disk gets low | about 1-3 h (estimate) |
| 4 | `python pc_run.py check` | all pre-training checks, including the real CUDA step on the T1000, then the READY / NOT READY decision | 10-30 min |
| 5 | `python pc_run.py train --benchmark 50` | measures speed and GPU memory, writes nothing | 1-2 min |
| 6 | `python pc_run.py train` | the real run; add `--hours 8` to stop cleanly after the epoch that passes 8 hours | many hours |
| 7 | `python pc_run.py test` | test-set score, **once**, after training has finished | minutes to an hour |

Notes
* **Resuming.** Ctrl+C (or a power cut) loses at most the epoch in progress. Run the same `train` command again to continue. Epochs, learning rate, best checkpoint and early stopping are restored.
* **Training only starts if the last `check` said READY.** `--force` overrides that on purpose. `--cpu` allows running without a GPU.
* **Out of GPU memory?** `python pc_run.py train --batch-size 4` (the default is 8). With 4 GB it should fit, but the benchmark step prints the real peak memory.
* **Different settings:** `--set NAME=VALUE ...` changes any setting of the scripts, for example `--set LEARNING_RATE=0.0005 MAX_EPOCHS=40`.
* **Another experiment** needs a new name: `--run-name second_try`. Runs live in `D:\isl_out\outputs\runs\<name>\` (config, vocabulary, normalisation, `best.ckpt`, `last.ckpt`, `history.json`).
* **After step 3** the 170 GB of pose parts are no longer needed by `train` and `test`; keep the CSV and `D:\isl_out`.
* **Windows power settings:** set sleep to "Never" while training. Keep the PC plugged in.
* **Speed:** the T1000 is slower than a Colab T4 (my guess: roughly 3x). Step 5 gives the real number.
