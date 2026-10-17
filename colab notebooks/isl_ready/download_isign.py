"""
Download the iSign CSV and the four pose archive parts from Hugging Face (resumable, size-verified, no Drive needed).

  set HF_TOKEN=hf_xxx            (PowerShell:  $env:HF_TOKEN = "hf_xxx")   <- your own token, with access to the gated dataset
  python download_isign.py --out D:\\isign

The files go to <out>\\iSign_v1.1.csv and <out>\\pose_parts\\iSign-poses_v1.1_part_aa .. ad (about 170 GB in total).
The dataset licence applies: keep these files on your own machine, do not redistribute them.
NOTE: written from the dataset's file names used earlier in this project; not run against Hugging Face by the assistant.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "Exploration-Lab/iSign"
CSV = "iSign_v1.1.csv"
POSE_PARTS = [f"iSign-poses_v1.1_part_{s}" for s in ("aa", "ab", "ac", "ad")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-gb", type=float, default=15.0)
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to your Hugging Face access token first (the dataset is gated).")
    out = Path(args.out)
    (out / "pose_parts").mkdir(parents=True, exist_ok=True)

    info = HfApi(token=token).dataset_info(REPO_ID, files_metadata=True)
    sizes = {s.rfilename: getattr(s, "size", None) for s in info.siblings}
    missing = [n for n in [CSV] + POSE_PARTS if n not in sizes]
    if missing:
        sys.exit(f"Not found in the repository: {missing}")
    plan = [(CSV, out)] + [(n, out / "pose_parts") for n in POSE_PARTS]
    todo = [(n, d) for n, d in plan if not (d / n).exists() or (d / n).stat().st_size != sizes[n]]
    need = sum(sizes[n] for n, _ in todo) / 1e9
    free = shutil.disk_usage(out).free / 1e9
    print(f"{len(todo)} file(s) to download, {need:.0f} GB; free space {free:.0f} GB")
    if free < need + args.margin_gb:
        sys.exit("Not enough free disk space for the download plus the safety margin.")
    for name, dest in todo:
        print(f"\ndownloading {name} ({sizes[name] / 1e9:.1f} GB) -> {dest}", flush=True)
        path = Path(hf_hub_download(repo_id=REPO_ID, filename=name, repo_type="dataset", token=token, local_dir=str(dest)))
        if path.stat().st_size != sizes[name]:
            sys.exit(f"Size mismatch for {name}: {path.stat().st_size} != {sizes[name]}. Run the command again to resume.")
        print("ok, size verified")
    print("\nall files present and size-verified in", out)


if __name__ == "__main__":
    main()
