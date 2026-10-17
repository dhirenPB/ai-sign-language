"""
Run the whole iSign pipeline on a (Windows) PC with the same tested code as the Colab cells.

  python pc_run.py setup --data-dir D:\\isign --out-dir D:\\isl_out     remember the folders, fetch + patch the repository
  python pc_run.py prepare                                              archive check, matching, splits, normalisation, diagnostics, pilot
  python pc_run.py convert [--review-scale]                             convert every shard (resumable); --review-scale = small set only
  python pc_run.py check                                                the pre-training checks (including the real CUDA step) and the decision
  python pc_run.py train [--benchmark 50] [--hours 8] [--run-name main] real training (resumable). --benchmark N only measures speed
  python pc_run.py test                                                 test-set evaluation, once, after training has finished

Common options: --set NAME=VALUE ... changes any setting of the cell scripts (for example --set BATCH_SIZE=16),
--workers N (data-loader processes for training), --cpu (allow CPU), --force (train although the decision is not READY).
Settings are remembered in pc_settings.json next to this file.
"""
import argparse
import ast
import builtins
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
CELLS = HERE / "colab_cells"
SETTINGS_FILE = HERE / "pc_settings.json"
REPO_COMMIT = "ff8b28cb0817d177f63dfa18c7c322806ee5dd96"
REPO_ZIP = f"https://github.com/BenSaunders27/ProgressiveTransformersSLP/archive/{REPO_COMMIT}.zip"
SETTINGS_MARKER = "# ---------- modules and libraries ----------"


def display(obj):
    print(obj.to_string() if hasattr(obj, "to_string") else obj)


builtins.display = display


def load_settings():
    if not SETTINGS_FILE.exists():
        raise SystemExit("No settings yet. Run:  python pc_run.py setup --data-dir <folder with iSign_v1.1.csv and pose_parts> --out-dir <work folder>")
    return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))


def folders(settings):
    data, out = Path(settings["data_dir"]), Path(settings["out_dir"])
    return {"DATA_DIR": data, "CSV_PATH": data / "iSign_v1.1.csv", "POSE_DIR": data / "pose_parts",
            "CODE_DIR": HERE, "REPO_DIR": out / "ProgressiveTransformersSLP", "WORK_DIR": out / "work",
            "DRIVE_OUT": out / "outputs", "PROC_PARENT": out / "outputs" / "processed",
            "LOCAL_SCAN_ROOTS": [str(out / "work"), str(out / "ProgressiveTransformersSLP")],
            "DISK_GUARD_PATH": str(out)}


def parse_sets(items):
    values = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            values[key] = ast.literal_eval(raw)
        except Exception:
            values[key] = raw
    return values


def overrides_for(args, **extra):
    o = folders(load_settings())
    o.update({"MIN_FREE_GB": 15, "NUM_WORKERS": 0})
    o.update(extra)
    o.update(parse_sets(args.set))
    return o


def run_script(name, ns):
    path = CELLS / name
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
    exec(compile(path.read_text(encoding="utf-8"), name, "exec"), ns)


def run_scripts(names, overrides):
    ns = {"__name__": "__main__", "ISL_OVERRIDES": overrides}
    for name in names:
        try:
            run_script(name, ns)
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            print(f"\n!!! stopped in {name}", flush=True)
            raise SystemExit(1)
    return ns


# ----------------------------------------------------------------------------- setup
def fetch_repository(repo_dir):
    zip_path = repo_dir.parent / "ProgressiveTransformersSLP.zip"
    print("downloading the Progressive Transformers repository at the tested commit ...")
    urllib.request.urlretrieve(REPO_ZIP, zip_path)
    tmp = repo_dir.parent / "_repo_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    shutil.move(str(inner), str(repo_dir))
    shutil.rmtree(tmp, ignore_errors=True)
    zip_path.unlink()


def stage_setup(args):
    settings = {"data_dir": str(Path(args.data_dir).resolve()), "out_dir": str(Path(args.out_dir).resolve())}
    SETTINGS_FILE.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    f = folders(settings)
    for key in ("WORK_DIR", "DRIVE_OUT", "PROC_PARENT"):
        f[key].mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import repo_patches as rp
    if not (f["REPO_DIR"] / "_originals").exists():
        fetch_repository(f["REPO_DIR"])
    rp.apply_all(f["REPO_DIR"], 5000)
    print("\nsettings saved to", SETTINGS_FILE)
    for key in ("DATA_DIR", "CSV_PATH", "POSE_DIR"):
        print(f"{key:9s} {f[key]}   {'ok' if f[key].exists() else 'NOT FOUND YET'}")
    try:
        import torch
        print(f"\ntorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}" +
              (f" | {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else ""))
    except ImportError:
        print("\ntorch is not installed yet: see SETUP_PC.md")
    print(f"free space on the output drive: {shutil.disk_usage(settings['out_dir']).free / 1e9:.0f} GB")


# ----------------------------------------------------------------------------- stages that need the pose archive
def stage_prepare(args):
    run_scripts(["06_restore_and_diagnose.py"], overrides_for(args, DIAG_N_SEQUENCES=args.diag_n))


def stage_convert(args):
    run_scripts(["06_restore_and_diagnose.py", "07_convert.py"],
                overrides_for(args, FULL_CONVERT=not args.review_scale, DIAG_N_SEQUENCES=args.diag_n))


def stage_check(args):
    run_scripts(["06_restore_and_diagnose.py", "07_convert.py", "08_checks_to_report.py"],
                overrides_for(args, FULL_CONVERT=args.full, DIAG_N_SEQUENCES=args.diag_n))


# ----------------------------------------------------------------------------- stages that only need the converted shards
def training_namespace(overrides):
    """Settings + normalisation + manifests, without opening the (possibly deleted) pose archive."""
    ns = {"__name__": "__main__", "ISL_OVERRIDES": overrides}
    head = (CELLS / "06_restore_and_diagnose.py").read_text(encoding="utf-8").split(SETTINGS_MARKER)[0]
    exec(compile(head, "settings", "exec"), ns)
    sys.path.insert(0, str(ns["CODE_DIR"]))
    import isign_pose as ip
    import pandas as pd
    ns["ip"], ns["pd"] = ip, pd
    ns["NORM"] = ip.make_normalization(ns["Z_MODE"], None, ns["CONF_THRESHOLD"], ns["ZERO_FILL_IS_MISSING"], ns["STORAGE_DTYPE"])
    ns["NORM_HASH"] = ip.params_hash(ns["NORM"])
    ns["PROC_ROOT"] = ns["PROC_PARENT"] / f"norm_{ns['NORM_HASH']}_archive_order"
    if not ns["PROC_ROOT"].exists():
        raise SystemExit(f"No converted data in {ns['PROC_ROOT']}. Run 'python pc_run.py convert' first.")
    ns["rows_of"] = {}
    for split in ns["SPLITS"]:
        manifest = pd.read_csv(ns["DRIVE_OUT"] / "splits" / f"manifest_{split}.csv")
        ns["rows_of"][split] = list(zip(manifest["uid"].astype(str), manifest["text"].astype(str)))
    return ns


def run_in_namespace(ns, names):
    for name in names:
        try:
            run_script(name, ns)
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            print(f"\n!!! stopped in {name}", flush=True)
            raise SystemExit(1)


def stage_train(args):
    extra = {"NUM_WORKERS": args.workers, "RUN_NAME": args.run_name, "FORCE_TRAIN": args.force,
             "ALLOW_CPU_TRAINING": args.cpu, "BENCHMARK_STEPS": args.benchmark}
    if args.hours is not None:
        extra["TIME_BUDGET_HOURS"] = args.hours
    if args.max_epochs is not None:
        extra["MAX_EPOCHS"] = args.max_epochs
    if args.batch_size is not None:
        extra["BATCH_SIZE"] = args.batch_size
    ns = training_namespace(overrides_for(args, **extra))
    run_in_namespace(ns, ["10_setup_training.py", "11_train.py"])


def stage_test(args):
    ns = training_namespace(overrides_for(args, NUM_WORKERS=0, RUN_NAME=args.run_name, ALLOW_CPU_TRAINING=args.cpu, FORCE_TRAIN=True))
    run_in_namespace(ns, ["10_setup_training.py", "12_test_eval.py"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override any setting of the cell scripts")
        p.add_argument("--cpu", action="store_true", help="allow training / testing without a GPU")
        p.add_argument("--diag-n", type=int, default=300, help="sequences in the diagnostics sample")

    p = sub.add_parser("setup")
    p.add_argument("--data-dir", required=True, help="folder with iSign_v1.1.csv and pose_parts/")
    p.add_argument("--out-dir", required=True, help="folder for the repository clone, work files and outputs")
    p.set_defaults(fn=stage_setup)
    p = sub.add_parser("prepare"); common(p); p.set_defaults(fn=stage_prepare)
    p = sub.add_parser("convert"); common(p); p.add_argument("--review-scale", action="store_true"); p.set_defaults(fn=stage_convert)
    p = sub.add_parser("check"); common(p); p.add_argument("--full", action="store_true", help="convert any missing shards first")
    p.set_defaults(fn=stage_check)
    p = sub.add_parser("train"); common(p)
    p.add_argument("--run-name", default="main"); p.add_argument("--benchmark", type=int, default=0, metavar="STEPS")
    p.add_argument("--hours", type=float, default=None, help="stop cleanly after the epoch that passes this time budget")
    p.add_argument("--workers", type=int, default=2); p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=stage_train)
    p = sub.add_parser("test"); common(p); p.add_argument("--run-name", default="main"); p.set_defaults(fn=stage_test)
    args = ap.parse_args()
    t0 = time.time()
    args.fn(args)
    print(f"\n[{args.stage}] finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
