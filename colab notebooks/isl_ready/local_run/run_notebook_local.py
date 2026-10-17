"""
Execute the readiness notebook locally on a CPU against the synthetic archive (see make_synthetic_archive.py).

It runs the notebook's own cells, in order, in one namespace. Only the environment is adapted:
  * '/content/drive/MyDrive' and '/content' are mapped to a local folder,
  * google.colab.drive is a no-op stub, '%pip' lines are dropped, '%%writefile' cells write locally,
  * a few sizes in the settings are reduced so that a CPU finishes in minutes.

Usage:
  python run_notebook_local.py <notebook.ipynb> <local root> [--workers N]
      [--pre cell_file.py ...  --from "# CELL 12"]       # simulate a restarted kernel: run these cells first,
                                                         # then the notebook cells from the one starting with the title
"""
import argparse
import builtins
import json
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def install_stubs():
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: print("[stub] drive.mount", a[:1]),
                                        flush_and_unmount=lambda: print("[stub] flush_and_unmount"))
    def upload():
        """Stand-in for files.upload(): 'upload' the project files from this checkout into the local /content folder."""
        here = Path(__file__).resolve().parent.parent
        content = Path(os.environ["ISL_LOCAL_CONTENT"])
        names = {}
        for src in list(here.glob("isign_*.py")) + [here / "repo_patches.py"] + list((here / "colab_cells").glob("0[678]_*.py")):
            (content / src.name).write_bytes(src.read_bytes())
            names[src.name] = b""
        # a repeated upload gets "(1)" appended by the browser; make sure the cell copes
        (content / "isign_pose (1).py").write_bytes((here / "isign_pose.py").read_bytes())
        names["isign_pose (1).py"] = b""
        names.pop("isign_pose.py")
        (content / "isign_pose.py").unlink()
        return names

    colab.files = types.SimpleNamespace(upload=upload)
    google.colab = colab
    sys.modules.update({"google": google, "google.colab": colab})

    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

    builtins.display = display


install_stubs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pre", nargs="*", default=[])
    ap.add_argument("--from", dest="from_title", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    local_drive = (root / "drive" / "MyDrive").as_posix()
    local_content = (root / "content").as_posix()
    (root / "content").mkdir(exist_ok=True)
    os.environ["ISL_LOCAL_CONTENT"] = local_content

    overrides = [(r"^SHARD_SIZE = \d+", "SHARD_SIZE = 48"), (r"^DIAG_N_SEQUENCES = \d+", "DIAG_N_SEQUENCES = 60"),
                 (r"^NUM_WORKERS = \d+", f"NUM_WORKERS = {args.workers}"), (r"^BATCH_SIZE = \d+", "BATCH_SIZE = 4"),
                 (r"^OVERFIT_N = \d+", "OVERFIT_N = 16"), (r"^OVERFIT_STEPS = \d+", "OVERFIT_STEPS = 40"),
                 (r"^OVERFIT_BATCH = \d+", "OVERFIT_BATCH = 4"),
                 (r"^OVERFIT_N, OVERFIT_STEPS, OVERFIT_MAX_FRAMES, OVERFIT_BATCH = 32, 300, 250, 8",
                  "OVERFIT_N, OVERFIT_STEPS, OVERFIT_MAX_FRAMES, OVERFIT_BATCH = 16, 40, 250, 4")]

    def localise(src):
        src = src.replace("/content/drive/MyDrive", local_drive).replace("/content", local_content)
        return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%pip"))

    def reduce_sizes(code):
        if not re.search(r"^SHARD_SIZE = \d+", code, flags=re.M):
            return code
        for pattern, new in overrides:
            code = re.sub(pattern, new, code, flags=re.M)
        print("[override] reduced sizes for a CPU run")
        return code

    nb = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    queue = []
    for path in args.pre:
        queue.append((f"pre:{Path(path).name}", Path(path).read_text(encoding="utf-8")))
    started = args.from_title is None
    for i, src in enumerate(cells):
        first_line = next((l for l in src.split("\n") if l.strip()), "")
        if not started:
            started = first_line.startswith(args.from_title)
        if started:
            queue.append((f"cell {i}", src))

    G = {"__name__": "__main__"}
    t0 = time.time()
    for name, src in queue:
        first = src.split("\n", 1)[0]
        if first.startswith("%%writefile"):
            if args.pre:
                continue
            target = Path(localise(first.split(None, 1)[1]).strip())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(src.split("\n", 1)[1] + "\n", encoding="utf-8")
            print(f"[{name}] wrote {target.name}")
            continue
        if src.lstrip().startswith(("# CELLS 3-6", "# CELL 10 (original)")):
            print(f"[{name}] skipped (commented transfer cell)")
            continue
        title = next((l for l in src.split("\n") if l.strip()), "")[:90]
        print(f"\n[{name}] {title}   (+{time.time() - t0:.0f}s)", flush=True)
        try:
            exec(compile(reduce_sizes(localise(src)), name.replace(" ", "_"), "exec"), G)
        except BaseException:
            traceback.print_exc()
            print(f"\n!!! uncaught exception in {name}; stopping", flush=True)
            sys.exit(1)
    print(f"\nfinished all cells in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
