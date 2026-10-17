# FRESH VM SETUP — use after the runtime was replaced (new session, CPU or GPU). Drive must be mounted first.
# Restores the project files from Drive, installs pose-format, clones the repository and applies the patches.
import importlib, shutil, subprocess, sys
from pathlib import Path

DRIVE_CODE = Path("/content/drive/MyDrive/ISL_Project/pre_training_ready/code")
CODE_DIR, CELLS_DIR, REPO_DIR = Path("/content/isl_code"), Path("/content/isl_cells"), Path("/content/ProgressiveTransformersSLP")
REPO_URL = "https://github.com/BenSaunders27/ProgressiveTransformersSLP.git"
MODULES = ["isign_pose.py", "isign_report.py", "repo_patches.py", "isign_data.py", "isign_train.py"]
CELLS = ["06_restore_and_diagnose.py", "07_convert.py", "08_checks_to_report.py"]

assert DRIVE_CODE.exists(), "Drive is not mounted, or ISL_Project/pre_training_ready/code is missing"
CODE_DIR.mkdir(exist_ok=True)
CELLS_DIR.mkdir(exist_ok=True)
for f in DRIVE_CODE.glob("*.py"):
    shutil.copy2(f, CODE_DIR / f.name)
for f in (DRIVE_CODE / "cells").glob("*.py"):
    shutil.copy2(f, CELLS_DIR / f.name)
missing = [n for n in MODULES if not (CODE_DIR / n).exists()] + [n for n in CELLS if not (CELLS_DIR / n).exists()]
assert not missing, f"missing on Drive (upload them again): {missing}"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pose-format"], check=True)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for m in ("isign_pose", "isign_report", "repo_patches", "isign_data", "isign_train"):
    sys.modules.pop(m, None)
importlib.invalidate_caches()
import repo_patches as rp

def _force_remove(func, path, _exc):          # read-only git files (Windows) must be made writable before deleting
    import os, stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR, onerror=_force_remove)
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
rp.apply_all(REPO_DIR, 5000)
print("\nready. Next: 06_restore_and_diagnose, 07_convert, 08_checks_to_report (each with exec(open(...).read())).")
