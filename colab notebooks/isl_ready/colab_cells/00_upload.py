# UPLOAD — choose the project files from your computer (Colab opens a file chooser; select all 7 at once).
import importlib, re, shutil, sys
from pathlib import Path
from google.colab import files

MODULES = ["isign_pose.py", "isign_report.py", "repo_patches.py", "isign_data.py", "isign_train.py"]
CELLS = ["06_restore_and_diagnose.py", "07_convert.py", "08_checks_to_report.py"]
CODE_DIR, CELLS_DIR = Path("/content/isl_code"), Path("/content/isl_cells")
DRIVE_CODE = Path("/content/drive/MyDrive/ISL_Project/pre_training_ready/code")      # a copy is kept here for later sessions
for d in (CODE_DIR, CELLS_DIR, DRIVE_CODE, DRIVE_CODE / "cells"):
    d.mkdir(parents=True, exist_ok=True)

uploaded = files.upload()
got = {}
for name in uploaded:
    clean = re.sub(r" \(\d+\)(?=\.py$)", "", name)                 # the browser renames repeated uploads "name (1).py"
    if clean in MODULES:
        target, drive_target = CODE_DIR / clean, DRIVE_CODE / clean
    elif clean in CELLS:
        target, drive_target = CELLS_DIR / clean, DRIVE_CODE / "cells" / clean
    else:
        print("ignored:", name)
        continue
    shutil.copy2(Path("/content") / name, target)
    Path("/content", name).unlink()
    shutil.copy2(target, drive_target)
    got[clean] = target
missing = [n for n in MODULES + CELLS if n not in got]
assert not missing, f"not uploaded: {missing} (select all 8 files in the chooser)"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for m in ("isign_pose", "isign_report", "repo_patches", "isign_data", "isign_train"):
    sys.modules.pop(m, None)
importlib.invalidate_caches()
import isign_pose as ip
assert ip.DEFAULT_NORMALIZATION["xy"]["mode"] == "image_size", "old isign_pose still loaded: restart the runtime, run this cell again"
for n in MODULES + CELLS:
    print(f"ok  {n:30s} {got[n].stat().st_size / 1e3:6.0f} KB")
print("\nisign_pose reloaded: x/y are normalised with each pose file's own image size.")
