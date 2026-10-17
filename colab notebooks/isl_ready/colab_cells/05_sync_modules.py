# SYNC MODULES — copy the current project modules from your Drive into this session and reload them.
import importlib, shutil, sys
from pathlib import Path

CODE_DIR = Path("/content/isl_code")
DRIVE_CODE = Path("/content/drive/MyDrive/ISL_Project/pre_training_ready/code")
assert DRIVE_CODE.exists(), f"{DRIVE_CODE} not found"
CODE_DIR.mkdir(exist_ok=True)
for f in sorted(DRIVE_CODE.glob("*.py")):
    shutil.copy2(f, CODE_DIR / f.name)
    print("copied", f.name, f"({f.stat().st_size / 1e3:.0f} KB)")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for m in ("isign_pose", "isign_report", "repo_patches", "isign_data", "isign_train"):
    sys.modules.pop(m, None)
importlib.invalidate_caches()
import isign_pose as ip
assert ip.DEFAULT_NORMALIZATION["xy"]["mode"] == "image_size", "old isign_pose still loaded: restart the runtime and run this cell again"
print("isign_pose reloaded: x/y are normalised with each pose file's own image size")
