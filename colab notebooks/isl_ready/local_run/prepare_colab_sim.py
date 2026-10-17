"""Prepare the fake Drive like the user's real one before the new Colab cells run: modules in the code folder, stale
old-normalisation files present, no new outputs, no pilot. Usage: python prepare_colab_sim.py <local root>"""
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
isl_ready = Path(__file__).resolve().parent.parent
out = root / "drive" / "MyDrive" / "ISL_Project" / "pre_training_ready"
shutil.rmtree(out / "code", ignore_errors=True)            # like the real Drive: no project files there yet
for old in (root / "content" / "isl_code").glob("*.py"):    # and no (new) modules in the session either
    old.unlink()
shutil.rmtree(out / "processed", ignore_errors=True)
shutil.rmtree(root / "content" / "isl_work" / "pilot", ignore_errors=True)
for f in out.glob("normalization_*.json"):
    f.unlink()
(out / "normalization_0a40d852b8.json").write_text(json.dumps({"stale": "old normalisation"}), encoding="utf-8")
(out / "processed" / "norm_0a40d852b8_archive_order" / "train").mkdir(parents=True)
print("fake Drive prepared:", sorted(p.name for p in out.iterdir()))
