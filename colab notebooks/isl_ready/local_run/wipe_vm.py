"""Simulate a brand-new Colab VM: delete the local /content (code, repository clone, pilot) but keep the fake Drive."""
import os
import shutil
import stat
import sys
from pathlib import Path


def force_remove(func, path, _exc):          # git object files are read-only on Windows
    os.chmod(path, stat.S_IWRITE)
    func(path)


shutil.rmtree(Path(sys.argv[1]) / "content", onerror=force_remove)
print("local /content wiped; fake Drive kept")
