"""Offline landmark-extraction experiment for ISLRTC clips (needs the extra venv, see docs/islrtc_pose_poc.md)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from islrtc_pose.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
