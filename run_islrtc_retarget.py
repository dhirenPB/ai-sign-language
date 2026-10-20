"""EXPERIMENT: retarget extracted landmarks to the Yap & Render Y Bot rig (needs the pose venv: numpy)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from islrtc_pose.retarget_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
