"""Index the ISLRTC dictionary without installing the package:  python run_islrtc_index.py --data-dir C:\path\to\islrtc"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from islrtc_index.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
