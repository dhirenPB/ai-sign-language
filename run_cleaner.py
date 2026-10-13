"""Run the Phase 1 cleaner without installing the package:  python run_cleaner.py data"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transcript_cleaner.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
