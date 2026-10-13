"""Run the transcript -> ISL gloss translator without installing:

    python run_translator.py translate data/structured --dry-run     (no API key needed)
    python run_translator.py translate data/structured --limit 20    (first real test on the free tier)
    python run_translator.py evaluate
    python run_translator.py rules
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from isl_gloss.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
