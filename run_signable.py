"""Make Phase 1 gloss more signable (synonyms + name spelling) without editing the gloss files:  python run_signable.py --gloss data/translations/*.gloss.json"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from isl_gloss.signable_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
