"""Start the interactive pipeline tester without installing (use the project's .venv so spaCy is available):

    .venv\\Scripts\\python.exe run_webui.py            then open http://127.0.0.1:8765/
    .venv\\Scripts\\python.exe run_webui.py --port 9000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from isl_gloss.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["serve", *sys.argv[1:]]))
