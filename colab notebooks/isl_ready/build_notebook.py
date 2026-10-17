"""
Assemble isl_text_to_pose_pre_training_ready.ipynb from notebook_cells.txt and the module files.
Generator only: it does not execute any notebook cell.
  * `writefile <module>` cells embed the module file verbatim (written to /content/isl_code on Colab);
  * `commented ...` cells are the original Hugging Face -> Drive transfer cells, commented out.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "isl_text_to_pose_clean_pipeline.ipynb"
OUT = HERE.parent / "isl_text_to_pose_pre_training_ready.ipynb"
CODE_DIR_ON_COLAB = "/content/isl_code"

original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
orig_code = {}
for cell in original["cells"]:
    if cell["cell_type"] == "code":
        text = "".join(cell["source"])
        first = text.split("\n", 1)[0]
        if first.startswith("# CELL "):
            orig_code[int(first.split()[2])] = text


def comment_out(text):
    return "\n".join(("# " + line) if line.strip() else "#" for line in text.split("\n"))


def commented_cell(title, numbers):
    parts = [f"# {title}",
             "# ALREADY DONE: the dataset is on Drive. Everything below is commented out on purpose;",
             "# select all lines and press Ctrl+/ to re-enable. (Original code, unchanged.)", ""]
    for n in numbers:
        parts += [comment_out(orig_code[n]), ""]
    return "\n".join(parts).rstrip("\n")


raw_cells, current = [], None
for line in (HERE / "notebook_cells.txt").read_text(encoding="utf-8").split("\n"):
    if line.startswith("#%%CELL"):
        current = {"header": line.split()[1:], "lines": []}
        raw_cells.append(current)
    else:
        current["lines"].append(line)

cells = []
for rc in raw_cells:
    kind = rc["header"][0]
    body = "\n".join(rc["lines"]).strip("\n")
    if kind == "writefile":
        module = rc["header"][1]
        body = f"%%writefile {CODE_DIR_ON_COLAB}/{module}\n" + (HERE / module).read_text(encoding="utf-8").rstrip("\n")
        kind = "code"
    elif kind == "commented":
        title = {"meta": "CELLS 3-6 (original): Hugging Face login and metadata download",
                 "transfer": "CELL 10 (original): transfer the large pose/video parts from Hugging Face to Drive"}[rc["header"][1]]
        body = commented_cell(title, [int(x) for x in rc["header"][2:]])
        kind = "code"
    cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": body.splitlines(keepends=True)}
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "name": OUT.name, "gpuType": "T4"},
                   "accelerator": "GPU",
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUT.name}: {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")

# CELL 12 onward as one script for exec() in a running session (data checks -> model -> tests -> report)
code_cells = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
start = next(i for i, s in enumerate(code_cells) if s.lstrip().startswith("# CELL 12"))
script = "# Notebook CELLS 12-25 as one script. Needs the state created by 06_restore_and_diagnose.py and 07_convert.py.\n\n" + \
    "\n\n# " + "-" * 70 + "\n\n".join([""] + code_cells[start:]) + "\n"
(HERE / "colab_cells" / "08_checks_to_report.py").write_text(script, encoding="utf-8")
print(f"Wrote colab_cells/08_checks_to_report.py ({len(code_cells) - start} notebook cells, {len(script) / 1e3:.0f} KB)")
