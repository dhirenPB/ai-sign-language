"""Writers for translation results: full JSON audit trail, a reviewer CSV, and a quick-look text file."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import RESULT_SCHEMA_VERSION, __version__
from .models import TRANSLATED
from .translator import RunOutcome

REVIEW_COLUMNS = [
    "sentence_id", "group_id", "speaker_id", "start_timestamp", "source_text", "llm_input_text", "candidate_gloss",
    "final_gloss", "needs_human_review", "llm_status", "uncertain_parts", "uncertainty_notes", "rules_applied", "translation_status", "validation_status", "warnings",
    # to be filled in by the ISL-fluent reviewer
    "meaning_preserved", "grammatically_acceptable", "omissions", "additions", "consistent", "corrected_gloss", "comments",
]


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _review_row(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "sentence_id": d["sentence_id"], "group_id": d["group_id"], "speaker_id": d["speaker_id"],
        "start_timestamp": d["start_timestamp"], "source_text": d["source_text"], "llm_input_text": d["llm_input_text"],
        "candidate_gloss": d["candidate_gloss"], "final_gloss": d["final_gloss"],
        "needs_human_review": d["needs_human_review"], "llm_status": d["llm_status"], "uncertain_parts": "; ".join(d["uncertain_parts"] or []),
        "uncertainty_notes": d["uncertainty_notes"],
        "rules_applied": ";".join(r["rule_id"] for r in d["applied_rules"]),
        "translation_status": d["translation_status"], "validation_status": d["validation"]["status"],
        "warnings": ";".join(w["code"] for w in d["validation"]["warnings"] if w["severity"] != "info"),
    }


def write_results(out_dir: Path, name: str, outcome: RunOutcome, meta: dict[str, Any], source_file: Optional[str],
                  extra: Optional[dict[str, Any]] = None) -> dict[str, str]:
    out_dir = Path(out_dir)
    rows = [r.to_dict(meta) for r in outcome.results]
    doc = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "report_type": "isl_gloss_translation",
        "tool_version": __version__,
        "source_id": name,
        "source_file": source_file,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "review_status": "unreviewed",
        "review_note": "Gloss is a machine-generated candidate. It has NOT been reviewed by an ISL-fluent signer and must "
                       "not be treated as production-accurate.",
        "complete": outcome.complete,
        "stop_reason": outcome.stop_reason,
        "counts": outcome.counts(),
        "assembly_validation": outcome.assembly,
        "metadata": meta,
        **(extra or {}),
        "results": rows,
    }
    paths = {"json": out_dir / f"{name}.gloss.json", "csv": out_dir / f"{name}.review_sheet.csv",
             "txt": out_dir / f"{name}.gloss.txt"}
    _atomic(paths["json"], (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=paths["csv"].name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(_review_row(d) for d in rows)
    os.replace(tmp, paths["csv"])

    lines = [f"# {name}  (UNREVIEWED candidate gloss)", ""]
    for d in rows:
        lines.append(f"[{d['sentence_id']}] {d['source_text']}")
        if d["translation_status"] == TRANSLATED:
            flag = "" if d["validation"]["status"] == "pass" else f"   <{d['validation']['status']}>"
            lines.append(f"    GLOSS: {d['final_gloss']}{flag}")
        else:
            lines.append(f"    ({d['translation_status']})")
        lines.append("")
    _atomic(paths["txt"], "\n".join(lines).encode("utf-8"))
    return {k: str(v) for k, v in paths.items()}


def write_groups(out_dir: Path, name: str, sentences: list, groups: list, summary: dict[str, Any]) -> str:
    """Stage 2 + Stage 3 output for inspection: every sentence and every group, with the reasons for each boundary."""
    path = Path(out_dir) / f"{name}.groups.json"
    doc = {
        "report_type": "sentences_and_groups",
        "tool_version": __version__,
        "source_id": name,
        "summary": summary,
        "sentences": [s.to_dict() for s in sentences],
        "groups": [g.to_dict(include_sentences=False) for g in groups],
    }
    _atomic(path, (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return str(path)
