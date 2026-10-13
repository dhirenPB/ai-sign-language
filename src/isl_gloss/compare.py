"""Compare the new pipeline (sentences -> groups -> LLM) with the old direct mode (cleaned chunks in fixed windows -> LLM)
on a small set of representative passages.

Automatic checks can show REGRESSIONS (more failures, omissions, lost numbers/names/negation, alignment errors). They
cannot show that the grouped output is BETTER ISL: that needs the ISL-fluent reviewer, who gets a side-by-side sheet.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from .config import TranslatorConfig
from .grouping import group_sentences, group_stats
from .models import FAILED, SKIPPED, TRANSLATED
from .prepare import chunks_from_doc, chunks_from_sentences
from .sentences import build_sentences
from .translator import RunOutcome, Translator

# Warning codes that signal possible meaning damage; the grouped mode should not produce more of them.
MEANING_CODES = ("possible_omission", "possible_addition", "number_not_preserved", "negation_may_be_lost",
                 "name_may_be_missing", "prose_in_gloss", "invalid_token_format")
REVIEWER_COLUMNS = ["better_version (direct/grouped/same)", "direct_meaning_ok", "grouped_meaning_ok",
                    "wrong_pronoun_reference", "added_info_from_neighbour", "comments"]


def cleaned_doc_from_text(text: str, source_id: str) -> dict:
    """Run a raw transcript through the existing cleaner (unchanged) to get the cleaned JSON document."""
    from transcript_cleaner.config import CleaningConfig
    from transcript_cleaner.pipeline import process_bytes

    return process_bytes(text.encode("utf-8"), f"{source_id}.txt", CleaningConfig(), source_id).doc


def _metrics(outcome: RunOutcome, requests: int) -> dict[str, Any]:
    rs = outcome.results
    codes = Counter(w["code"] for r in rs for w in r.validation["warnings"] if w["severity"] != "info")
    return {
        "sentences": len(rs),
        "translated": sum(r.status == TRANSLATED for r in rs),
        "failed": sum(r.status == FAILED for r in rs),
        "skipped": sum(r.status == SKIPPED for r in rs),
        "llm_uncertain": sum(bool(r.candidate and r.candidate.uncertain) for r in rs),
        "validation_review_required": sum(r.validation["status"] == "review_required" for r in rs),
        "meaning_warnings": {c: codes[c] for c in MEANING_CODES if codes[c]},
        "all_warnings": dict(codes),
        "gloss_tokens": sum(len(r.final_tokens) for r in rs if r.status == TRANSLATED),
        "assembly_passed": outcome.assembly.get("passed", False),
        "assembly_errors": [e["code"] for e in outcome.assembly.get("errors", [])],
        "requests": requests,
        "complete": outcome.complete,
        "stop_reason": outcome.stop_reason,
    }


def _verdict(direct: dict, grouped: dict) -> dict[str, Any]:
    if not (direct["complete"] and grouped["complete"]) or not direct["translated"] or not grouped["translated"]:
        why = {m: x["stop_reason"] for m, x in (("direct", direct), ("grouped", grouped)) if x["stop_reason"]}
        return {
            "automatic_regressions": [],
            "automatic_verdict": "INCOMPLETE: not every sentence was translated in both modes, so nothing can be concluded",
            "incomplete_reasons": why,
            "note": "Re-run the same command later; finished requests are cached.",
        }
    regressions = []
    if grouped["failed"] > direct["failed"]:
        regressions.append(f"more failed sentences ({direct['failed']} -> {grouped['failed']})")
    if not grouped["assembly_passed"]:
        regressions.append(f"alignment/assembly errors: {grouped['assembly_errors']}")
    for code in MEANING_CODES:
        a, b = direct["meaning_warnings"].get(code, 0), grouped["meaning_warnings"].get(code, 0)
        if b > a:
            regressions.append(f"{code}: {a} -> {b}")
    return {
        "automatic_regressions": regressions,
        "automatic_verdict": "no regression found by the automatic checks" if not regressions
        else "possible regression: review these before adopting grouping",
        "note": "Automatic checks cannot show better ISL. An ISL-fluent reviewer must judge the side-by-side sheet.",
    }


def run_compare(spec: dict, cfg: TranslatorConfig, make_translator: Optional[Callable[[str], Translator]], out_dir: Path,
                dry_run: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir)
    totals: dict[str, Counter] = {"direct": Counter(), "grouped": Counter()}
    passages_out, rows = [], []
    # one translator per mode, each with its own cache, so neither mode gets answers for free from the other
    trs = None if dry_run else {"direct": make_translator("direct"), "grouped": make_translator("grouped")}
    all_direct: list = []
    all_grouped: list = []
    requests = {"direct": 0, "grouped": 0}

    for p in spec["passages"]:
        doc = cleaned_doc_from_text(p["text"], p["id"])
        chunks = chunks_from_doc(doc, cfg)
        sentences = build_sentences(doc, cfg)
        groups = group_sentences(sentences, cfg, source_id=p["id"])
        info = {"id": p["id"], "description": p.get("description"), "look_for": p.get("look_for"),
                "cleaned_chunks": len(chunks), "sentences": len(sentences), "grouping": group_stats(groups)}
        if dry_run:
            info["direct_requests"] = -(-len(chunks) // cfg.batch_size)
            info["grouped_requests"] = len(groups)
            passages_out.append(info)
            continue

        before = trs["direct"].client.requests_made
        direct = trs["direct"].run(chunks)
        mid = trs["direct"].client.requests_made
        direct_requests = mid - before
        before = trs["grouped"].client.requests_made
        grouped = trs["grouped"].run(chunks_from_sentences(sentences), groups=groups)
        grouped_requests = trs["grouped"].client.requests_made - before
        requests["direct"] += direct_requests
        requests["grouped"] += grouped_requests
        all_direct.append(direct)
        all_grouped.append(grouped)
        info["direct"] = _metrics(direct, direct_requests)
        info["grouped"] = _metrics(grouped, grouped_requests)
        passages_out.append(info)

        by_parent: dict[str, list] = {}
        for r in grouped.results:
            by_parent.setdefault(r.prepared.source.source_metadata["parent_chunk_id"], []).append(r)
        for d in direct.results:
            kids = by_parent.get(d.prepared.source.chunk_id, [])
            g_gloss = " / ".join(k.to_dict({})["final_gloss"] or f"({k.status})" for k in kids)
            d_gloss = d.to_dict({})["final_gloss"] or f"({d.status})"
            flags = lambda rs: ";".join(sorted({w["code"] for r in rs for w in r.validation["warnings"] if w["severity"] != "info"}))
            rows.append({
                "passage": p["id"], "chunk_id": d.prepared.source.chunk_id, "source_text": d.prepared.source.text,
                "direct_gloss": d_gloss, "grouped_gloss": g_gloss, "same": d_gloss == g_gloss,
                "direct_flags": flags([d]), "grouped_flags": flags(kids),
                "grouped_group_id": ",".join(sorted({k.group_id or "" for k in kids})),
                "grouped_uncertainty_notes": " | ".join(k.candidate.notes for k in kids if k.candidate and k.candidate.notes),
            })

    summary: dict[str, Any] = {"report_type": "pipeline_comparison", "dry_run": dry_run, "passages": passages_out}
    if dry_run:
        summary["requests"] = {"direct": sum(x["direct_requests"] for x in passages_out),
                               "grouped": sum(x["grouped_requests"] for x in passages_out)}
    else:
        agg = {}
        for mode, outs in (("direct", all_direct), ("grouped", all_grouped)):
            merged = RunOutcome(results=[r for o in outs for r in o.results], complete=all(o.complete for o in outs),
                                 stop_reason=next((o.stop_reason for o in outs if o.stop_reason), None))
            merged.assembly = {"passed": all(o.assembly.get("passed") for o in outs),
                               "errors": [e for o in outs for e in o.assembly.get("errors", [])]}
            agg[mode] = _metrics(merged, requests[mode])
        summary["totals"] = agg
        summary["chunks_with_different_gloss"] = sum(not r["same"] for r in rows)
        summary["chunks_compared"] = len(rows)
        summary.update(_verdict(agg["direct"], agg["grouped"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / "comparison.json", (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        cols = list(rows[0].keys()) + REVIEWER_COLUMNS if rows else REVIEWER_COLUMNS
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix="comparison_sheet.csv.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, out_dir / "comparison_sheet.csv")
    return summary


def _write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
