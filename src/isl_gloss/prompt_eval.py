"""Evaluation tool: run two prompt versions on the SAME inputs, keep every candidate output, record observed failure types.

This is analysis tooling, not part of the translation pipeline:
* The detectors below only OBSERVE. They never change a gloss, never run during translation, and are not a Layer 2
  (rule-based secondary pass) implementation. The pipeline's own checks are unchanged.
* The detectors are crude (word lists and string matching) and will miss problems and raise false alarms. Their counts
  are a rough guide to where a human reviewer should look. They are NOT a measure of translation accuracy, and a lower
  count is NOT evidence that a prompt is better. Only an ISL-fluent reviewer can judge that.
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

from .compare import cleaned_doc_from_text
from .config import TranslatorConfig
from .grouping import group_sentences, group_stats
from .models import FAILED, TRANSLATED
from .prepare import chunks_from_sentences
from .scoring import score_rows
from .sentences import build_sentences
from .translator import Translator
from .validate import _NOT_NAMES

TITLES = {"professor", "prof", "doctor", "dr", "mr", "mrs", "ms", "miss", "sir", "madam", "lord", "king", "queen", "teacher"}
_COND = re.compile(r"\b(if|unless|whenever|provided that|in case|as long as)\b", re.I)
_COND_TOK = {"IF", "UNLESS", "WHENEVER", "WHEN", "EVERY_TIME", "EVERYTIME", "IN_CASE", "SUPPOSE", "PROVIDED", "AS_LONG_AS"}
_NEG = re.compile(r"\b(not|no|never|none|nothing|nobody|nowhere|neither|nor|cannot|without)\b|n't\b|’t\b", re.I)
_NEG_TOK = {"NOT", "NO", "NEVER", "NONE", "NOTHING", "NOBODY", "NOWHERE", "NEITHER", "NOR", "CANNOT", "WITHOUT", "DONT",
            "DO_NOT", "CANT", "UNLESS"}
_CAUSE = re.compile(r"\b(because|therefore|hence|as a result|due to)\b", re.I)
_CAUSE_TOK = {"BECAUSE", "REASON", "WHY", "CAUSE", "THEREFORE", "HENCE", "SO", "RESULT", "DUE_TO", "AS_A_RESULT"}
_TIME = {"yesterday": {"YESTERDAY"}, "today": {"TODAY"}, "tomorrow": {"TOMORROW"}, "tonight": {"TONIGHT", "NIGHT"}}
FUNCTION_TOKENS = {"THAT", "AND", "IT", "THE", "A", "AN", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "OF", "TO", "THERE", "WOULD"}

FAILURE_TYPES = [
    "empty_gloss_not_flagged", "empty_gloss_flagged_for_review", "name_not_fingerspelled", "name_compound_label",
    "fs_multiword_token", "name_missing", "condition_dropped", "negation_dropped", "cause_dropped", "time_dropped",
    "number_dropped", "pronoun_replaced_by_noun", "notation_invalid_token", "notation_inconsistent_multiword",
    "alignment_problem",
]
_PERSON_PRONOUN_SRC = re.compile(r"\b(he|she|they|him|her|them|his|hers|their|theirs)\b", re.I)
_PERSON_PRONOUN_TOK = {"HE", "SHE", "THEY", "HIM", "HER", "THEM", "HIS", "HERS", "THEIR", "THEIRS", "PERSON", "INDEX", "PRO", "PRONOUN"}


def reviewer_columns(versions: list[str]) -> list[str]:
    cols = ["better_prompt (" + "/".join(versions) + "/same/neither)"]
    for stem in ("meaning_preserved", "notation_ok", "names_ok", "pronouns_ok"):
        cols += [f"{stem}_{v}" for v in versions]
    return cols + ["comments"]


def _parts(tokens: list[str]) -> set[str]:
    out = set()
    for t in tokens:
        body = t.removeprefix("FS:")
        out.add(body)
        out.update(p for p in re.split(r"[_-]", body) if p)
    return out


def names_in(text: str, known: Optional[set] = None) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']*", text)
    out = {w.upper() for i, w in enumerate(words) if i > 0 and w[0].isupper() and len(w) > 1
           and w.lower() not in _NOT_NAMES and w.lower() not in TITLES}
    if known and words and words[0].upper() in known:
        out.add(words[0].upper())
    return out


def diagnose(row: dict, passage_names: set, multiword_labels: set) -> list[str]:
    """Observed failure types for one result row (a ChunkResult.to_dict). Evaluation only: nothing is modified."""
    text, status = row["llm_input_text"] or row["source_text"], row["translation_status"]
    tokens = row["final_tokens"] if row["final_tokens"] is not None else []
    cand = row["candidate_tokens"] or []
    out: list[str] = []
    if status == TRANSLATED:
        if not tokens:
            out.append("empty_gloss_flagged_for_review" if (row["llm"] or {}).get("needs_human_review") else "empty_gloss_not_flagged")
            return out
    else:
        return out
    have = _parts(cand)  # judge what the MODEL produced, before any rule-pass change
    names = names_in(text, passage_names)
    ordinary = [t for t in cand if not t.startswith("FS:")]
    if any(n in re.split(r"[_-]", t) for t in ordinary for n in names):
        out.append("name_not_fingerspelled")
    if any(sum(p in names or p.lower() in TITLES for p in re.split(r"[_-]", t)) >= 2 for t in ordinary if "_" in t):
        out.append("name_compound_label")
    if any(t.startswith("FS:") and "_" in t[3:] for t in cand):
        out.append("fs_multiword_token")
    if any(n not in have for n in names):
        out.append("name_missing")
    if _COND.search(text) and not (have & _COND_TOK):
        out.append("condition_dropped")
    if _NEG.search(text) and not (have & _NEG_TOK):
        out.append("negation_dropped")
    if _CAUSE.search(text) and not (have & _CAUSE_TOK):
        out.append("cause_dropped")
    if _PERSON_PRONOUN_SRC.search(text) and not (have & _PERSON_PRONOUN_TOK):
        out.append("pronoun_replaced_by_noun")
    if any(re.search(rf"\b{w}\b", text, re.I) and not (have & exp) for w, exp in _TIME.items()):
        out.append("time_dropped")
    if any(num not in " ".join(cand) for num in re.findall(r"\d+", text)):
        out.append("number_dropped")
    if any(not re.fullmatch(r"(?:FS:)?[A-Z0-9]+(?:[_-][A-Z0-9]+)*", t) for t in cand):
        out.append("notation_invalid_token")
    if any(f"{a}_{b}" in multiword_labels for a, b in zip(cand, cand[1:])):
        out.append("notation_inconsistent_multiword")
    if any(w["code"] in ("source_text_mismatch", "source_text_not_echoed", "duplicate_result_for_sentence", "stray_results_ignored")
           for w in row["validation"]["warnings"]):
        out.append("alignment_problem")
    return out


def _function_words(row: dict) -> list[str]:
    return [t for t in (row["candidate_tokens"] or []) if t in FUNCTION_TOKENS]


def run_prompt_comparison(spec: dict, cfg: TranslatorConfig, make_translator: Optional[Callable[[TranslatorConfig], Translator]],
                          out_dir: Path, versions: list[str], dry_run: bool = False,
                          annotations: Optional[dict] = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    prepared = []
    for p in spec["passages"]:
        doc = cleaned_doc_from_text(p["text"], p["id"])
        sentences = build_sentences(doc, cfg)
        groups = group_sentences(sentences, cfg, source_id=p["id"])
        prepared.append((p, sentences, groups))
    info = [{"id": p["id"], "covers": p.get("covers"), "sentences": len(s), "groups": len(g), "grouping": group_stats(g)}
            for p, s, g in prepared]
    summary: dict[str, Any] = {"report_type": "prompt_comparison", "versions": versions, "dry_run": dry_run, "passages": info}
    if dry_run:
        summary["requests_per_version"] = sum(x["groups"] for x in info)
        return summary

    per_version: dict[str, dict] = {}
    all_rows: dict[str, list[dict]] = {}
    for v in versions:
        cfg_v = replace(cfg, prompt_version=v)
        tr = make_translator(cfg_v)
        before = tr.client.requests_made
        rows: list[dict] = []
        complete, stop = True, None
        assembly_ok = True
        for p, sentences, groups in prepared:
            outcome = tr.run(chunks_from_sentences(sentences), groups=groups)
            complete &= outcome.complete
            stop = stop or outcome.stop_reason
            assembly_ok &= bool(outcome.assembly.get("passed"))
            for r in outcome.results:
                d = r.to_dict(tr.metadata())
                d["passage_id"] = p["id"]
                rows.append(d)
        known = {pid: set().union(*(names_in(r["llm_input_text"] or r["source_text"]) for r in rows if r["passage_id"] == pid))
                 for pid in {r["passage_id"] for r in rows}}
        labels = {t for r in rows for t in (r["candidate_tokens"] or []) if "_" in t and not t.startswith("FS:")}
        for r in rows:
            r["observed_failures"] = diagnose(r, known[r["passage_id"]], labels)
            r["function_word_tokens"] = _function_words(r)
        all_rows[v] = rows
        failures = Counter(f for r in rows for f in r["observed_failures"])
        val_codes = Counter(w["code"] for r in rows for w in r["validation"]["warnings"] if w["severity"] != "info")
        translated = [r for r in rows if r["translation_status"] == TRANSLATED]
        usage = {**tr.stats, "requests_made": tr.client.requests_made - before}
        score = score_rows(rows, annotations, usage=usage)
        for r in rows:
            r["scored_issues"] = score.get("per_sentence_issues", {}).get(r["sentence_id"], [])
        per_version[v] = {
            "usage": usage, "score": score,
            "config": tr.metadata()["llm_config"],
            "complete": complete, "stop_reason": stop, "assembly_passed": assembly_ok,
            "requests": tr.client.requests_made - before,
            "sentences": len(rows), "translated": len(translated),
            "failed": sum(r["translation_status"] == FAILED for r in rows),
            "needs_human_review": sum(bool(r["needs_human_review"]) for r in rows),
            "model_flagged_review": sum(bool((r["llm"] or {}).get("needs_human_review")) for r in rows),
            "sentences_with_an_observed_failure": sum(bool(r["observed_failures"]) for r in rows),
            "observed_failure_types": {k: failures[k] for k in FAILURE_TYPES if failures[k]},
            "pipeline_validation_codes": dict(val_codes),
            "gloss_tokens": sum(len(r["final_tokens"] or []) for r in translated),
            "english_function_word_tokens": sum(len(r["function_word_tokens"]) for r in rows),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic(out_dir / f"results_prompt_{v}.json", json.dumps(
            {"prompt_version": v, "config": per_version[v]["config"], "complete": complete, "usage": usage, "results": rows},
            indent=2, ensure_ascii=False) + "\n")

    summary["per_version"] = per_version
    summary["all_versions_complete"] = all(x["complete"] for x in per_version.values())
    summary["interpretation"] = (
        "Counts come from crude automatic detectors on a small sample (observation only). They are not a measure of "
        "translation accuracy and do not show that either prompt is better. If a version did not finish, nothing can "
        "be concluded. An ISL-fluent reviewer must judge the side-by-side sheet.")
    _atomic(out_dir / "comparison.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_sheet(out_dir / "comparison_sheet.csv", versions, all_rows)
    return summary


def score_saved_results(result_files: list[Path], annotations: Optional[dict], out_dir: Path) -> dict[str, Any]:
    """Score already-saved results_prompt_<version>.json files side by side on the same sentences. Calls no model and
    changes no input file; the baseline runs (5.0/6.0/7.0) are read exactly as saved. A run that predates usage recording
    has no token/latency numbers, and that is reported as 'not recorded' rather than guessed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_version: dict[str, list[dict]] = {}
    summary: dict[str, Any] = {"report_type": "scored_results", "inputs": [str(p) for p in result_files], "per_version": {},
                               "interpretation": (
                                   "Counts come from provisional engineering annotations and crude detectors. A lower count, a shorter "
                                   "gloss or fewer review flags is NOT evidence of a better translation. Only an ISL-fluent reviewer "
                                   "can judge the side-by-side sheet.")}
    for path in result_files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        v = str(data["prompt_version"])
        if v in rows_by_version:
            raise ValueError(f"two result files for prompt version {v}")
        rows = data["results"]
        usage = data.get("usage")
        score = score_rows(rows, annotations, usage=usage if usage else {"recorded": False})
        for r in rows:
            r["scored_issues"] = score.get("per_sentence_issues", {}).get(r["sentence_id"], [])
            r.setdefault("observed_failures", [])
        rows_by_version[v] = rows
        summary["per_version"][v] = {"source_file": str(path), "complete": data.get("complete"), "config": data.get("config"),
                                     "score": score}
    sids = [tuple(r["sentence_id"] for r in rows) for rows in rows_by_version.values()]
    summary["same_sentences_in_every_file"] = len(set(sids)) == 1
    versions = sorted(rows_by_version)
    _atomic(out_dir / "scored_comparison.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_sheet(out_dir / "scored_comparison_sheet.csv", versions, rows_by_version)
    return summary


def _write_sheet(path: Path, versions: list[str], rows_by_version: dict[str, list[dict]]) -> None:
    first = versions[0]
    index = {v: {r["sentence_id"]: r for r in rows} for v, rows in rows_by_version.items()}
    cols = ["passage_id", "sentence_id", "source_text"]
    for v in versions:
        cols += [f"gloss_{v}", f"needs_review_{v}", f"uncertainty_notes_{v}", f"observed_failures_{v}", f"scored_issues_{v}"]
    cols += ["same_gloss"] + reviewer_columns(versions)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for base in rows_by_version[first]:
            sid = base["sentence_id"]
            row = {"passage_id": base["passage_id"], "sentence_id": sid, "source_text": base["source_text"]}
            glosses = []
            for v in versions:
                r = index[v].get(sid)
                glosses.append(r["final_gloss"] if r else None)
                row[f"gloss_{v}"] = (r["final_gloss"] if r and r["final_gloss"] is not None else f"({r['translation_status']})") if r else ""
                row[f"needs_review_{v}"] = r["needs_human_review"] if r else ""
                row[f"uncertainty_notes_{v}"] = (r["uncertainty_notes"] or "") if r else ""
                row[f"observed_failures_{v}"] = ";".join(r["observed_failures"]) if r else ""
                row[f"scored_issues_{v}"] = ";".join(r.get("scored_issues", [])) if r else ""
            row["same_gloss"] = len(set(glosses)) == 1
            w.writerow(row)
    os.replace(tmp, path)


def _atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
