"""Layer 2 orchestration: saved Layer 1 results -> spaCy candidate -> comparison -> selection -> artefacts.

* Layer 1 is never run and never modified: the input file is only read (its SHA-256 is recorded before and after), and no
  LLM request is made. Output goes to its own folder, which must not be the folder holding the Layer 1 file.
* Re-running with the same input and settings rewrites the same files atomically (same names, same content apart from the
  timestamp in the manifest); nothing is appended, so there are no duplicates.
* A failure for one sentence, in either translator, is recorded for that sentence and never stops the run or erases the other
  translator's result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .. import __version__
from .candidate_comparator import Comparison, compare_candidates
from .selection import POLICY_VERSION, SelectionPolicy, select
from .spacy_translator import ADAPTER_VERSION, REFERENCE, SETUP_HELP, SpacyResult, SpacyTranslator, dependency_versions

LAYER2_SCHEMA_VERSION = "1.0"
REVIEWER_COLUMNS = ["preferred_candidate (LLM/SPACY/BOTH_ACCEPTABLE/NEITHER_ACCEPTABLE/UNCERTAIN)", "meaning_preserved_llm",
                    "meaning_preserved_spacy", "linguistically_acceptable_llm", "linguistically_acceptable_spacy",
                    "missing_or_introduced_meaning", "reviewer_comments"]
HONESTY_NOTE = ("Candidates are machine-generated and UNREVIEWED. Technical checks and warning counts say nothing about linguistic "
                "accuracy. No ISL-fluent reviewer has judged either candidate, so it is not known whether combining the two "
                "approaches improves the gloss.")


@dataclass
class Layer2Config:
    enabled: bool = True  # False: run_layer2 returns immediately and writes nothing
    llm_field: str = "candidate_tokens"  # Candidate A: the model's own gloss; "final_tokens" = after Layer 1's small rule pass
    policy: SelectionPolicy = field(default_factory=SelectionPolicy)

    def __post_init__(self):
        if self.llm_field not in ("candidate_tokens", "final_tokens"):
            raise ValueError("llm_field must be 'candidate_tokens' or 'final_tokens'")

    @classmethod
    def from_json(cls, path: str | Path) -> "Layer2Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = SelectionPolicy(**data.pop("policy", {}))
        unknown = set(data) - {"enabled", "llm_field"}
        if unknown:
            raise ValueError(f"Unknown Layer 2 config keys: {sorted(unknown)}")
        return cls(policy=policy, **data)


# --------------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_layer1_results(path: str | Path) -> tuple[dict, list[dict]]:
    """Read a saved Layer 1 file (results_prompt_<v>.json or <id>.gloss.json). Read-only."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: not a Layer 1 results file (no non-empty 'results' list).")
    ids = [r.get("sentence_id") for r in rows]
    if any(not isinstance(i, str) for i in ids):
        raise ValueError(f"{path}: a result row has no sentence_id.")
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(f"{path}: duplicate sentence ids {dupes[:5]}")
    return doc, rows


def _atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    import io
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    _atomic(path, buf.getvalue(), encoding="utf-8-sig")


# --------------------------------------------------------------------------------------------------------------------
def _warnings_view(c) -> list[dict]:
    return ([{"kind": "structural", "code": i["code"], "severity": "invalid", "message": i["message"]} for i in c.structural_issues]
            + [{"kind": w["feature"], "code": w["code"], "severity": w["severity"], "message": w["message"]} for w in c.warnings])


def _row(l1: dict, spacy: SpacyResult, cmp: Comparison, sel, text: str) -> dict[str, Any]:
    return {
        "sentence_id": l1["sentence_id"], "group_id": l1.get("group_id"), "source_text": l1.get("source_text"), "cleaned_source_text": text,
        "speaker_id": l1.get("speaker_id"), "start_timestamp": l1.get("start_timestamp"), "end_timestamp": l1.get("end_timestamp"),
        "turn_index": l1.get("turn_index"),
        "llm_gloss": cmp.llm.gloss if cmp.llm.tokens else (cmp.llm.gloss or ""), "spacy_gloss": spacy.spacy_gloss,
        "llm_status": cmp.llm.translator_status, "spacy_status": spacy.spacy_status,
        "llm_usable": cmp.llm.usable, "spacy_usable": cmp.spacy.usable,
        "llm_warnings": _warnings_view(cmp.llm), "spacy_warnings": _warnings_view(cmp.spacy),
        "llm_stated_review_flag": cmp.llm.stated_review_flag, "llm_uncertainty_notes": l1.get("uncertainty_notes"),
        "spacy_error": spacy.spacy_error, "shared_warnings": cmp.shared_warnings,
        "candidate_difference": cmp.difference, **sel.to_dict(),
    }


def build_items(rows: list[dict], translator: SpacyTranslator, cfg: Layer2Config,
                log: Callable[[str], None] = lambda m: None) -> tuple[list[SpacyResult], list[dict]]:
    """Layer 2 for already-loaded Layer 1 result rows: spaCy on the same cleaned text, comparison, selection. Pure computation:
    nothing is read from or written to disk, and no LLM is called. Used by run_layer2 and by the interactive demo page."""
    spacy_results: list[SpacyResult] = []
    items: list[dict] = []
    for l1 in rows:
        text = l1.get("llm_input_text") if l1.get("llm_input_text") is not None else (l1.get("source_text") or "")
        meta = {k: l1.get(k) for k in ("speaker_id", "start_timestamp", "end_timestamp", "turn_index")}
        res = translator.translate(l1["sentence_id"], text, group_id=l1.get("group_id"), metadata=meta)
        if res.spacy_status in ("error", "unavailable"):
            log(f"spaCy {res.spacy_status} for {l1['sentence_id']}: {(res.spacy_error or {}).get('kind')}: {(res.spacy_error or {}).get('message', '')[:160]}")
        cmp = compare_candidates(l1["sentence_id"], text, l1, res, cfg.llm_field)
        sel = select(cmp, cfg.policy)
        spacy_results.append(res)
        items.append(_row(l1, res, cmp, sel, text))
    return spacy_results, items


def run_layer2(layer1_path: str | Path, out_dir: str | Path, cfg: Optional[Layer2Config] = None,
               translator: Optional[SpacyTranslator] = None, log: Callable[[str], None] = lambda m: None) -> dict[str, Any]:
    cfg = cfg or Layer2Config()
    if not cfg.enabled:
        return {"enabled": False, "message": "Layer 2 is disabled by configuration; nothing was run or written."}
    layer1_path, out_dir = Path(layer1_path), Path(out_dir)
    if out_dir.resolve() == layer1_path.resolve().parent:
        raise ValueError("The Layer 2 output folder must be different from the folder holding the Layer 1 file, so Layer 1 artefacts "
                         "are never mixed with or overwritten by Layer 2 output.")
    translator = translator or SpacyTranslator()
    hash_before = sha256_file(layer1_path)
    doc, rows = load_layer1_results(layer1_path)

    spacy_results, items = build_items(rows, translator, cfg, log)

    hash_after = sha256_file(layer1_path)
    summary = summarise(items, spacy_results)
    manifest = {
        "schema_version": LAYER2_SCHEMA_VERSION, "tool_version": __version__, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layer1_input": {"path": str(layer1_path), "sha256_before": hash_before, "sha256_after": hash_after, "unchanged": hash_before == hash_after,
                         "layer1_complete": doc.get("complete"), "layer1_stop_reason": doc.get("stop_reason"),
                         "prompt_version": doc.get("prompt_version") or (rows[0].get("metadata") or {}).get("prompt_version"),
                         "llm_config": doc.get("config") or (rows[0].get("metadata") or {}).get("llm_config")},
        "llm_requests_made_by_layer2": 0, "candidate_a_field": cfg.llm_field,
        "spacy": {**dependency_versions(), "reference": REFERENCE}, "selection_policy": cfg.policy.to_dict(),
        "selection_policy_version": POLICY_VERSION, "adapter_version": ADAPTER_VERSION, "setup_help": SETUP_HELP,
        "review_status": "unreviewed", "note": HONESTY_NOTE,
    }
    write_artifacts(out_dir, items, spacy_results, summary, manifest)
    return {"enabled": True, "summary": summary, "manifest": manifest, "rows": items, "out_dir": str(out_dir)}


# --------------------------------------------------------------------------------------------------------------------
def summarise(items: list[dict], spacy_results: list[SpacyResult]) -> dict[str, Any]:
    n = len(items)
    sel = Counter(r["selection_status"] for r in items)
    rules = Counter(r["selection_rule"] for r in items)
    basis = Counter(r["selection_basis"] for r in items)
    llm_state = Counter("failed" if r["llm_status"] not in ("translated", "skipped_no_translatable_text") else
                        ("empty" if any(w["code"] == "empty_output" for w in r["llm_warnings"]) else
                         ("skipped" if r["llm_status"] == "skipped_no_translatable_text" else "success")) for r in items)
    sp_state = Counter(r.spacy_status for r in spacy_results)

    def warning_counts(key):
        return dict(sorted(Counter(w["code"] for r in items for w in r[key] if w["severity"] == "warning").items()))

    def structural_counts(key):
        return dict(sorted(Counter(w["code"] for r in items for w in r[key] if w["severity"] == "invalid").items()))

    both = [r for r in items if r["llm_usable"] and r["spacy_usable"]]
    return {
        "sentences": n,
        "spacy": {"success": sp_state.get("ok", 0), "empty": sp_state.get("empty", 0), "failed": sp_state.get("error", 0),
                  "dependency_unavailable": sp_state.get("unavailable", 0)},
        "llm": {"success": llm_state.get("success", 0), "empty": llm_state.get("empty", 0), "failed": llm_state.get("failed", 0),
                "skipped_no_translatable_text": llm_state.get("skipped", 0)},
        "structural_validation_failures": {"llm": sum(not r["llm_usable"] for r in items), "spacy": sum(not r["spacy_usable"] for r in items),
                                           "llm_by_code": structural_counts("llm_warnings"), "spacy_by_code": structural_counts("spacy_warnings")},
        "both_candidates_usable": len(both),
        "candidates_differ_after_normalisation": sum(not r["candidate_difference"]["identical_after_normalisation"] for r in both),
        "candidates_identical_after_normalisation": sum(r["candidate_difference"]["identical_after_normalisation"] for r in both),
        "candidates_identical_as_given": sum(r["candidate_difference"]["identical_as_given"] for r in both),
        "selection": {"status_counts": dict(sorted(sel.items())), "rule_counts": dict(sorted(rules.items())), "basis_counts": dict(sorted(basis.items())),
                      "carried_forward": sum(r["selected_candidate"] is not None for r in items),
                      "llm_primary_by_policy_default": sel.get("llm_primary", 0), "candidates_agree": sel.get("candidates_agree", 0),
                      "spacy_selected": sum(r["selected_candidate"] == "SPACY" for r in items),
                      "spacy_selected_as_technical_fallback": sel.get("spacy_fallback_selected", 0),
                      "llm_declined_spacy_offered_not_substituted": sel.get("llm_abstained_alternative_available", 0),
                      "llm_unusable_spacy_rejected": sel.get("llm_failed_alternative_rejected", 0), "both_invalid": sel.get("both_invalid", 0),
                      "needs_human_review_total": sum(bool(r["needs_human_review"]) for r in items),
                      "selected_candidate_counts": dict(Counter(r["selected_candidate"] or "NONE" for r in items)),
                      "evidence_based_selections": 0,
                      "note": "'llm_primary' is a policy default (the LLM is the primary candidate), not a finding that it beats spaCy. No selection "
                              "here rests on human-reviewed evidence."},
        "heuristic_warning_counts": {"llm": warning_counts("llm_warnings"), "spacy": warning_counts("spacy_warnings"),
                                     "shared_by_both": dict(Counter(c for r in items for c in r["shared_warnings"])),
                                     "caution": "Heuristic flags for a reviewer to look at. They are not translation errors and say nothing about accuracy."},
        "linguistic_accuracy_established": False,
        "note": HONESTY_NOTE,
    }


ARTIFACT_NAMES = ("spacy_raw_outputs.json", "candidates_side_by_side.json", "candidates_side_by_side.csv", "selection_decisions.csv",
                  "warnings.csv", "summary.json", "run_manifest.json", "summary_report.md")


def check_outputs_writable(out_dir: Path) -> None:
    """Fail BEFORE writing anything if an existing output file is locked (for example open in Excel on Windows), so a re-run can
    never leave the folder half old and half new."""
    locked = []
    for name in ARTIFACT_NAMES:
        p = Path(out_dir) / name
        if p.exists():
            try:
                with open(p, "ab"):
                    pass
            except OSError:
                locked.append(name)
    if locked:
        raise OSError(f"Cannot update {out_dir}: these files are locked, probably open in another program ({', '.join(locked)}). "
                      "Close them and run again; nothing was changed.")


def write_artifacts(out_dir: Path, items: list[dict], spacy_results: list[SpacyResult], summary: dict, manifest: dict) -> None:
    out_dir = Path(out_dir)
    check_outputs_writable(out_dir)
    _atomic(out_dir / "spacy_raw_outputs.json", _json({"report_type": "layer2_spacy_raw_outputs", "schema_version": LAYER2_SCHEMA_VERSION,
                                                       "spacy": manifest["spacy"], "results": [r.to_dict() for r in spacy_results]}))
    _atomic(out_dir / "candidates_side_by_side.json", _json({"report_type": "layer2_candidates", "schema_version": LAYER2_SCHEMA_VERSION,
                                                              "review_status": "unreviewed", "note": HONESTY_NOTE, "results": items}))
    cols = ["sentence_id", "group_id", "source_text", "cleaned_source_text", "llm_gloss", "spacy_gloss", "llm_status", "spacy_status",
            "llm_warnings", "spacy_warnings", "candidate_difference", "primary_candidate", "selection_status", "selected_candidate", "selection_basis",
            "selection_reason", "needs_human_review", "alternative_usable_as_fallback", "llm_stated_review_flag", "llm_uncertainty_notes", "selection_rule"] + REVIEWER_COLUMNS
    flat = []
    for r in items:
        d = r["candidate_difference"]
        diff = ("identical" if d["identical_after_normalisation"] else
                "same words, different order" if d["same_words_different_order"] else
                f"only LLM: {' '.join(d['only_in_llm'])} | only spaCy: {' '.join(d['only_in_spacy'])}" if d["comparable"] else "not comparable")
        flat.append({**r, "llm_warnings": ";".join(w["code"] for w in r["llm_warnings"]),
                     "spacy_warnings": ";".join(w["code"] for w in r["spacy_warnings"]), "candidate_difference": diff})
    _csv(out_dir / "candidates_side_by_side.csv", cols, flat)
    _csv(out_dir / "selection_decisions.csv", ["sentence_id", "primary_candidate", "selection_status", "selected_candidate", "selection_basis",
                                               "selection_rule", "needs_human_review",
                                               "selection_reason"], items)
    wrows = [{"sentence_id": r["sentence_id"], "translator": t, "kind": w["kind"], "code": w["code"], "severity": w["severity"], "message": w["message"]}
             for r in items for t, key in (("LLM", "llm_warnings"), ("SPACY", "spacy_warnings")) for w in r[key]]
    _csv(out_dir / "warnings.csv", ["sentence_id", "translator", "kind", "code", "severity", "message"], wrows)
    _atomic(out_dir / "summary.json", _json(summary))
    _atomic(out_dir / "run_manifest.json", _json(manifest))
    _atomic(out_dir / "summary_report.md", render_report(summary, manifest))


def render_report(s: dict, m: dict) -> str:
    sel, sp, llm = s["selection"], s["spacy"], s["llm"]
    v = m["spacy"]
    L = [f"# Layer 2 evaluation: LLM candidate vs spaCy candidate", "",
         f"**Status: {m['review_status']}.** {HONESTY_NOTE}", "",
         "## Inputs", f"* Layer 1 file: `{m['layer1_input']['path']}` (prompt {m['layer1_input']['prompt_version']}; unchanged by this run: "
         f"{m['layer1_input']['unchanged']})",
         *([f"* **WARNING: the Layer 1 run was incomplete** ({m['layer1_input']['layer1_stop_reason']}); sentences it did not translate show as unusable LLM candidates."]
           if m["layer1_input"].get("layer1_complete") is False else []),
         f"* spaCy {v['spacy']}, en_core_web_sm {v['en_core_web_sm']}, nltk {v['nltk']}; reference commit `{v['reference']['commit'][:7]}`",
         f"* LLM requests made by Layer 2: {m['llm_requests_made_by_layer2']}; Candidate A field: `{m['candidate_a_field']}`",
         f"* Selection policy: {json.dumps(m['selection_policy'])}", "", "## Counts",
         f"* Sentences: {s['sentences']}",
         f"* spaCy: {sp['success']} produced a gloss, {sp['empty']} empty, {sp['failed']} failed, {sp['dependency_unavailable']} dependency unavailable",
         f"* LLM (Layer 1): {llm['success']} produced a gloss, {llm['empty']} empty, {llm['failed']} failed, {llm['skipped_no_translatable_text']} nothing to translate",
         f"* Technically unusable: LLM {s['structural_validation_failures']['llm']}, spaCy {s['structural_validation_failures']['spacy']}",
         f"* Both usable: {s['both_candidates_usable']}; of those, {s['candidates_differ_after_normalisation']} differ and "
         f"{s['candidates_identical_after_normalisation']} are identical after normalising case/FS:/separators",
         "", "## Selection",
         "The LLM is the primary candidate; spaCy is an independent alternative that is always generated and kept.",
         f"* Carried forward: {sel['carried_forward']} of {s['sentences']}. LLM kept as primary by policy default: {sel['llm_primary_by_policy_default']} "
         f"(a default, not a finding that it beats spaCy); the two agree: {sel['candidates_agree']}",
         f"* spaCy selected: {sel['spacy_selected']} (technical fallback after an LLM failure: {sel['spacy_selected_as_technical_fallback']})",
         f"* LLM declined on purpose, spaCy offered but NOT substituted: {sel['llm_declined_spacy_offered_not_substituted']}",
         f"* LLM unusable and spaCy rejected by the fallback check: {sel['llm_unusable_spacy_rejected']}; neither usable: {sel['both_invalid']}",
         f"* Selections resting on human-reviewed evidence: {sel['evidence_based_selections']}",
         f"* Flagged for human review: {sel['needs_human_review_total']} of {s['sentences']} (the LLM's own flags, warnings, or an LLM failure). "
         "Every row is still in the reviewer sheet, because nothing is reviewed yet.",
         f"* Status counts: {sel['status_counts']}; rules: {sel['rule_counts']}; basis: {sel['basis_counts']}", "",
         "## Heuristic warnings (not errors)", f"* LLM: {s['heuristic_warning_counts']['llm']}", f"* spaCy: {s['heuristic_warning_counts']['spacy']}",
         f"* Raised for both: {s['heuristic_warning_counts']['shared_by_both']}", "",
         "## What this does and does not show",
         "* It shows how often each independent translator produced a technically usable candidate and how often they agree.",
         "* It does **not** show that either candidate is correct, nor that combining them improves quality: no ISL-fluent reviewer has judged them.",
         "* Fill in `candidates_side_by_side.csv` (reviewer columns) to establish that.", ""]
    return "\n".join(L)
