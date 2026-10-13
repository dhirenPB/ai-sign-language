"""Diagnostic scoring of saved LLM results against provisional annotations (evaluation only).

Nothing here changes a gloss or runs during translation, and it is not a Layer 2 (rule-based) component. Every number is a
DIAGNOSTIC SIGNAL for where a reviewer should look:
* the annotations (evaluation/layer1_annotations.json) are engineering judgements, not ISL-reviewed;
* a lower count, a shorter gloss or fewer review flags is NOT evidence of a better translation;
* the detectors are word-list and string checks and will miss problems and raise false alarms.

Categories tracked separately (as requested for the Layer 1 evaluation):
  1 empty glosses for understandable sentences        2 handling of genuinely unintelligible input
  3 proper-name formatting errors / false positives    4 missing meaning-bearing markers
  5 pronouns not preserved                             6 unnecessary review flags
  7 unsupported labels and paraphrases                 8 inconsistent multiword notation
  9 English function-word leakage                      10 missing / duplicated / misaligned outputs
  11 token usage, latency, request count (only when the run recorded them)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

FUNCTION_TOKENS = {"THAT", "AND", "IT", "THE", "A", "AN", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "OF", "TO", "THERE", "WOULD"}
# labels that legitimately have no counterpart in the source words, so they are not reported as "unsupported"
STRUCTURAL = {"NOT", "NO", "NEVER", "NOTHING", "NONE", "IF", "UNLESS", "WHENEVER", "WHEN", "WHY", "WHAT", "WHERE", "WHO", "HOW",
              "FUTURE", "PAST", "PERSON", "I", "ME", "WE", "US", "YOU", "HE", "SHE", "IT", "THEY", "HER", "HIM", "HIS", "MY",
              "YOUR", "OUR", "THEIR", "THEM", "ONE", "TWO", "THREE", "FINISH", "WILL", "CAN", "MUST", "SAME"}
# a few irregular source forms, so their lemma labels are not reported as having no source counterpart
IRREGULAR = {"saw": "see", "seen": "see", "told": "tell", "bought": "buy", "said": "say", "says": "say", "kept": "keep",
             "taught": "teach", "left": "leave", "went": "go", "had": "have", "did": "do", "dried": "dry", "dries": "dry", "flows": "flow",
             "us": "we", "our": "we", "me": "i", "my": "i", "belongs": "belong", "does": "do", "leaves": "leave"}


def load_annotations(path: Optional[str | Path]) -> Optional[dict]:
    if not path or not Path(path).is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parts(tokens: list[str], include_fs: bool = False) -> set[str]:
    out: set[str] = set()
    for t in tokens:
        if t.startswith("FS:") and not include_fs:
            continue
        body = t.removeprefix("FS:")
        out.add(body)
        out.update(p for p in re.split(r"[_-]", body) if p)
    return out


def _source_words(text: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", text)}
    return words | {IRREGULAR[w] for w in words if w in IRREGULAR}


def _overlaps(piece: str, words: set[str]) -> bool:
    p = piece.lower()
    if len(p) < 3:
        return True
    return any(w == p or (len(w) >= 3 and (w.startswith(p[:4]) or p.startswith(w[:4]))) for w in words)


def _is_empty(row: dict) -> bool:
    return row["translation_status"] == "translated" and not (row["candidate_tokens"] or [])


def score_row(row: dict, ann: dict, run_labels: set[str]) -> list[str]:
    """Issue codes for one result row. Judges the model's own candidate, before any pipeline rule."""
    sid, issues = row["sentence_id"], []
    if row["translation_status"] != "translated":
        return ["no_result"] if row["translation_status"] in ("failed", "not_attempted") else []
    cand = row["candidate_tokens"] or []
    llm = row["llm"] or {}
    flagged = bool(llm.get("needs_human_review"))
    notes = (row["uncertainty_notes"] or "").strip()
    unintelligible = sid in ann.get("unintelligible", [])

    # 1 and 2: empty glosses and unintelligible input
    if not cand:
        if unintelligible:
            issues.append("unintelligible_empty_flagged_with_reason" if (flagged and notes) else "unintelligible_empty_without_reason")
        else:
            issues.append("empty_gloss_for_understandable_sentence")
            if not (flagged and notes):
                issues.append("empty_gloss_without_reason")
    elif unintelligible:
        issues.append("unintelligible_input_given_an_invented_gloss")
    if not cand:  # an empty gloss is counted once above; it is not also "missing" every name, marker and pronoun
        if sid in ann.get("expected_review", []) and not flagged:
            issues.append("expected_review_not_flagged")
        return issues

    # 3: names (formatting errors and false positives)
    fs_words = {t[3:] for t in cand if t.startswith("FS:")}
    ordinary = [t for t in cand if not t.startswith("FS:")]
    expected = ann.get("named_entities", {}).get(sid, [])
    for w in expected:
        if w in fs_words:
            continue
        in_label = [t for t in ordinary if w in re.split(r"[_-]", t)]
        if not in_label:
            issues.append("name_missing")
        elif any(sum(p in expected for p in re.split(r"[_-]", t)) >= 2 for t in in_label):
            issues.append("name_in_compound_label")
        else:
            issues.append("name_not_fingerspelled")
    for w in ann.get("not_names", {}).get(sid, []):
        if w in fs_words:
            issues.append("name_false_positive")
    if any("_" in t[3:] for t in cand if t.startswith("FS:")):
        issues.append("fs_token_joins_several_words")

    # 4 and 5: meaning-bearing markers and pronouns
    have = _parts(cand)
    for category, options in ann.get("expected_markers", {}).get(sid, {}).items():
        if not (have & set(options)):
            issues.append(f"marker_missing:{category}")
    for options in ann.get("pronouns_to_keep", {}).get(sid, []):
        if not (have & set(options)):
            issues.append("pronoun_not_preserved")

    # 6: review flags
    allowed = set(ann.get("expected_review", [])) | set(ann.get("review_optional", []))
    if flagged and sid not in allowed:
        issues.append("review_flag_possibly_unnecessary")
    if sid in ann.get("expected_review", []) and not flagged:
        issues.append("expected_review_not_flagged")

    # 7: unsupported labels and paraphrases
    allfs = _parts(cand, include_fs=True)
    for term in ann.get("terms_not_to_paraphrase", {}).get(sid, []):
        if term not in allfs:
            issues.append("culturally_specific_term_paraphrased_or_dropped")
    words = _source_words(row["llm_input_text"] or row["source_text"])
    for t in ordinary:
        if t in STRUCTURAL or not re.fullmatch(r"[A-Z_-]+", t):
            continue
        pieces = [p for p in re.split(r"[_-]", t) if p]
        if "-" in t or any(not _overlaps(p, words) and p not in STRUCTURAL for p in pieces):
            issues.append("label_without_source_counterpart:" + t)

    # 8: notation across the run
    for label in run_labels:
        a, _, b = label.partition("_")
        if "_" in b or not a or not b:
            continue
        src = row["llm_input_text"] or row["source_text"]
        if _overlaps(a, _source_words(src)) and re.search(rf"\b{re.escape(b[:3].lower())}", src.lower()):
            if label not in cand and (a in cand):
                issues.append("multiword_notation_inconsistent:" + label)
    for a, b in zip(cand, cand[1:]):
        if f"{a}_{b}" in run_labels:
            issues.append("multiword_notation_inconsistent:" + f"{a}_{b}")

    # 9: English function-word leakage
    leaked = [t for t in cand if t in FUNCTION_TOKENS]
    if leaked:
        issues.append("function_word_tokens:" + ",".join(leaked))

    # 10: alignment
    codes = {w["code"] for w in row["validation"]["warnings"]}
    for c in ("source_text_mismatch", "source_text_not_echoed", "duplicate_result_for_sentence", "stray_results_ignored"):
        if c in codes:
            issues.append("alignment:" + c)
    return list(dict.fromkeys(issues))  # each issue once per sentence


def score_rows(rows: list[dict], ann: Optional[dict], usage: Optional[dict] = None) -> dict[str, Any]:
    """Per-category counts (with the sentence ids behind each count) for one prompt version's results."""
    ids = [r["sentence_id"] for r in rows]
    summary: dict[str, Any] = {
        "sentences": len(rows),
        "annotations": "PROVISIONAL engineering annotations, not ISL-reviewed" if ann else "none supplied",
        "alignment": {
            "missing_or_failed_results": sum(r["translation_status"] != "translated" and r["translation_status"] != "skipped_no_translatable_text" for r in rows),
            "duplicate_ids": len(ids) - len(set(ids)),
        },
    }
    if usage is not None:
        summary["usage"] = usage
    if not ann:
        return summary
    run_labels = {t for r in rows for t in (r["candidate_tokens"] or []) if "_" in t and not t.startswith("FS:") and len(t.split("_")) == 2}
    per_row: dict[str, list[str]] = {}
    where: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        issues = score_row(r, ann, run_labels)
        per_row[r["sentence_id"]] = issues
        for code in dict.fromkeys(i.split(":")[0] for i in issues):
            where[code].append(r["sentence_id"])
    counts = Counter(i.split(":")[0] for issues in per_row.values() for i in issues)
    summary["issue_counts"] = dict(sorted(counts.items()))
    summary["issue_sentences"] = {k: v for k, v in sorted(where.items())}
    summary["details"] = {sid: [i for i in v if ":" in i] for sid, v in per_row.items() if any(":" in i for i in v)}
    summary["per_sentence_issues"] = per_row
    summary["categories"] = {
        "1_empty_gloss_for_understandable_sentence": counts.get("empty_gloss_for_understandable_sentence", 0),
        "2_unintelligible_handled_correctly": counts.get("unintelligible_empty_flagged_with_reason", 0),
        "2_unintelligible_handled_wrongly": counts.get("unintelligible_empty_without_reason", 0) + counts.get("unintelligible_input_given_an_invented_gloss", 0),
        "3_name_formatting_errors": sum(counts.get(k, 0) for k in ("name_missing", "name_in_compound_label", "name_not_fingerspelled", "fs_token_joins_several_words")),
        "3_name_false_positives": counts.get("name_false_positive", 0),
        "4_meaning_markers_missing": counts.get("marker_missing", 0),
        "5_pronouns_not_preserved": counts.get("pronoun_not_preserved", 0),
        "6_review_flags_possibly_unnecessary": counts.get("review_flag_possibly_unnecessary", 0),
        "6_expected_review_not_flagged": counts.get("expected_review_not_flagged", 0),
        "7_unsupported_labels_or_paraphrases": counts.get("culturally_specific_term_paraphrased_or_dropped", 0) + counts.get("label_without_source_counterpart", 0),
        "8_inconsistent_multiword_notation": counts.get("multiword_notation_inconsistent", 0),
        "9_function_word_tokens_present": counts.get("function_word_tokens", 0),
        "10_alignment_problems": counts.get("alignment", 0) + summary["alignment"]["missing_or_failed_results"] + summary["alignment"]["duplicate_ids"],
    }
    marker_by_type = Counter(i.split(":", 1)[1] for issues in per_row.values() for i in issues if i.startswith("marker_missing:"))
    summary["meaning_markers_missing_by_type"] = dict(marker_by_type)
    summary["model_flagged_review"] = sum(bool((r["llm"] or {}).get("needs_human_review")) for r in rows)
    summary["empty_glosses_total"] = sum(_is_empty(r) for r in rows)
    summary["gloss_tokens"] = sum(len(r["candidate_tokens"] or []) for r in rows)
    return summary
