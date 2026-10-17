"""Layer 2C: compare the LLM candidate (A) and the spaCy candidate (B) for one sentence.

What this module does: structural validity checks, conservative "possible omission" warnings, notation warnings, and a
descriptive difference between the two glosses.

What it does NOT do:
* It never edits, re-orders, re-cases or "repairs" either candidate.
* Structural checks can reject a candidate as unusable; they cannot show that a usable one is linguistically correct.
* Semantic warnings are heuristics on English words and gloss words. They can miss real problems and raise false alarms:
  ISL may express negation, questions or conditions non-manually or by sentence structure, so a missing lexical marker is a
  reason to LOOK, not an error. Counts of warnings are never a quality score.
* `FS:NAME` is the project's notation, not a universal ISL convention; no word is converted into a fingerspelled name here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from ..validate import _NOT_NAMES

LLM, SPACY = "LLM", "SPACY"
TOKEN_OK = re.compile(r"^(?:FS:)?[A-Z0-9]+(?:[_-][A-Z0-9]+)*$")  # the project's Layer 1 token format

_NEG_SRC = re.compile(r"\b(not|no|never|none|nothing|nobody|nowhere|neither|nor|cannot|without)\b|n['’]t\b", re.I)
_NEG_TOK = {"NOT", "NO", "NEVER", "NONE", "NOTHING", "NOBODY", "NOWHERE", "NEITHER", "NOR", "CANNOT", "CANT", "DONT", "WITHOUT",
            "UNLESS", "N'T", "N’T"}
_COND_SRC = re.compile(r"\b(if|unless|whenever|in case|as long as|provided that)\b", re.I)
_COND_TOK = {"IF", "UNLESS", "WHENEVER", "WHEN", "EVERYTIME", "EVERY", "PROVIDED", "SUPPOSE", "CASE"}
_WH_SRC = re.compile(r"\b(what|where|when|why|who|whom|whose|which|how)\b", re.I)
_WH_TOK = {"WHAT", "WHERE", "WHEN", "WHY", "WHO", "WHOM", "WHOSE", "WHICH", "HOW"}
_CAUSE_SRC = re.compile(r"\b(because|therefore|hence|thus|due to|as a result|so that)\b", re.I)
_CAUSE_TOK = {"BECAUSE", "REASON", "WHY", "CAUSE", "THEREFORE", "HENCE", "THUS", "SO", "RESULT", "DUE", "SINCE"}
_MONTHS = "january february march april may june july august september october november december".split()
_DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
_TIME_SRC = re.compile(r"\b(yesterday|today|tomorrow|tonight|" + "|".join(_MONTHS[:4] + _MONTHS[5:] + _DAYS) + r")\b", re.I)
_QUESTION_START = re.compile(r"^\s*(what|where|when|why|who|whom|whose|which|how|did|do|does|is|are|was|were|can|could|will|would)\b", re.I)
_COMMON_CAPITALISED = {"i", "he", "she", "it", "they", "we", "you", "hello", "hi", "hey", "okay", "ok", "yes", "no", "but", "and",
                       "so", "if", "the", "this", "that", "these", "those", "then", "now", "today", "yesterday", "tomorrow",
                       "tonight", "january", "february", "march", "april", "june", "july", "august", "september", "october",
                       "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
_TITLES = {"professor", "prof", "doctor", "dr", "mr", "mrs", "ms", "miss", "sir", "madam", "lord", "king", "queen"}


# --- normalisation (used only to COMPARE; candidates themselves are never changed) -----------------------------------
def words_of(gloss: Optional[str]) -> list[str]:
    """Upper-case words of a gloss: FS: prefix removed, '_' and '-' and spaces treated as separators."""
    if not gloss:
        return []
    out = []
    for tok in gloss.split():
        out += [w for w in re.split(r"[_\-]", tok.removeprefix("FS:").removeprefix("fs:").upper()) if w]
    return out


def _warning(code: str, feature: str, message: str, severity: str = "warning") -> dict[str, str]:
    return {"code": code, "feature": feature, "severity": severity, "message": message}


@dataclass
class CandidateCheck:
    translator: str  # LLM | SPACY
    gloss: Optional[str]
    tokens: list
    translator_status: str  # the translator's own status, as reported
    structural_issues: list = field(default_factory=list)  # each makes the candidate unusable
    warnings: list = field(default_factory=list)  # possible problems; never make a candidate unusable on their own
    usable: bool = False
    stated_review_flag: bool = False  # the translator itself asked for human review (LLM only)

    def to_dict(self) -> dict[str, Any]:
        return {"translator": self.translator, "gloss": self.gloss, "translator_status": self.translator_status, "usable": self.usable,
                "structural_issues": self.structural_issues, "warnings": self.warnings, "stated_review_flag": self.stated_review_flag}


# --- structural checks -----------------------------------------------------------------------------------------------
def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _bad_tokens(tokens: Any) -> bool:
    return not isinstance(tokens, list) or any(not isinstance(t, str) or not t.strip() for t in tokens)


def check_llm(row: dict, expected_id: str, source_text: str, field_name: str = "candidate_tokens") -> CandidateCheck:
    """Candidate A from a saved Layer 1 result row. `candidate_tokens` is the model's own gloss; `final_tokens` (after the
    existing small rule pass) can be chosen with field_name."""
    status = row.get("translation_status") or "missing"
    tokens = row.get(field_name)
    gloss = row.get("candidate_gloss") if field_name == "candidate_tokens" else row.get("final_gloss")
    llm = row.get("llm") or {}
    chk = CandidateCheck(LLM, gloss, tokens if isinstance(tokens, list) else [], status, stated_review_flag=bool(llm.get("needs_human_review")))
    issues = chk.structural_issues
    if not row:
        issues.append(_issue("missing_output", "No Layer 1 result exists for this sentence."))
    elif row.get("sentence_id") != expected_id:
        issues.append(_issue("sentence_id_mismatch", f"Layer 1 row is for {row.get('sentence_id')!r}, expected {expected_id!r}."))
    if row and status == "skipped_no_translatable_text":
        issues.append(_issue("no_translatable_text", "Layer 1 had nothing to translate for this sentence."))
    elif row and status != "translated":
        err = row.get("error") or {}
        issues.append(_issue("translator_failed", f"Layer 1 status {status}: {err.get('kind', '')} {err.get('message', '')}".strip()))
    elif row:
        if tokens is None or _bad_tokens(tokens):
            issues.append(_issue("malformed_output", "candidate tokens are missing or are not a list of non-empty strings."))
        elif not tokens:
            reason = (row.get("uncertainty_notes") or "").strip()
            issues.append(_issue("empty_output", "The LLM returned an empty gloss" + (f" with the stated reason: {reason}" if reason else " and gave no reason") + "."))
        echoed = llm.get("echoed_source_text")
        if echoed is not None and " ".join(echoed.split()) != " ".join(source_text.split()):
            issues.append(_issue("source_text_mismatch", "The source text the LLM echoed differs from the sentence it was given; the gloss may belong to another sentence."))
        elif echoed is None and row.get("metadata", {}).get("prompt_version") not in (None, "5.0"):
            chk.warnings.append(_warning("source_text_not_echoed", "alignment", "The LLM did not echo the sentence, so alignment could not be double-checked."))
        if (row.get("validation") or {}).get("status") == "failed":
            codes = [w.get("code") for w in row["validation"].get("warnings", []) if w.get("severity") == "failed"]
            issues.append(_issue("layer1_validation_failed", f"Layer 1 technical validation failed: {', '.join(map(str, codes))}"))
    chk.usable = not issues
    if chk.usable:
        chk.warnings += semantic_warnings(source_text, " ".join(chk.tokens)) + _llm_notation_warnings(chk.tokens)
    return chk


def check_spacy(result: Any, expected_id: str, source_text: str) -> CandidateCheck:
    """Candidate B from a SpacyResult (or its dict)."""
    d = result.to_dict() if hasattr(result, "to_dict") else (result or {})
    status = d.get("spacy_status") or "missing"
    tokens = d.get("spacy_tokens")
    chk = CandidateCheck(SPACY, d.get("spacy_gloss"), tokens if isinstance(tokens, list) else [], status)
    issues = chk.structural_issues
    if not d:
        issues.append(_issue("missing_output", "No spaCy result exists for this sentence."))
    elif d.get("sentence_id") != expected_id:
        issues.append(_issue("sentence_id_mismatch", f"spaCy result is for {d.get('sentence_id')!r}, expected {expected_id!r}."))
    elif status == "unavailable":
        issues.append(_issue("translator_unavailable", (d.get("spacy_error") or {}).get("message", "spaCy dependencies are missing.")))
    elif status == "error":
        err = d.get("spacy_error") or {}
        issues.append(_issue("translator_failed", f"{err.get('kind', '')}: {err.get('message', '')}".strip(": ")))
    elif status == "empty":
        err = d.get("spacy_error") or {}
        issues.append(_issue("empty_source" if err.get("kind") == "empty_source" else "empty_output", err.get("message", "spaCy returned no tokens.")))
    elif status != "ok":
        issues.append(_issue("malformed_output", f"unknown spaCy status {status!r}"))
    elif _bad_tokens(tokens) or not (d.get("spacy_gloss") or "").strip():
        issues.append(_issue("malformed_output", "spaCy tokens are missing or are not a list of non-empty strings."))
    if d and d.get("source_text") is not None and " ".join(str(d["source_text"]).split()) != " ".join(source_text.split()):
        issues.append(_issue("source_text_mismatch", "The text spaCy translated differs from the sentence under comparison."))
    chk.usable = not issues
    if chk.usable:
        chk.warnings += semantic_warnings(source_text, " ".join(chk.tokens))
        if any(" " in t for t in chk.tokens):
            chk.warnings.append(_warning("multiword_token", "notation", "A token is a merged phrase containing spaces (reference noun-chunk "
                                         "formatting); downstream code that splits on spaces will see more tokens.", "info"))
    return chk


# --- conservative semantic checks ------------------------------------------------------------------------------------
def _names(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'’]*", text)
    seen, out = set(), []
    for i, w in enumerate(words):
        base = re.sub(r"['’]s$", "", w)
        low = base.lower()
        if i == 0 or not base[0].isupper() or len(base) < 2 or low in _COMMON_CAPITALISED or low in _NOT_NAMES or low in _TITLES:
            continue
        if low not in seen:
            seen.add(low)
            out.append(base)
    return out


def semantic_warnings(source_text: str, gloss: str) -> list[dict]:
    """Possible omissions, from the English source and the gloss words. Warnings only; the gloss is never changed."""
    have = set(words_of(gloss))
    joined = re.sub(r"[^a-z0-9 ]", "", " ".join(words_of(gloss)).lower())
    out: list[dict] = []
    if _NEG_SRC.search(source_text) and not (have & _NEG_TOK or any(w.startswith("NON") for w in have)):
        out.append(_warning("possible_negation_not_represented", "negation",
                            "The source has a negation but no negation word appears in the gloss (ISL may mark it non-manually)."))
    if _COND_SRC.search(source_text) and not (have & _COND_TOK):
        out.append(_warning("possible_condition_not_represented", "condition",
                            "The source has a conditional word but no conditional marker appears in the gloss."))
    is_question = source_text.strip().endswith("?") or bool(_QUESTION_START.match(source_text))
    if is_question and _WH_SRC.search(source_text) and not (have & _WH_TOK):
        out.append(_warning("possible_question_word_not_represented", "question", "The source is a wh-question but no question word appears in the gloss."))
    elif is_question and not _WH_SRC.search(source_text):
        out.append(_warning("yes_no_question_marking_not_checkable", "question",
                            "Yes/no questions are usually marked non-manually in ISL; this check cannot see that.", "info"))
    if _CAUSE_SRC.search(source_text) and not (have & _CAUSE_TOK):
        out.append(_warning("possible_cause_marker_not_represented", "cause", "The source states a cause or result but no cause marker appears in the gloss."))
    for m in dict.fromkeys(x.lower() for x in _TIME_SRC.findall(source_text)):
        if m.upper() not in have:
            out.append(_warning("possible_time_expression_not_represented", "time", f"The source mentions {m!r} but it does not appear in the gloss."))
    for num in dict.fromkeys(re.findall(r"\d+(?:[.,]\d+)?", source_text)):
        if num not in gloss:
            out.append(_warning("number_not_found", "number", f"The source contains the number {num} but it does not appear in the gloss."))
    for name in _names(source_text):
        if re.sub(r"[^a-z0-9]", "", name.lower()) not in joined.replace(" ", "") :
            out.append(_warning("possible_name_not_found", "name", f"The capitalised word {name!r} may be a name but does not appear in the gloss."))
    return out


def _llm_notation_warnings(tokens: list[str]) -> list[dict]:
    out = []
    for t in tokens:
        if not TOKEN_OK.match(t):
            out.append(_warning("token_not_in_project_format", "notation", f"{t!r} is not UPPERCASE letters/digits with '_' or '-' (optional FS: prefix)."))
        elif t.startswith("FS:") and "_" in t[3:]:
            out.append(_warning("fs_token_joins_several_words", "notation", f"{t!r} joins several words after FS:; the project convention is one FS: token per word."))
    return out


# --- descriptive difference ------------------------------------------------------------------------------------------
def describe_difference(llm: CandidateCheck, spacy: CandidateCheck) -> dict[str, Any]:
    """How the two glosses differ, as facts. This is NOT a similarity score or a quality judgement."""
    a, b = words_of(llm.gloss) if llm.usable else [], words_of(spacy.gloss) if spacy.usable else []
    ca, cb = Counter(a), Counter(b)
    notation = []
    if llm.usable and spacy.usable:
        if any(t.startswith("FS:") for t in llm.tokens):
            notation.append("LLM uses FS: fingerspelling tokens; the spaCy reference has no such convention")
        if llm.gloss != llm.gloss.upper() or spacy.gloss != spacy.gloss.lower():
            notation.append("LLM gloss is UPPERCASE; the spaCy reference output is lower-case lemmas")
        if any(" " in t for t in spacy.tokens):
            notation.append("spaCy merges noun phrases and entities into single tokens containing spaces")
        if any(("_" in t or "-" in t) for t in llm.tokens):
            notation.append("LLM joins multiword concepts with '_' or '-'")
    both = llm.usable and spacy.usable
    return {
        "comparable": both,
        "identical_as_given": bool(both and llm.gloss == spacy.gloss),
        "identical_after_normalisation": bool(both and a == b),
        "same_words_different_order": bool(both and a != b and ca == cb),
        "llm_word_count": len(a) if llm.usable else None, "spacy_word_count": len(b) if spacy.usable else None,
        "only_in_llm": sorted((ca - cb).elements()) if both else [], "only_in_spacy": sorted((cb - ca).elements()) if both else [],
        "notation_differences": notation,
        "normalisation": "upper-case words; FS: prefix removed; '_', '-' and spaces are separators",
        "note": "descriptive only: not a similarity score and not evidence that either gloss is better",
    }


def shared_warning_codes(llm: CandidateCheck, spacy: CandidateCheck) -> list[str]:
    """Semantic warnings raised for BOTH candidates: a possible shared omission, which agreement between them does not resolve."""
    sem = lambda c: {w["code"] for w in c.warnings if w["severity"] == "warning" and w["feature"] not in ("notation", "alignment")}
    return sorted(sem(llm) & sem(spacy)) if llm.usable and spacy.usable else []


@dataclass
class Comparison:
    sentence_id: str
    source_text: str
    llm: CandidateCheck
    spacy: CandidateCheck
    difference: dict
    shared_warnings: list


def compare_candidates(sentence_id: str, source_text: str, llm_row: Optional[dict], spacy_result: Any,
                       llm_field: str = "candidate_tokens") -> Comparison:
    llm = check_llm(llm_row or {}, sentence_id, source_text, llm_field)
    spacy = check_spacy(spacy_result, sentence_id, source_text)
    return Comparison(sentence_id, source_text, llm, spacy, describe_difference(llm, spacy), shared_warning_codes(llm, spacy))
