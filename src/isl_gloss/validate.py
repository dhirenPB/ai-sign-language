"""Layer 3: deterministic diagnostics. These checks find likely problems; they cannot prove a gloss is correct ISL."""

from __future__ import annotations

import re
from typing import Optional

from .models import Candidate, PreparedChunk
from .rules import TOKEN_OK

_STOP = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being", "to", "of", "for", "from", "and", "or",
    "that", "this", "it", "its", "in", "on", "at", "as", "by", "with", "so", "if", "then", "do", "does", "did", "will",
    "would", "can", "could", "may", "might", "shall", "should", "have", "has", "had", "i", "you", "we", "they", "he", "she",
}
_NEGATION_SRC = re.compile(
    r"\b(not|no|never|none|nothing|nobody|nowhere|neither|nor|cannot|without)\b|n't\b|’t\b", re.I)
_NEGATION_TOK = re.compile(r"(^|_|-)(NOT|NO|NEVER|NONE|NOTHING|NOBODY|NOWHERE|NEITHER|NOR|CANNOT|WITHOUT|DONT|DO_NOT|CANT)($|_|-)")
# Capitalised words that are not names even when they appear mid-sentence (ASR often capitalises arbitrarily).
_NOT_NAMES = _STOP | {
    "hey", "hello", "hi", "okay", "ok", "yes", "so", "but", "now", "today", "yesterday", "tomorrow", "when", "what",
    "why", "how", "who", "where", "which", "there", "here", "then", "also", "because", "again", "please", "thank", "thanks",
    "i'm", "i'll", "i've", "i'd", "my", "your", "our", "their", "his", "her", "me", "us", "them", "not", "never", "just",
    "only", "even", "still", "well", "oh", "wow", "lord", "god", "though", "although", "unless", "whereas",
    "whenever", "while", "since", "once", "until", "if", "or", "and", "but", "yet", "however", "therefore",
}


def _w(severity: str, code: str, message: str) -> dict:
    return {"code": code, "severity": severity, "message": message}


def load_vocabulary(path: Optional[str]) -> Optional[frozenset]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return frozenset(line.strip().upper() for line in fh if line.strip() and not line.startswith("#"))


def validate_gloss(prepared: PreparedChunk, candidate: Candidate, final_tokens: list[str], applied_rules: list[dict],
                   rule_pass_completed: bool, vocabulary: Optional[frozenset] = None) -> dict:
    warnings: list[dict] = []
    text = prepared.llm_text

    if not rule_pass_completed:
        warnings.append(_w("failed", "rule_pass_failed",
                           "The rule pass did not complete; final_gloss is the unprocessed LLM candidate."))
    status = candidate.status if candidate.status != "ok" else ("uncertain" if candidate.uncertain else "ok")
    reason = candidate.notes or "no reason given"
    parts = f" Doubtful: {'; '.join(candidate.uncertain_parts)}." if candidate.uncertain_parts else ""
    if not final_tokens:
        if status == "review_required":  # translation uncertainty, reported honestly: not a technical failure
            warnings.append(_w("review", "llm_review_required", f"The model could not gloss this sentence and marked it review-required: {reason}"))
            if not (candidate.notes or "").strip():  # an empty gloss is only acceptable with a stated reason
                warnings.append(_w("failed", "empty_gloss_without_reason",
                                   "The gloss is empty and review-required but no reason was given in uncertainty_notes."))
        else:  # contract violation: an understandable sentence must never come back empty without that status
            warnings.append(_w("failed", "empty_gloss", "The final gloss is empty but the model did not mark the sentence review_required "
                                                         "(an understandable sentence must get a best-effort gloss)."))
    elif status == "review_required":
        warnings.append(_w("review", "llm_review_required", f"The model marked this sentence review-required: {reason}{parts}"))
    if status == "uncertain":
        warnings.append(_w("review", "llm_marked_uncertain", f"The LLM flagged this entry as uncertain: {reason}.{parts}"))
    if prepared.unclear_audio_excluded:
        warnings.append(_w("review", "unclear_audio_excluded",
                           "Unclear-audio markers were cut from the input; the gloss covers only the intelligible words."))
    if prepared.skipped and not prepared.unclear_audio_excluded:
        warnings.append(_w("info", "spans_skipped", f"{len(prepared.skipped)} span(s) were excluded from translation by design."))

    raw_bad = [t for t in candidate.tokens if not TOKEN_OK.match(t)]
    if raw_bad:
        warnings.append(_w("info", "candidate_format_issue",
                           f"{len(raw_bad)} candidate token(s) were not in gloss format (e.g. {raw_bad[0]!r}); the rule pass normalises these."))
    still_bad = [t for t in final_tokens if not TOKEN_OK.match(t)]
    if still_bad:
        warnings.append(_w("review", "invalid_token_format", f"Final tokens still not in gloss format: {still_bad[:5]}"))
    if any(" " in t for t in candidate.tokens) or any(re.search(r"[a-z]{3,} [a-z]{3,}", t) for t in candidate.tokens):
        warnings.append(_w("review", "prose_in_gloss", "A candidate token looks like a phrase or sentence rather than a gloss token."))

    content = [w for w in re.findall(r"[A-Za-z0-9']+", text) if w.lower() not in _STOP]
    if len(content) >= 4 and final_tokens:
        ratio = len(final_tokens) / len(content)
        if ratio < 0.3:
            warnings.append(_w("review", "possible_omission", f"Gloss has {len(final_tokens)} tokens for {len(content)} content words in the source."))
        elif ratio > 2.5:
            warnings.append(_w("review", "possible_addition", f"Gloss has {len(final_tokens)} tokens for {len(content)} content words in the source."))

    joined = " ".join(final_tokens)
    for num in re.findall(r"\d+", text):
        if num not in joined:
            warnings.append(_w("review", "number_not_preserved", f"The number {num} from the source does not appear in the gloss."))
    if _NEGATION_SRC.search(text) and not any(_NEGATION_TOK.search(t) for t in final_tokens):
        warnings.append(_w("review", "negation_may_be_lost", "The source contains a negation but no negation token is in the gloss."))

    words = re.findall(r"[A-Za-z][A-Za-z']*", text)
    names = {w.upper() for i, w in enumerate(words) if i > 0 and w[0].isupper() and w.lower() not in _NOT_NAMES and len(w) > 1}
    present = set()
    for t in final_tokens:  # FS:JONATHAN_YOUNG counts as containing both JONATHAN and YOUNG
        body = t.removeprefix("FS:")
        present.add(body)
        present.update(p for p in re.split(r"[_-]", body) if p)
    missing = sorted(n for n in names if n not in present)
    if missing:
        warnings.append(_w("review", "name_may_be_missing", f"Capitalised word(s) not found in the gloss: {', '.join(missing[:6])}. May be names or sign-label substitutions."))

    if vocabulary is not None:
        unknown = sorted({t for t in final_tokens if not t.startswith("FS:") and t not in vocabulary})
        if unknown:
            warnings.append(_w("review", "unknown_tokens", f"Tokens not in the supplied vocabulary: {', '.join(unknown[:8])}"))

    if any(a == b for a, b in zip(final_tokens, final_tokens[1:])):
        warnings.append(_w("info", "adjacent_duplicate_tokens", "Adjacent identical tokens (may be intended reduplication)."))
    if applied_rules:
        warnings.append(_w("info", "rules_changed_output", "Rule pass changed the LLM candidate: " + ", ".join(r["rule_id"] for r in applied_rules)))

    severities = {w["severity"] for w in warnings}
    status = "failed" if "failed" in severities else "review_required" if "review" in severities else "pass"
    return {"status": status, "warnings": warnings}


def status_from(warnings: list[dict]) -> str:
    severities = {w["severity"] for w in warnings}
    return "failed" if "failed" in severities else "review_required" if "review" in severities else "pass"


def validate_assembly(sentence_ids: list[str], groups: Optional[list[tuple[str, list[str]]]], result_ids: list[str],
                      times: Optional[list[Optional[str]]] = None) -> dict:
    """Run-level integrity checks: nothing lost, duplicated or reordered between the sentences that went in, the groups
    they were put in, and the results that came out. `groups` is [(group_id, [sentence_id, ...]), ...] or None.
    These checks cannot judge linguistic correctness."""
    errors: list[dict] = []
    warnings: list[dict] = []

    def err(code, msg):
        errors.append({"code": code, "message": msg})

    dup_in = sorted({s for s in sentence_ids if sentence_ids.count(s) > 1})
    if dup_in:
        err("duplicate_sentence_id", f"Sentence ids are not unique: {dup_in[:5]}")

    expected, got = set(sentence_ids), set(result_ids)
    if expected - got:
        err("missing_result", f"No result for sentence(s): {sorted(expected - got)[:5]}")
    if got - expected:
        err("unexpected_result", f"Result(s) for unknown sentence id(s): {sorted(got - expected)[:5]}")
    dup_out = sorted({s for s in result_ids if result_ids.count(s) > 1})
    if dup_out:
        err("duplicate_result", f"More than one result for: {dup_out[:5]}")
    if not (expected - got or got - expected or dup_out or dup_in) and result_ids != sentence_ids:
        err("order_changed", "Results are not in the original sentence order.")

    if groups is not None:
        gids = [g for g, _ in groups]
        if len(set(gids)) != len(gids):
            err("duplicate_group_id", "Group ids are not unique.")
        flat = [s for _, members in groups for s in members]
        if sorted(flat) != sorted(sentence_ids):
            lost = sorted(set(sentence_ids) - set(flat))
            extra = sorted(set(flat) - set(sentence_ids)) + sorted({s for s in flat if flat.count(s) > 1})
            err("group_coverage", f"Groups must cover every sentence exactly once. Lost: {lost[:5]} Extra/duplicated: {extra[:5]}")
        elif flat != sentence_ids:
            err("group_order", "Concatenated group members are not in transcript order.")
        if any(not members for _, members in groups):
            err("empty_group", "A group has no sentences.")

    if times:
        seen = [t for t in times if t]
        if any(b < a for a, b in zip(seen, seen[1:])):  # HH:MM:SS.mmm compares correctly as text
            warnings.append({"code": "timestamps_out_of_order", "message": "Sentence start times go backwards somewhere."})
    return {"passed": not errors, "errors": errors, "warnings": warnings}
