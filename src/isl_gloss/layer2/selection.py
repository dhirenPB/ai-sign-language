"""Layer 2D: a conservative, configurable selection policy. The LLM is the PRIMARY candidate; spaCy is an independent ALTERNATIVE.

Principles
* Both candidates are always generated and kept. Neither is ever edited.
* Whenever the LLM candidate is technically usable it is carried forward as the primary candidate. That is a POLICY DEFAULT
  (`selection_basis = policy_default_primary`), not a finding that the LLM gloss is better than the spaCy gloss.
* spaCy replaces the LLM only if (a) the LLM technically failed AND the spaCy output passes a stricter usability check, or
  (b) the optional, OFF-by-default R-SEL-05 finds a concrete problem in the LLM candidate and none in spaCy.
* An empty LLM gloss is NOT enough on its own. If the LLM deliberately declined the sentence (empty gloss plus its own
  review flag, for example for garbled or ambiguous input), that is the model's judgement about the sentence; spaCy is then
  OFFERED as an alternative but not substituted (switch `fallback_on_llm_abstention` to change this).
* Length, token count, similarity and the number of English words never decide anything.
* Every decision records its rule (R-SEL-xx) and basis. `verification` is always
  "technical_checks_only_not_linguistically_verified".

Rules, in the order tried
  R-SEL-00  nothing to translate in the source                                  -> no_source_text
  R-SEL-03  LLM usable, spaCy usable and identical after normalisation          -> candidates_agree (LLM)
  R-SEL-05  OPTIONAL, OFF: LLM has a listed concrete warning, spaCy has none    -> selected_by_check (SPACY, review)
  R-SEL-06  LLM usable                                                          -> llm_primary (LLM); spaCy kept as alternative
  R-SEL-01  neither candidate technically usable                                -> both_invalid (review)
  R-SEL-08  LLM unusable; spaCy fails the fallback check (or fallback is off)   -> llm_failed_alternative_rejected (review)
  R-SEL-07  LLM abstained on purpose; spaCy passes the fallback check           -> llm_abstained_alternative_available (review)
  R-SEL-02  LLM technically failed; spaCy passes the fallback check             -> spacy_fallback_selected (SPACY, review)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .candidate_comparator import LLM, SPACY, CandidateCheck, Comparison

POLICY_VERSION = "2.0.0"
VERIFICATION = "technical_checks_only_not_linguistically_verified"
PRIMARY = LLM


@dataclass(frozen=True)
class SelectionPolicy:
    spacy_fallback_when_llm_fails: bool = True  # R-SEL-02: allow spaCy to stand in when the LLM technically failed
    fallback_on_llm_abstention: bool = False  # R-SEL-07 -> R-SEL-02: also substitute spaCy when the LLM declined on purpose (off)
    treat_identical_as_agreement: bool = True  # R-SEL-03
    allow_check_based_preference: bool = False  # R-SEL-05: OFF; no semantic check has been validated against ISL reviewers
    preference_codes: tuple = ("number_not_found",)  # the only LLM warnings R-SEL-05 may act on when switched on

    def to_dict(self) -> dict[str, Any]:
        return {"policy_version": POLICY_VERSION, "primary_candidate": PRIMARY, **{k: (list(v) if isinstance(v, tuple) else v)
                                                                                    for k, v in self.__dict__.items()}}


@dataclass
class Selection:
    selection_status: str
    selected_candidate: Optional[str]  # LLM | SPACY | None
    selection_reason: str
    needs_human_review: bool
    rule_id: str
    selection_basis: str = "none"  # policy_default_primary | agreement | technical_fallback | check_based | none
    alternative_usable_as_fallback: Optional[bool] = None  # did spaCy pass the stricter fallback check (None = not evaluated)
    alternative_check_reasons: list = field(default_factory=list)
    verification: str = VERIFICATION
    primary_candidate: str = PRIMARY
    alternative_candidate: str = SPACY

    def to_dict(self) -> dict[str, Any]:
        return {"primary_candidate": self.primary_candidate, "alternative_candidate": self.alternative_candidate,
                "selection_status": self.selection_status, "selected_candidate": self.selected_candidate,
                "selection_basis": self.selection_basis, "selection_reason": self.selection_reason,
                "needs_human_review": self.needs_human_review, "selection_rule": self.rule_id,
                "alternative_usable_as_fallback": self.alternative_usable_as_fallback,
                "alternative_check_reasons": self.alternative_check_reasons, "verification": self.verification}


def _problems(c: CandidateCheck) -> list[str]:
    return [w["code"] for w in c.warnings if w["severity"] == "warning"]


def _why_unusable(c: CandidateCheck) -> str:
    return "; ".join(i["code"] for i in c.structural_issues) or "unknown"


def _unresolved(c: CandidateCheck) -> bool:
    """Does this candidate still carry something a reviewer should look at?"""
    return bool(_problems(c)) or c.stated_review_flag


def alternative_check(spacy: CandidateCheck) -> tuple[bool, list[str]]:
    """Is the spaCy output good enough to stand in for a failed LLM candidate? Stricter than 'technically usable':
    it also rejects any candidate that raises a possible-omission warning or contains a punctuation-only token.
    Still a technical gate, not a linguistic judgement."""
    if not spacy.usable:
        return False, [f"not technically usable: {_why_unusable(spacy)}"]
    reasons = [f"raises {code}" for code in _problems(spacy)]
    if any(not re.search(r"[A-Za-z0-9]", t) for t in spacy.tokens):
        reasons.append("contains a punctuation-only token")
    return not reasons, reasons


def _alt_note(spacy: CandidateCheck) -> str:
    return "usable" if spacy.usable else f"not usable ({_why_unusable(spacy)})"


def select(cmp: Comparison, policy: SelectionPolicy = SelectionPolicy()) -> Selection:
    llm, spacy = cmp.llm, cmp.spacy
    codes_l = {i["code"] for i in llm.structural_issues}
    codes_s = {i["code"] for i in spacy.structural_issues}
    if not llm.usable and not spacy.usable and "no_translatable_text" in codes_l and "empty_source" in codes_s:
        return Selection("no_source_text", None, "Nothing translatable remained in this sentence after cleaning.", False, "R-SEL-00")

    if llm.usable:
        diff = cmp.difference
        if spacy.usable and policy.treat_identical_as_agreement and diff["identical_after_normalisation"]:
            shared = f" Both raise the same possible omission(s): {', '.join(cmp.shared_warnings)}." if cmp.shared_warnings else ""
            return Selection("candidates_agree", LLM, "The two independent candidates contain the same words after normalising case, FS: and separators. "
                             "Agreement is not proof of correctness." + shared,
                             _unresolved(llm) or _unresolved(spacy) or bool(cmp.shared_warnings), "R-SEL-03", "agreement")
        if policy.allow_check_based_preference and spacy.usable:
            hit = [c for c in _problems(llm) if c in policy.preference_codes]
            if hit and not _problems(spacy):
                return Selection("selected_by_check", SPACY, f"The LLM candidate raised {', '.join(hit)} and the spaCy candidate raised no warning. "
                                 "A heuristic check, not a linguistic judgement.", True, "R-SEL-05", "check_based")
        shared = f" Shared possible omission(s): {', '.join(cmp.shared_warnings)}." if cmp.shared_warnings else ""
        ok, reasons = alternative_check(spacy) if spacy.usable else (False, [f"not technically usable: {_why_unusable(spacy)}"])
        return Selection("llm_primary", LLM, "The LLM candidate is the primary candidate and is technically usable. It is carried forward by POLICY DEFAULT, "
                         f"not because it was judged better than the spaCy candidate, which is kept as an alternative ({_alt_note(spacy)})." + shared,
                         _unresolved(llm) or bool(cmp.shared_warnings), "R-SEL-06", "policy_default_primary", ok, reasons)

    if not spacy.usable:
        return Selection("both_invalid", None, f"Neither candidate is technically usable (LLM: {_why_unusable(llm)}; spaCy: {_why_unusable(spacy)}).",
                         True, "R-SEL-01", "none", False, [f"not technically usable: {_why_unusable(spacy)}"])

    ok, reasons = alternative_check(spacy)
    abstained = codes_l == {"empty_output"} and llm.stated_review_flag  # the model itself declined and asked for review
    if not ok or not policy.spacy_fallback_when_llm_fails:
        why = ("; ".join(reasons) if reasons else "spaCy fallback is switched off")
        return Selection("llm_failed_alternative_rejected", None, f"The LLM candidate is unusable ({_why_unusable(llm)}) and the spaCy candidate was not "
                         f"accepted as a stand-in: {why}. Neither is carried forward.", True, "R-SEL-08", "none", ok, reasons)
    if abstained and not policy.fallback_on_llm_abstention:
        return Selection("llm_abstained_alternative_available", None, "The LLM declined this sentence on purpose (empty gloss and its own review flag; its reason is in the notes). "
                         "The spaCy candidate passed the fallback check and is "
                         "OFFERED as an alternative, but is not substituted for the model's own judgement.", True, "R-SEL-07", "none", True, [])
    return Selection("spacy_fallback_selected", SPACY, f"The LLM candidate is unusable ({_why_unusable(llm)}); the spaCy candidate passed the fallback check "
                     "and is used as a provisional stand-in. Not a linguistic judgement.", True, "R-SEL-02", "technical_fallback", True, [])
