"""Mandatory rule pass. Runs on every valid LLM candidate, in a fixed order, deterministically.

Design notes
* Rules are small, named, versioned, individually testable and individually switchable. Each states its input
  assumptions, precondition, effect and known limitation (see the Rule fields).
* The rules are our own. VirtualISLInterpreter (the reference) has no licence file, so no code was copied, and it
  transforms English parse trees, not gloss tokens, so its steps are NOT applied directly; its ideas (time first,
  question words last, article/copula dropping) are re-expressed here as narrow rules, the risky ones disabled.
* Lexical substitution is deliberately absent. Rules never invent or translate vocabulary.
* Every change is logged as before/after token lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

RULE_SET_VERSION = "0.1.0"

TOKEN_OK = re.compile(r"^(?:FS:)?[A-Z0-9]+(?:[_-][A-Z0-9]+)*$")
WH_TOKENS = {"WHAT", "WHERE", "WHEN", "WHY", "WHO", "WHOM", "WHICH", "HOW", "HOW_MANY", "HOW_MUCH"}
TIME_TOKENS = {"YESTERDAY", "TODAY", "TOMORROW", "TONIGHT", "NOW", "LATER", "EARLIER", "MORNING", "AFTERNOON", "EVENING"}
COPULA_TOKENS = {"BE", "IS", "ARE", "AM", "WAS", "WERE", "BEEN", "BEING"}
_WH_START = re.compile(r"^\s*(what|where|when|why|who|whom|whose|which|how)\b", re.I)


@dataclass
class RuleContext:
    source_text: str  # the cleaned English as sent to the LLM
    skipped_words: frozenset  # upper-cased words from spans the cleaner marked skip / excluded

    @property
    def is_question(self) -> bool:
        return self.source_text.rstrip().endswith("?") or bool(_WH_START.match(self.source_text))


@dataclass(frozen=True)
class Rule:
    rule_id: str
    version: str
    kind: str  # format | syntactic | lexical-removal
    description: str
    assumptions: str
    limitation: str
    enabled_by_default: bool
    transform: Callable[[list[str], RuleContext], Optional[list[str]]]  # None => precondition not met


# --------------------------------------------------------------------------------------------------------
def _normalise_token(tok: str) -> str:
    t = tok.strip().upper()
    t = re.sub(r"\s+", "_", t)
    prefix = "FS:" if t.startswith("FS:") else ""
    body = t[len(prefix):]
    body = re.sub(r"[^A-Z0-9_\-]", "", body)
    body = re.sub(r"_{2,}", "_", body).strip("_-")
    return prefix + body if body else ""


def _fmt01(tokens, ctx):
    out = [n for n in (_normalise_token(t) for t in tokens) if n]
    return out if out != tokens else None


def _art01(tokens, ctx):
    out = []
    for i, t in enumerate(tokens):
        if t in ("THE", "AN"):
            continue
        if t == "A":
            neighbours = [tokens[j] for j in (i - 1, i + 1) if 0 <= j < len(tokens)]
            if any(len(n) == 1 for n in neighbours):
                out.append(t)  # looks like part of a spelled-out sequence, keep
            continue
        out.append(t)
    return out if out != tokens else None


def _skip01(tokens, ctx):
    if not ctx.skipped_words:
        return None
    out = [t for t in tokens if t.removeprefix("FS:") not in ctx.skipped_words]
    return out if out != tokens else None


def _cop01(tokens, ctx):
    out = [t for t in tokens if t not in COPULA_TOKENS]
    return out if out != tokens and out else None


def _wh01(tokens, ctx):
    wh = [i for i, t in enumerate(tokens) if t in WH_TOKENS]
    if len(wh) != 1 or not ctx.is_question or wh[0] == len(tokens) - 1:
        return None
    return tokens[:wh[0]] + tokens[wh[0] + 1:] + [tokens[wh[0]]]


def _time01(tokens, ctx):
    idx = [i for i, t in enumerate(tokens) if t in TIME_TOKENS]
    if len(idx) != 1 or idx[0] == 0:
        return None
    return [tokens[idx[0]]] + tokens[:idx[0]] + tokens[idx[0] + 1:]


RULES: list[Rule] = [
    Rule("R-FMT-01", "1", "format",
         "Normalise token format: upper-case, strip punctuation/odd characters, spaces inside a token -> underscore, "
         "drop tokens that become empty.",
         "Tokens are short strings from the LLM.", "Cannot repair a token that was wrong in substance.", True, _fmt01),
    Rule("R-ART-01", "1", "lexical-removal",
         "Remove the articles THE and AN; remove A unless it sits next to a single-letter token (spelled-out letters).",
         "Articles are not signed in ISL. Gloss tokens are upper-case after R-FMT-01.",
         "A lone 'A' used as a letter or grade between words is removed.", True, _art01),
    Rule("R-SKIP-01", "1", "lexical-removal",
         "Remove tokens identical to a word the cleaner marked skip_translation (or an excluded marker), including "
         "the fingerspelled form FS:WORD.",
         "Skipped words were already cut from the LLM input, so a match means the model brought them back.",
         "Only exact whole-token matches are removed.", True, _skip01),
    Rule("R-COP-01", "1", "lexical-removal",
         "Remove copula tokens (BE IS ARE AM WAS WERE BEEN BEING).",
         "The copula is generally unsigned in ISL.",
         "Some constructions may need an overt sign; needs reviewer sign-off.", False, _cop01),
    Rule("R-ORD-WH-01", "1", "syntactic",
         "Move a single question word (WHAT, WHERE, ...) to the end when the source sentence is a question.",
         "Exactly one WH token; source ends with '?' or starts with a WH word.",
         "Placement of question words varies by signer and construction; needs reviewer sign-off.", False, _wh01),
    Rule("R-ORD-TIME-01", "1", "syntactic",
         "Move a single time-reference token (YESTERDAY, TODAY, ...) to the front.",
         "Exactly one token from a short time list, not already first.",
         "Misses multi-word time phrases; may split a phrase the LLM kept together; needs reviewer sign-off.",
         False, _time01),
]
RULES_BY_ID = {r.rule_id: r for r in RULES}


class RulePassError(RuntimeError):
    def __init__(self, rule_id: str, cause: Exception):
        super().__init__(f"rule {rule_id} raised {type(cause).__name__}: {cause}")
        self.rule_id = rule_id


class RuleEngine:
    def __init__(self, enable: tuple = (), disable: tuple = ()):
        unknown = [r for r in (*enable, *disable) if r not in RULES_BY_ID]
        if unknown:
            raise ValueError(f"Unknown rule ids: {unknown}. Known: {sorted(RULES_BY_ID)}")
        self.active = [r for r in RULES
                       if (r.enabled_by_default or r.rule_id in enable) and r.rule_id not in disable]

    @property
    def active_ids(self) -> list[str]:
        return [r.rule_id for r in self.active]

    def apply(self, tokens: list[str], ctx: RuleContext) -> tuple[list[str], list[dict]]:
        """Run every active rule once, in order. Returns (final_tokens, applied_rules)."""
        current, applied = list(tokens), []
        for rule in self.active:
            try:
                new = rule.transform(current, ctx)
            except Exception as exc:  # noqa: BLE001 - surfaced as RulePassError, candidate stays untouched upstream
                raise RulePassError(rule.rule_id, exc) from exc
            if new is not None and new != current:
                applied.append({
                    "rule_id": rule.rule_id, "rule_version": rule.version, "description": rule.description,
                    "status": "applied", "before": current, "after": new,
                })
                current = new
        return current, applied


def skipped_words_from_spans(spans: list[dict], remaining_text: str) -> frozenset:
    """Upper-cased words from skipped spans, minus any word that also occurs in the text the LLM was given
    (so a legitimate 'music' elsewhere in the sentence is never removed)."""
    kept = {w.upper() for w in re.findall(r"[^\W_]+", remaining_text)}
    words = {w.upper() for s in spans for w in re.findall(r"[^\W_]+", s["text"])}
    return frozenset(words - kept)
