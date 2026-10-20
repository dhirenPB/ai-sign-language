"""Make a gloss more signable without changing what it says: a deterministic step AFTER Phase 1 (the LLM gloss is never edited in place).

For every gloss word that has no sign (neither a Yap avatar animation nor a dictionary video) it tries, in this order:
  1. a synonym from the curated, validated table (`exact` by default; `close` only when asked for),
  2. fingerspelling, if the word is consistently capitalised in the source text (a name or proper term with no sign),
and then merges an immediate repeat that the substitution itself created (HELLO HI -> HELLO HELLO -> HELLO).
A word's own sign always wins, `FS:` tokens are never touched, and every change is recorded with its reason so a reviewer can see and undo it.
Nothing here is reviewed by an ISL signer: a synonym records that two English words mean (nearly) the same, not that ISL signs them alike.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from islrtc_index.parse import normalize_label
from islrtc_index.vocab import Vocabulary

LEVELS = ("off", "exact", "close")
MIN_NAME_LENGTH = 4
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# Ordinary English words that auto-captions sometimes capitalise ("Done", "Though", "Again") must never be mistaken for names.
_FALLBACK_STOP_WORDS = frozenset("""about above after again all also although always among and another any are around because been before being both but could did does done during each either else enough even ever every first for from further had has have her here hers him his how however into its itself just last less many may might more most much must neither never next none nor not now off often once only other ought our ours out over own perhaps rather said same shall she should since some still such than that the their them then there these they this those though through thus too under until upon very was were what when where whether which while who whom whose why will with within without would yet you your yours""".split())


def common_english_words() -> frozenset:
    """spaCy's English stop-word list when spaCy is installed (it needs no model), otherwise a built-in list."""
    try:
        from spacy.lang.en.stop_words import STOP_WORDS

        return frozenset(STOP_WORDS) | _FALLBACK_STOP_WORDS
    except Exception:  # spaCy is an optional dependency of this project
        return _FALLBACK_STOP_WORDS


@dataclass
class Change:
    index: int
    kind: str  # synonym | spell | merge
    before: str
    after: str
    reason: str
    confidence: Optional[str] = None


@dataclass
class Normalized:
    tokens: List[str]
    changes: List[Change] = field(default_factory=list)


def token_form(key: str) -> str:
    """'a lot' -> 'A_LOT' (the gloss token form)."""
    return re.sub(r"\s+", "_", key.strip()).upper()


def consistently_capitalised(texts: Iterable[str]) -> Set[str]:
    """Words capitalised everywhere they occur in the source, at least once in the middle of a sentence (so not merely sentence-initial).

    Auto-captions capitalise unreliably, so a word that appears in lower case anywhere is never included.
    """
    seen: Dict[str, Dict[str, int]] = {}
    for text in texts:
        for m in WORD.finditer(text):
            word = m.group(0)
            rec = seen.setdefault(word.lower(), {"lower": 0, "cap_mid": 0})
            before = text[: m.start()].rstrip()
            at_sentence_start = not before or before[-1] in ".?!"
            if word == word.lower():
                rec["lower"] += 1
            elif not at_sentence_start:
                rec["cap_mid"] += 1
    common = common_english_words()
    return {w for w, r in seen.items() if r["lower"] == 0 and r["cap_mid"] >= 1 and len(w) >= MIN_NAME_LENGTH and w not in common}


def normalize_tokens(tokens: List[str], vocab: Vocabulary, synonyms: Dict[str, Dict[str, object]], level: str = "exact",
                     capitalised: Optional[Set[str]] = None) -> Normalized:
    """Apply the rules to one gloss. `synonyms` is the validated table (key -> {to, confidence}); `capitalised` enables the spelling rule."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    out: List[str] = []
    substituted: List[bool] = []  # parallel to `out`: was this token produced by a synonym?
    changes: List[Change] = []
    for i, tok in enumerate(tokens):
        key = normalize_label(tok)
        if tok.upper().startswith("FS:") or not key or vocab.available(key):
            out.append(tok)
            substituted.append(False)
            continue
        entry = synonyms.get(key) if level != "off" else None
        target = None
        if entry and (entry["confidence"] == "exact" or level == "close"):
            target = next((t for t in entry["to"] if vocab.available(t)), None)
        if target:
            out.append(token_form(target))
            substituted.append(True)
            changes.append(Change(i, "synonym", tok, token_form(target), f"no sign for {tok}; {target} means (nearly) the same", entry["confidence"]))
        elif capitalised and key in capitalised and re.fullmatch(r"[a-z]+", key):
            out.append(f"FS:{key.upper()}")
            substituted.append(False)
            changes.append(Change(i, "spell", tok, f"FS:{key.upper()}", "no sign, and capitalised everywhere it appears in the source (a name or proper term): spelled"))
        else:
            out.append(tok)
            substituted.append(False)

    # Merge an immediate repeat only when a substitution created it; never touch a repeat the speaker really made.
    kept: List[str] = []
    kept_substituted: List[bool] = []
    for i, tok in enumerate(out):
        if kept and kept[-1] == tok and substituted[i] != kept_substituted[-1]:
            changes.append(Change(i, "merge", tok, "", "repeat created by a synonym substitution removed"))
            continue
        kept.append(tok)
        kept_substituted.append(substituted[i])
    return Normalized(kept, changes)


def coverage(tokens: List[str], vocab: Vocabulary) -> Dict[str, int]:
    """How many tokens can be shown: avatar, dictionary video, spelled (FS:), or no sign."""
    c = {"avatar": 0, "dictionary": 0, "spelled": 0, "no_sign": 0, "total": 0}
    for t in tokens:
        c["total"] += 1
        if t.upper().startswith("FS:"):
            c["spelled"] += 1
            continue
        src = vocab.source(t)
        c["avatar" if src == "yap" else "dictionary" if src == "dictionary" else "no_sign"] += 1
    return c
