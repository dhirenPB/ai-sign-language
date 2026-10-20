"""What can actually be shown as a sign, and the synonym table that points unsigned words at signed ones.

`Vocabulary.available(word)` answers "would the resolver find a sign for this word?" using the same rules as yap-render `lib/sign/resolver.ts`:
a Yap avatar word, or a dictionary key with exactly one usable sign, or one with several signs and a fixed default. Synonyms are only ever a
fallback for words that are NOT available, and every target is validated here against the real vocabulary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .parse import normalize_label

SENSE_STATUSES = ("verified", "source_label", "needs_review")
USABLE_STATUSES = ("verified", "source_label")
CONFIDENCE_LEVELS = ("exact", "close")


def parse_yap_tokens(ts_file: Path) -> Set[str]:
    """Read SUPPORTED_TOKENS from yap-render `lib/islVocabulary.ts` (the avatar words that have an animation)."""
    text = ts_file.read_text(encoding="utf-8")
    body = text.split("SUPPORTED_TOKENS: readonly string[] = [", 1)[1].split("]", 1)[0]
    return {normalize_label(t) for t in re.findall(r"'([A-Z_]+)'", body)}


class Vocabulary:
    """Availability of signs for normalised words."""

    def __init__(self, yap: Iterable[str], assets: Iterable[Dict[str, object]], select: Optional[Dict[str, str]] = None):
        self.yap: Set[str] = {normalize_label(t) for t in yap}
        self.select = dict(select or {})
        self._senses: Dict[str, Dict[str, Dict[str, object]]] = {}
        for a in assets:
            if a.get("status") not in SENSE_STATUSES:
                continue
            keys = {a["label_norm"]} | {normalize_label(al["text"]) for al in a.get("aliases", []) if al.get("verified")}
            for k in keys:
                if k:
                    self._senses.setdefault(k, {})[a["asset_id"]] = a

    @classmethod
    def from_files(cls, yap_ts: Optional[Path], manifest: Path, mappings: Optional[Path] = None) -> "Vocabulary":
        assets = json.loads(manifest.read_text(encoding="utf-8"))["assets"]
        select = json.loads(mappings.read_text(encoding="utf-8")).get("select", {}) if mappings and mappings.is_file() else {}
        return cls(parse_yap_tokens(yap_ts) if yap_ts else set(), assets, select)

    def source(self, word: str) -> Optional[str]:
        """'yap' (avatar animation), 'dictionary' (recorded video) or None (no sign)."""
        key = normalize_label(word)
        if key in self.yap:
            return "yap"
        senses = self._senses.get(key)
        if not senses:
            return None
        if len(senses) == 1:
            return "dictionary" if next(iter(senses.values()))["status"] in USABLE_STATUSES else None
        chosen = self.select.get(key)
        return "dictionary" if chosen in senses and senses[chosen]["status"] in SENSE_STATUSES else None

    def available(self, word: str) -> bool:
        return self.source(word) is not None


def validate_synonyms(table: Dict[str, object], vocab: Vocabulary) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    """Validate the curated table against the real vocabulary. Returns ({normalised from: {to, confidence, note}}, problems).

    An entry is dropped (and reported) when it is malformed, maps a word to itself, repeats a source word, has no target that is actually available,
    or when its source word already has a sign (the word's own sign always wins, so such an entry could only cause confusion later).
    Targets that are not available are removed from an entry; targets that are themselves synonym sources are rejected (no chains).
    """
    problems: List[str] = []
    entries = table.get("entries")
    if not isinstance(entries, list):
        return {}, ["table has no 'entries' list"]
    sources = {normalize_label(str(e.get("from", ""))) for e in entries if isinstance(e, dict)}
    out: Dict[str, Dict[str, object]] = {}
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            problems.append(f"entry {i}: not an object")
            continue
        src = normalize_label(str(e.get("from", "")))
        conf = e.get("confidence")
        label = f"entry {i} ({e.get('from')!r})"
        if not src:
            problems.append(f"{label}: empty 'from'")
            continue
        if conf not in CONFIDENCE_LEVELS:
            problems.append(f"{label}: confidence must be one of {CONFIDENCE_LEVELS}, got {conf!r}")
            continue
        if src in out:
            problems.append(f"{label}: duplicate source word")
            continue
        if vocab.available(src):
            problems.append(f"{label}: the word already has a sign ({vocab.source(src)}), so the synonym is unnecessary; dropped")
            continue
        targets: List[str] = []
        for t in e.get("to") or []:
            tk = normalize_label(str(t))
            if tk == src:
                problems.append(f"{label}: maps to itself")
            elif tk in sources:
                problems.append(f"{label}: target {t!r} is itself a synonym source (chains are not allowed)")
            elif not vocab.available(tk):
                problems.append(f"{label}: target {t!r} has no sign in the Yap avatar vocabulary or the dictionary; removed")
            elif tk not in targets:
                targets.append(tk)
        if not targets:
            problems.append(f"{label}: no usable target; dropped")
            continue
        out[src] = {"to": targets, "confidence": conf, "note": e.get("note", "")}
    return out, problems


def load_synonym_table(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
