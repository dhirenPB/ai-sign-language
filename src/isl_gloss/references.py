"""Reviewed ISL reference support for the LLM layer: a documented data format, a validator and a small retrieval step.

Rules this module enforces:
* Only entries that name a reviewer, a review date and a source are ever accepted. An entry without them is REJECTED (reported,
  never sent to the model). Nothing is invented here, and the shipped file (references/reference_set.json) is an empty template.
* Retrieval is deterministic and small: only entries relevant to the target sentences are returned, capped per request. A whole
  dictionary is never sent.
* This is reference RETRIEVAL for the LLM. It is not a rule-based layer: it never edits, reorders or corrects a gloss, and it is
  not Layer 2. The model decides how (and whether) to use an entry.

Format: see docs/reference_set_format.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

FORMAT_VERSION = "1.0"
TOKEN_OK = re.compile(r"^(?:FS:)?[A-Z0-9]+(?:[_-][A-Z0-9]+)*$")
TAGS = ("condition", "negation", "question", "number", "time", "cause", "name", "pronoun", "comparison", "emphasis", "context")
_STOP = {"that", "this", "with", "from", "have", "will", "would", "there", "their", "they", "them", "what", "when", "were",
         "been", "into", "your", "about", "which", "while", "then", "than", "also", "some", "just", "very", "does"}

_FEATURES = {
    "condition": re.compile(r"\b(if|unless|whenever|in case|as long as|provided that)\b", re.I),
    "negation": re.compile(r"\b(not|no|never|none|nothing|nobody|nowhere|neither|nor|cannot|without)\b|n't\b", re.I),
    "cause": re.compile(r"\b(because|therefore|hence|due to|as a result|since)\b", re.I),
    "number": re.compile(r"\d"),
    "time": re.compile(r"\b(yesterday|today|tomorrow|tonight|now|later|before|after|last|next|every)\b", re.I),
    "pronoun": re.compile(r"\b(he|she|it|they|him|her|them|his|its|their|this|that)\b", re.I),
    "comparison": re.compile(r"\b(more|less|than|better|worse|most|least|bigger|smaller)\b", re.I),
    "emphasis": re.compile(r"\b(truly|really|very|extremely|absolutely|only|even|always|so)\b", re.I),
}
_QUESTION_START = re.compile(r"^\s*(what|where|when|why|who|whom|whose|which|how|did|do|does|is|are|was|were|can|could|will|would)\b", re.I)


@dataclass
class ReferenceSet:
    conventions: list[dict] = field(default_factory=list)
    vocabulary: list[dict] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # entries refused at load time, with the reason
    source_path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.conventions) + len(self.vocabulary) + len(self.pairs)

    # ---------------------------------------------------------------------------------------------------------
    def retrieve(self, texts: list[str], max_entries: int = 5) -> dict[str, list[dict]]:
        """Entries relevant to these target sentences. Notation conventions are short and global, so up to `max_entries`
        of them are always included; vocabulary needs an English phrase to occur in a sentence; sentence pairs are ranked by
        shared meaning features (condition, negation, ...) and shared content words and capped at `max_entries`."""
        joined = " ".join(texts)
        features = detect_features(texts)
        words = _content_words(joined)
        vocab = [e for e in sorted(self.vocabulary, key=lambda e: e["id"])
                 if re.search(rf"\b{re.escape(e['english'].lower())}\b", joined.lower())]
        scored = []
        for e in self.pairs:
            shared_tags = len(set(e.get("tags", [])) & features)
            shared_words = len(_content_words(e["english"]) & words)
            score = 2 * shared_tags + shared_words
            if score > 0:
                scored.append((-score, e["id"], e))
        pairs = [e for _, _, e in sorted(scored, key=lambda t: (t[0], t[1]))[:max_entries]]
        return {"conventions": sorted(self.conventions, key=lambda e: e["id"])[:max_entries],
                "vocabulary": vocab[:max_entries], "pairs": pairs}

    def format_block(self, retrieved: dict[str, list[dict]]) -> str:
        """The block appended to the user message ('' when nothing is relevant, so empty sets cost no tokens)."""
        if not any(retrieved.values()):
            return ""
        lines = ["Reference entries (reviewed by ISL-proficient people; the only verified ISL material for this request):"]
        if retrieved["conventions"]:
            lines.append("Notation conventions:")
            lines += [f"- {_one_line(e['statement'])}" for e in retrieved["conventions"]]
        if retrieved["vocabulary"]:
            lines.append("Vocabulary:")
            for e in retrieved["vocabulary"]:
                gloss = e["gloss"] if isinstance(e["gloss"], str) else " ".join(e["gloss"])
                note = f" ({_one_line(e['notes'])})" if e.get("notes") else ""
                lines.append(f'- "{_one_line(e["english"])}" -> {gloss}{note}')
        if retrieved["pairs"]:
            lines.append("Reviewed sentence examples:")
            for e in retrieved["pairs"]:
                note = f" ({_one_line(e['notes'])})" if e.get("notes") else ""
                lines.append(f'- "{_one_line(e["english"])}" -> {json.dumps(e["gloss"])}{note}')
        return "\n".join(lines)

    def retrieved_ids(self, retrieved: dict[str, list[dict]]) -> list[str]:
        return [e["id"] for group in retrieved.values() for e in group]


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def detect_features(texts: list[str]) -> set[str]:
    joined = " ".join(texts)
    out = {name for name, rx in _FEATURES.items() if rx.search(joined)}
    if any(t.strip().endswith("?") or _QUESTION_START.match(t) for t in texts):
        out.add("question")
    if any(re.search(r"(?<=[a-z,] )[A-Z][a-z]+", t) for t in texts):
        out.add("name")
    if len(texts) > 1:
        out.add("context")
    return out


# ---------------------------------------------------------------------------------------------------------------------
def _check_common(entry: dict, kind: str, seen: set[str]) -> Optional[str]:
    if not isinstance(entry, dict):
        return "entry is not an object"
    eid = entry.get("id")
    if not isinstance(eid, str) or not eid.strip():
        return "missing id"
    if eid in seen:
        return f"duplicate id {eid!r}"
    for key in ("reviewed_by", "reviewed_on", "source"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            return f"missing {key} (only reviewed entries with a named reviewer, a review date and a source are accepted)"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["reviewed_on"].strip()):
        return "reviewed_on must be a date YYYY-MM-DD"
    if kind == "convention" and not (isinstance(entry.get("statement"), str) and entry["statement"].strip()):
        return "convention needs a statement"
    if kind in ("vocabulary", "pair") and not (isinstance(entry.get("english"), str) and entry["english"].strip()):
        return "missing english"
    if kind in ("vocabulary", "pair"):
        gloss = entry.get("gloss")
        tokens = [gloss] if isinstance(gloss, str) else gloss
        if kind == "pair" and not isinstance(gloss, list):
            return "a sentence pair needs gloss as a list of tokens"
        if not tokens or not all(isinstance(t, str) and TOKEN_OK.match(t) for t in tokens):
            return "gloss must be non-empty and use UPPERCASE tokens (A-Z, 0-9, underscore; FS: prefix allowed)"
    if kind == "pair":
        bad = [t for t in entry.get("tags", []) if t not in TAGS]
        if bad:
            return f"unknown tag(s) {bad}; allowed: {list(TAGS)}"
    return None


def load_references(path: Optional[str | Path]) -> ReferenceSet:
    """Load and validate a reference file. Invalid or unreviewed entries are rejected, not repaired, and never retrievable."""
    if not path:
        return ReferenceSet()
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{p}: format_version must be {FORMAT_VERSION!r}")
    rs = ReferenceSet(source_path=str(p))
    seen: set[str] = set()
    for key, kind, target in (("notation_conventions", "convention", rs.conventions), ("vocabulary", "vocabulary", rs.vocabulary),
                              ("sentence_pairs", "pair", rs.pairs)):
        for entry in data.get(key, []):
            problem = _check_common(entry, kind, seen)
            if problem:
                rs.rejected.append({"section": key, "id": entry.get("id") if isinstance(entry, dict) else None, "reason": problem})
                continue
            seen.add(entry["id"])
            target.append(entry)
    return rs


def describe(rs: ReferenceSet) -> dict[str, Any]:
    return {"source": rs.source_path, "accepted": {"notation_conventions": len(rs.conventions), "vocabulary": len(rs.vocabulary),
                                                   "sentence_pairs": len(rs.pairs)},
            "rejected": rs.rejected}
