"""Mechanical parsing of ISLRTC dictionary file names.

The Hugging Face re-encode ships no label table (no Parquet/CSV), so the file name is the only label the dataset gives. Nothing here is
inferred semantically: each rule is a literal pattern seen in the file names, and anything that does not fit a rule is flagged for review
instead of being guessed. The parsed label is therefore "the dataset's own label", not a verified gloss.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

EXPLANATION_PAREN = re.compile(r"^(sign and )?expla[a-z]*(\s*\d+)?$|^example(\s*\d+)?$|^meaning$", re.I)
SIGN_VARIANT_PAREN = re.compile(r"^sign\s*(\d*)$", re.I)
NUMBER_PAREN = re.compile(r"^\d+$")
ABBREVIATION_PAREN = re.compile(r"^[A-Z0-9&.\- ]{2,12}$")
LANGUAGE_TAG = re.compile(r"\s*-\s*English(?:\s+(\d+))?$", re.I)
MAX_LABEL_WORDS = 8
MAX_LIST_PART_WORDS = 4


def normalize_label(value: str) -> str:
    """Must stay identical to `normalizeLabel` in yap-render `lib/sign/normalize.ts`; tests compare both on shared cases."""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"^fs:", "", text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^\w\s']|_", " ", text)
    text = text.replace("'", "")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Alias:
    text: str
    source: str
    verified: bool = True


@dataclass
class ParsedLabel:
    stem: str
    label: str
    label_norm: str
    variant: Optional[str] = None
    qualifier: Optional[str] = None
    entry_type: str = "term"  # term | explanation | phrase
    language_tag: Optional[str] = None
    aliases: List[Alias] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip(" -,")


def parse_stem(stem: str, folder: str = "") -> ParsedLabel:
    """Parse one file stem (file name without `.mp4`) found in `folder` ('A'..'Z' or 'Numbers')."""
    text = stem
    variant: Optional[str] = None
    qualifiers: List[str] = []
    aliases: List[Alias] = []
    flags: List[str] = []
    entry_type = "term"

    for group in re.findall(r"\(([^)]*)\)", text):
        inner = _clean(group)
        sign = SIGN_VARIANT_PAREN.match(inner)
        if sign:
            variant = f"sign_{sign.group(1)}" if sign.group(1) else "sign"
        elif NUMBER_PAREN.match(inner):
            variant = inner
        elif EXPLANATION_PAREN.match(inner):
            entry_type = "explanation"
        elif ABBREVIATION_PAREN.match(group.strip()) and re.search(r"[A-Za-z]", group):
            # The dictionary itself prints the abbreviation next to the term, so this is source-provided, not invented.
            aliases.append(Alias(group.strip(), "filename_abbreviation"))
        elif inner:
            qualifiers.append(inner)
    raw_text = re.sub(r"\([^)]*\)", " ", text)  # underscores still intact: "1_One" is a pair, "120 Volt" is not
    text = _clean(raw_text)

    numeric_alias = None
    if folder == "Numbers":
        pair = re.match(r"^\s*(\d+)_(\D.*)$", raw_text)
        if pair:
            numeric_alias, text = pair.group(1), _clean(pair.group(2))
            aliases.append(Alias(numeric_alias, "numbers_filename_pair"))

    language_tag = None
    tag = LANGUAGE_TAG.search(text)
    if tag:
        language_tag = "English"
        if tag.group(1) and variant is None:
            variant = tag.group(1)  # "Hair Pins - English_1": the trailing number follows the tag, so it numbers the clip
        text = _clean(LANGUAGE_TAG.sub("", text))

    if re.search(r"\s\d+$", text) and not numeric_alias:
        flags.append("trailing_number")  # "Lean_1", "Form_16": a variant or part of the term? Not decidable from the name.

    words = text.split(" ") if text else []
    if len(words) > MAX_LABEL_WORDS:
        flags.append("long_label")
        entry_type = entry_type if entry_type == "explanation" else "phrase"

    parts = re.split(r"\s+or\s+|\s*,\s*", text, flags=re.I) if re.search(r"\s+or\s+|,", text, re.I) else []
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1 and all(len(p.split()) <= MAX_LIST_PART_WORDS for p in parts) and entry_type != "phrase":
        for part in parts:
            aliases.append(Alias(part, "filename_or_list"))
        flags.append("synonym_list")

    qualifier = " / ".join(qualifiers) if qualifiers else None
    if qualifier:
        flags.append("sense_qualifier")
    if language_tag:
        flags.append("english_tag")
    if folder and folder != "Numbers" and text and text[0].upper() != folder.upper():
        flags.append("folder_initial_mismatch")
    label_norm = normalize_label(text)
    if len(label_norm) < 2 or not re.search(r"[^\W_]", label_norm):
        flags.append("weak_label")

    return ParsedLabel(stem=stem, label=text, label_norm=label_norm, variant=variant, qualifier=qualifier, entry_type=entry_type,
                       language_tag=language_tag, aliases=aliases, flags=flags)
