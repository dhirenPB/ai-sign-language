"""Input stage: read cleaned chunks and cut out everything the translator must not see.

Spans marked skip_translation by the cleaner (non-English words, known mis-recognised terms, expression cues like
"hmm") and excluded markers ([inaudible], [Music]) are removed from the text sent to the LLM. They are recorded in the
result with their original positions, so nothing disappears silently and the later facial-expression stage can find them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import TranslatorConfig
from .models import PreparedChunk, SourceChunk


def chunks_from_doc(doc: dict, cfg: TranslatorConfig) -> list[SourceChunk]:
    """The cleaner's chunks as SourceChunks (old direct mode, and the baseline of the comparison)."""
    exclude = set(cfg.exclude_flag_codes)
    chunks = []
    for c in doc["chunks"]:
        excluded = [
            {"start": f["start"], "end": f["end"], "text": f["text"], "reason": f["code"]}
            for f in c.get("uncertainty_flags", []) if f["code"] in exclude
        ]
        chunks.append(SourceChunk(
            chunk_id=c["chunk_id"], text=c["text"], speaker_id=c.get("speaker_id"), turn_index=c.get("turn_index"),
            start_time=c.get("start_time"), end_time=c.get("end_time"),
            skip_spans=list(c.get("skip_translation_spans", [])), excluded_spans=excluded,
            source_metadata={"parent_chunk_id": c["chunk_id"], "source_id": doc.get("source_id")},
        ))
    return chunks


def load_cleaned_json(path: str | Path, cfg: TranslatorConfig) -> tuple[dict, list[SourceChunk]]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if "chunks" not in doc or "source_id" not in doc:
        raise ValueError(f"{path} is not a cleaned-transcript JSON (missing 'chunks'/'source_id').")
    return doc, chunks_from_doc(doc, cfg)


def chunks_from_sentences(sentences) -> list[SourceChunk]:
    """Stage-2 sentences as SourceChunks: sentence_id becomes the id that travels through LLM, rules and validation."""
    out = []
    for s in sentences:
        out.append(SourceChunk(
            chunk_id=s.sentence_id, text=s.source_text, speaker_id=s.speaker_id, turn_index=s.turn_index,
            start_time=s.start_timestamp, end_time=s.end_timestamp, skip_spans=list(s.skip_spans),
            excluded_spans=list(s.excluded_spans),
            source_metadata={**s.source_metadata, "incomplete": s.incomplete,
                             "depends_on_previous": s.depends_on_previous, "dependency_reasons": list(s.dependency_reasons)},
        ))
    return out


def _merge_spans(spans: list[dict], text: str) -> list[dict]:
    merged: list[dict] = []
    for s in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if not (0 <= s["start"] < s["end"] <= len(text)) or text[s["start"]:s["end"]] != s["text"]:
            raise ValueError(f"span {s['start']}-{s['end']} ({s['text']!r}) does not match the chunk text")
        if merged and s["start"] <= merged[-1]["end"]:
            last = merged[-1]
            last["end"] = max(last["end"], s["end"])
            last["text"] = text[last["start"]:last["end"]]
            if s["reason"] not in last["reason"].split("+"):
                last["reason"] += "+" + s["reason"]
        else:
            merged.append(dict(s))
    return merged


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"^[,;:\s]+", "", s)
    s = re.sub(r"[,;:]+(?=\s*[.!?])", "", s)
    return s if re.search(r"\w", s) else ""


def prepare_chunk(chunk: SourceChunk) -> PreparedChunk:
    spans = _merge_spans(chunk.skip_spans + chunk.excluded_spans, chunk.text)
    pieces, cursor = [], 0
    for s in spans:
        pieces.append(chunk.text[cursor:s["start"]])
        pieces.append(" ")  # keep neighbouring words apart
        cursor = s["end"]
    pieces.append(chunk.text[cursor:])
    return PreparedChunk(
        source=chunk,
        llm_text=_tidy("".join(pieces)),
        skipped=spans,
        unclear_audio_excluded=any("unclear_audio" in s["reason"].split("+") for s in spans),
    )
