"""Plain data structures shared by all pipeline stages."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Piece:
    """A contiguous run of spoken text from one speaker, with the metadata that came with it."""

    text: str
    speaker_id: Optional[str]
    turn_index: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    cue_index: Optional[int] = None  # 1-based position of the cue in the file
    cue_id: Optional[str] = None  # identifier as written in the file (e.g. SRT cue number)
    raw_start: Optional[int] = None  # offsets into ParsedSource.raw_text (None for cue formats)
    raw_end: Optional[int] = None


@dataclass
class Edit:
    stage: str  # ingest | normalize | clean | segment
    operation: str
    original: str
    cleaned: str
    reason: str
    confidence: str = "high"  # high | medium | low
    review_required: bool = False
    chunk_id: Optional[str] = None
    source_start_offset: Optional[int] = None
    source_end_offset: Optional[int] = None
    edit_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Flag:
    code: str
    text: str
    start: int  # character offsets inside the chunk text
    end: int
    note: str
    review_required: bool = False
    skip_translation: bool = False  # the span is kept in the text but must be skipped by the translator

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    speaker_id: Optional[str]
    turn_index: int
    start_time: Optional[str]
    end_time: Optional[str]
    source_start_offset: Optional[int]
    source_end_offset: Optional[int]
    cue_indices: Optional[list[int]]
    cue_ids: Optional[list[str]]
    uncertainty_flags: list[Flag] = field(default_factory=list)
    edit_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker_id": self.speaker_id,
            "turn_index": self.turn_index,
            "source_start_offset": self.source_start_offset,
            "source_end_offset": self.source_end_offset,
            "cue_indices": self.cue_indices,
            "cue_ids": self.cue_ids,
            "skip_translation_spans": [
                {"start": f.start, "end": f.end, "text": f.text, "reason": f.code}
                for f in self.uncertainty_flags if f.skip_translation
            ],
            "uncertainty_flags": [f.to_dict() for f in self.uncertainty_flags],
            "edit_notes": list(self.edit_notes),
        }


@dataclass
class ParsedSource:
    source_id: str
    filename: str
    fmt: str  # txt | srt | vtt
    raw_text: str  # decoded text; the basis for source offsets
    encoding: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pieces: list[Piece] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    @property
    def timestamps_available(self) -> bool:
        return any(p.start_time is not None or p.end_time is not None for p in self.pieces)

    @property
    def speakers_available(self) -> bool:
        return any(p.speaker_id is not None for p in self.pieces)

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})
