"""Data structures for the translation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# translation_status values
TRANSLATED = "translated"
SKIPPED = "skipped_no_translatable_text"  # everything in the chunk was marked skip/excluded
FAILED = "failed"
NOT_ATTEMPTED = "not_attempted"  # e.g. free-tier quota ran out before this chunk was reached


@dataclass
class SourceChunk:
    """One cleaned chunk (from structured/*.cleaned.json) or a bare sentence."""

    chunk_id: str
    text: str
    speaker_id: Optional[str] = None
    turn_index: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    skip_spans: list[dict] = field(default_factory=list)  # cleaner's skip_translation_spans
    excluded_spans: list[dict] = field(default_factory=list)  # [inaudible], [Music]... (from uncertainty_flags)
    source_metadata: dict = field(default_factory=dict)  # e.g. parent_chunk_id, split_method; copied to the result


@dataclass
class PreparedChunk:
    source: SourceChunk
    llm_text: str  # source text with skipped/excluded spans cut out; this is all the LLM ever sees
    skipped: list[dict]  # {start, end, text, reason}: positions refer to source.text
    unclear_audio_excluded: bool

    @property
    def translatable(self) -> bool:
        return bool(self.llm_text)


@dataclass
class Candidate:
    tokens: list[str]
    uncertain: bool = False  # True for status "uncertain" and "review_required"
    notes: str = ""
    status: str = "ok"  # ok | uncertain | review_required (translation confidence, NOT a technical failure)
    uncertain_parts: list[str] = field(default_factory=list)
    needs_human_review: bool = False  # the model's own flag; uncertainty is reported here, never inside the gloss
    echoed_source_text: Optional[str] = None  # the sentence text as the model copied it back (alignment check)

    def __post_init__(self) -> None:
        if self.status not in ("ok", "uncertain", "review_required"):
            self.status = "ok"
        if self.uncertain or self.status != "ok":
            self.needs_human_review = True
        if self.needs_human_review:
            self.uncertain = True
            if self.status == "ok":  # an empty gloss that needs review means "uninterpretable", otherwise "glossed, doubtful"
                self.status = "review_required" if not self.tokens else "uncertain"


@dataclass
class ChunkResult:
    prepared: PreparedChunk
    status: str = NOT_ATTEMPTED
    candidate: Optional[Candidate] = None
    final_tokens: list[str] = field(default_factory=list)
    applied_rules: list[dict] = field(default_factory=list)
    rule_pass_completed: bool = False
    validation: dict[str, Any] = field(default_factory=lambda: {"status": "failed", "warnings": []})
    error: Optional[dict] = None
    cache_hit: bool = False
    group_id: Optional[str] = None
    response_warnings: list[dict] = field(default_factory=list)  # problems with the model's reply for this sentence

    @property
    def needs_human_review(self) -> bool:
        """True if the model asked for review OR validation found anything beyond a clean pass. Everything is unreviewed
        until an ISL-fluent person has seen it; this flag says where to look first."""
        return bool(self.candidate and self.candidate.needs_human_review) or self.validation.get("status") != "pass"

    def to_dict(self, meta: dict[str, Any]) -> dict[str, Any]:
        src = self.prepared.source
        cand = self.candidate
        return {
            "sentence_id": src.chunk_id,
            "group_id": self.group_id,
            "source_text": src.text,
            "llm_input_text": self.prepared.llm_text,
            "skipped_spans": self.prepared.skipped,
            "translation_status": self.status,
            "candidate_gloss": " ".join(cand.tokens) if cand else None,
            "candidate_tokens": cand.tokens if cand else None,
            "final_gloss": " ".join(self.final_tokens) if self.status == TRANSLATED else None,
            "final_tokens": self.final_tokens if self.status == TRANSLATED else None,
            "applied_rules": self.applied_rules,
            "validation": self.validation,
            "review_status": "unreviewed",
            "uncertainty_notes": cand.notes if cand else None,
            "uncertain_parts": cand.uncertain_parts if cand else None,
            "needs_human_review": self.needs_human_review,
            "llm_status": cand.status if cand else None,
            "llm": {"status": cand.status, "needs_human_review": cand.needs_human_review, "uncertain": cand.uncertain,
                    "uncertain_parts": cand.uncertain_parts, "notes": cand.notes,
                    "echoed_source_text": cand.echoed_source_text} if cand else None,
            "error": self.error,
            "speaker_id": src.speaker_id,
            "start_timestamp": src.start_time,
            "end_timestamp": src.end_time,
            "turn_index": src.turn_index,
            "source_metadata": src.source_metadata,
            "metadata": {**meta, "rule_pass_completed": self.rule_pass_completed, "cache_hit": self.cache_hit},
        }
