"""Stage 3: context-based grouping.

Baseline strategy (deliberately simple, as requested; no embeddings until an evaluation justifies them):

* Groups are runs of ADJACENT sentences in transcript order. Nothing is reordered, merged or dropped.
* A group ends at a hard boundary: a new speaker turn / paragraph (`turn_index` change) or a different known speaker,
  or a pause longer than `group_max_gap_seconds` when timestamps exist. Sentences before and after such a boundary are
  not assumed to belong together, so unrelated passages are not combined just to fill a group.
* Inside a turn a group ends when adding the next sentence would break a limit: sentences, characters or estimated
  tokens (characters / 4). The cut goes at the LATEST sentence boundary where the next sentence does not depend on the
  previous one (see sentences.dependency_reasons). Only if every boundary in the group is a dependent one is a cut forced
  (`forced_boundary`), and then the next group receives read-only context from before the cut.
* A sentence is never cut; one that alone exceeds a limit becomes its own group (`oversize`).
* Read-only context: a group whose first sentence depends on its predecessor gets up to `context_sentences_before`
  earlier sentences; a group that ends mid-thought (or whose successor depends on it) gets up to
  `context_sentences_after` following sentences. Context is never translated and never counted as the group's sentences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .config import TranslatorConfig
from .timeutil import seconds
from .sentences import Sentence


def estimate_tokens(text: str) -> int:
    """Rough token estimate (characters / 4, rounded up). Used only to bound request size."""
    return -(-len(text) // 4)


@dataclass
class Group:
    group_id: str
    sentence_ids: list[str]
    sentences: list[Sentence]
    context_before_ids: list[str] = field(default_factory=list)
    context_after_ids: list[str] = field(default_factory=list)
    context_text: str = ""
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    boundary_reason: str = "end"  # why this group ended
    forced_boundary: bool = False
    oversize: bool = False

    def to_dict(self, include_sentences: bool = True) -> dict[str, Any]:
        d = {
            "group_id": self.group_id,
            "sentence_ids": list(self.sentence_ids),
            "context_before_ids": list(self.context_before_ids),
            "context_after_ids": list(self.context_after_ids),
            "context_text": self.context_text,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "boundary_reason": self.boundary_reason,
            "forced_boundary": self.forced_boundary,
            "oversize": self.oversize,
        }
        if include_sentences:
            d["sentences"] = [s.to_dict() for s in self.sentences]
        return d


def _hard_break(prev: Sentence, cur: Sentence, cfg: TranslatorConfig) -> Optional[str]:
    if cfg.group_break_on_turn_change and prev.turn_index is not None and cur.turn_index is not None \
            and prev.turn_index != cur.turn_index:
        return "turn_change"
    if prev.speaker_id and cur.speaker_id and prev.speaker_id != cur.speaker_id:
        return "speaker_change"
    if cfg.group_max_gap_seconds is not None and prev.end_timestamp and cur.start_timestamp:
        if seconds(cur.start_timestamp) - seconds(prev.end_timestamp) > cfg.group_max_gap_seconds:
            return "time_gap"
    return None


def _limit_hit(sents: list[Sentence], cfg: TranslatorConfig) -> Optional[str]:
    text = " ".join(s.source_text for s in sents)
    if len(sents) > cfg.group_max_sentences:
        return "max_sentences"
    if len(text) > cfg.group_max_chars:
        return "max_chars"
    if estimate_tokens(text) > cfg.group_max_tokens:
        return "max_tokens"
    return None


def group_sentences(sentences: list[Sentence], cfg: TranslatorConfig, source_id: Optional[str] = None) -> list[Group]:
    if not sentences:
        return []
    source_id = source_id or sentences[0].source_metadata.get("source_id") or "doc"
    closed: list[tuple[list[Sentence], str, bool]] = []  # (members, reason, forced)
    cur: list[Sentence] = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if cur:
            reason = _hard_break(cur[-1], s, cfg)
            if reason:
                closed.append((cur, reason, False))
                cur = []
        if cur:
            limit = _limit_hit(cur + [s], cfg)
            if limit:
                cut = None
                for j in range(len(cur), 0, -1):  # latest clean boundary first
                    following = s if j == len(cur) else cur[j]
                    if not following.depends_on_previous:
                        cut = j
                        break
                forced = cut is None
                if forced:
                    cut = len(cur)
                closed.append((cur[:cut], limit, forced))
                cur = cur[cut:]
                continue  # re-test the same sentence against the carried-over tail
        cur.append(s)
        i += 1
    closed.append((cur, "end", False))

    index = {s.sentence_id: k for k, s in enumerate(sentences)}
    groups: list[Group] = []
    for n, (members, reason, forced) in enumerate(closed, 1):
        first, last = index[members[0].sentence_id], index[members[-1].sentence_id]
        before = after = []
        if first > 0 and members[0].depends_on_previous and cfg.context_sentences_before:
            before = sentences[max(0, first - cfg.context_sentences_before):first]
        nxt = sentences[last + 1] if last + 1 < len(sentences) else None
        if nxt is not None and cfg.context_sentences_after and (members[-1].incomplete or nxt.depends_on_previous):
            after = sentences[last + 1:last + 1 + cfg.context_sentences_after]
        text = " ".join(s.source_text for s in members)
        starts = [s.start_timestamp for s in members if s.start_timestamp]
        ends = [s.end_timestamp for s in members if s.end_timestamp]
        groups.append(Group(
            group_id=f"{source_id}-g{n:04d}",
            sentence_ids=[s.sentence_id for s in members],
            sentences=list(members),
            context_before_ids=[s.sentence_id for s in before],
            context_after_ids=[s.sentence_id for s in after],
            context_text=" ".join([s.source_text for s in before] + [s.source_text for s in members]
                                  + [s.source_text for s in after]),
            start_timestamp=starts[0] if starts else None,
            end_timestamp=ends[-1] if ends else None,
            boundary_reason=reason,
            forced_boundary=forced,
            oversize=len(members) == 1 and (len(text) > cfg.group_max_chars or estimate_tokens(text) > cfg.group_max_tokens),
        ))
    return groups


def group_stats(groups: list[Group]) -> dict[str, Any]:
    sizes = [len(g.sentences) for g in groups]
    reasons: dict[str, int] = {}
    for g in groups:
        reasons[g.boundary_reason] = reasons.get(g.boundary_reason, 0) + 1
    return {
        "groups": len(groups),
        "sentences": sum(sizes),
        "sentences_per_group_avg": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        "sentences_per_group_max": max(sizes) if sizes else 0,
        "boundary_reasons": reasons,
        "forced_boundaries": sum(g.forced_boundary for g in groups),
        "oversize_groups": sum(g.oversize for g in groups),
        "groups_with_context_before": sum(bool(g.context_before_ids) for g in groups),
        "groups_with_context_after": sum(bool(g.context_after_ids) for g in groups),
    }
