"""Orchestration of the sequential hybrid pipeline:

    sentences -> groups -> LLM (one request per group) -> MANDATORY rule pass -> validation -> traceable results

There is no fallback chain. If the LLM step fails the sentence is reported as failed; no other translator is substituted,
and no sentence is ever dropped: every input sentence has exactly one result, in the original order.
The rule pass runs on every valid candidate, including candidates that validation later flags.

A "unit" is what one request carries: a Stage-3 group (with its read-only context), or, in the old direct mode, a fixed
window of sentences. The model sees bare numbered sentences; the number it echoes maps each gloss back to a sentence id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import RESULT_SCHEMA_VERSION
from .config import TranslatorConfig
from .grouping import Group
from .llm import (AuthError, LLMClient, LLMError, MissingApiKey, ModelUnavailable, QuotaExhausted,
                  TruncatedResponse, parse_results_json)
from .models import FAILED, NOT_ATTEMPTED, SKIPPED, TRANSLATED, Candidate, ChunkResult, PreparedChunk, SourceChunk
from .prompt import STATUS_VALUES, get_prompt
from .references import ReferenceSet, load_references
from .rules import RULE_SET_VERSION, RuleContext, RuleEngine, RulePassError, skipped_words_from_spans
from .store import ResultStore, cache_key
from .validate import status_from, validate_assembly, validate_gloss

_STOP_RUN = (QuotaExhausted, AuthError, MissingApiKey, ModelUnavailable)  # nothing further can succeed this run
MAX_SPLIT_DEPTH = 3
MAX_CONSECUTIVE_FAILED_UNITS = 3


@dataclass
class Unit:
    """One request. `members` are result indexes to translate; `before`/`after` are read-only context indexes."""

    members: list[int]
    before: list[int] = field(default_factory=list)
    after: list[int] = field(default_factory=list)
    group_id: Optional[str] = None


@dataclass
class RunOutcome:
    results: list[ChunkResult]
    complete: bool
    stop_reason: Optional[str] = None
    requests_made: int = 0
    dry_run_requests: list[dict] = field(default_factory=list)
    assembly: dict = field(default_factory=dict)

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
            key = f"validation_{r.validation['status']}" if r.status in (TRANSLATED, SKIPPED) else None
            if key:
                out[key] = out.get(key, 0) + 1
        out["cache_hits"] = sum(r.cache_hit for r in self.results)
        return out


class Translator:
    def __init__(self, cfg: TranslatorConfig, client: Optional[LLMClient], store: Optional[ResultStore] = None,
                 vocabulary: Optional[frozenset] = None, log: Callable[[str], None] = lambda m: None):
        self.cfg, self.client = cfg, client
        self.store = store if store is not None else ResultStore(None)  # NB: an empty ResultStore is falsy
        self.vocabulary, self.log = vocabulary, log
        self.spec = get_prompt(cfg.prompt_version)
        # generation settings that change what the model returns; part of the cache key so configurations never mix
        self.settings = f"effort={cfg.reasoning_effort};temp={cfg.temperature};think={cfg.thinking_budget};maxout={cfg.max_output_tokens}"
        self.system_prompt = self.spec.build_system(vocabulary)  # a supplied vocabulary becomes part of the prompt
        self.engine = RuleEngine(cfg.enable_rules, cfg.disable_rules)
        # what fresh requests cost (cache hits cost nothing); recorded so runs can be compared on usage and latency
        self.stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                      "latency_seconds": 0.0, "empty_gloss_retries": 0, "empty_gloss_recovered": 0, "reference_entries_sent": 0}
        # reviewed reference entries are retrieved per request, and only for prompts that know how to use them
        self.references: ReferenceSet = (load_references(cfg.references_file) if cfg.references_file and self.spec.supports_references
                                         else ReferenceSet())
        self.retry_empty = bool(cfg.empty_gloss_retry and self.spec.supports_references)

    # ------------------------------------------------------------------------------------------------------
    def metadata(self) -> dict:
        meta = self._base_metadata()
        if self.spec.supports_references:  # extra settings exist only for prompts that use them; older metadata is unchanged
            meta["llm_config"].update({
                "empty_gloss_retry": self.retry_empty,
                "references_file": self.cfg.references_file,
                "reference_entries_loaded": len(self.references)})
        return meta

    def _base_metadata(self) -> dict:
        return {
            "model": f"{self.cfg.provider}/{self.cfg.model}",
            "prompt_version": self.spec.version,
            "llm_config": {  # recorded so outputs from different configurations can be compared fairly
                "provider": self.cfg.provider, "model": self.cfg.model, "reasoning_effort": self.cfg.reasoning_effort,
                "temperature": self.cfg.temperature, "max_output_tokens": self.cfg.max_output_tokens,
                "prompt_version": self.spec.version,
                "thinking_budget": self.cfg.thinking_budget if self.cfg.provider == "gemini" else None,
            },
            "rule_set_version": RULE_SET_VERSION,
            "active_rules": self.engine.active_ids,
            "result_schema_version": RESULT_SCHEMA_VERSION,
        }

    def run(self, chunks: list[SourceChunk], dry_run: bool = False, groups: Optional[list[Group]] = None) -> RunOutcome:
        """Translate `chunks` (sentences). With `groups`, one request per group; without, fixed windows (old mode)."""
        results = self._prepare(chunks)
        index = {r.prepared.source.chunk_id: i for i, r in enumerate(results)}
        if len(index) != len(results):
            raise ValueError("sentence ids must be unique")
        units = self._units_from_groups(results, index, groups) if groups is not None else self._units_windows(results)

        outcome = RunOutcome(results=results, complete=True)
        failed_in_a_row = 0
        for n, unit in enumerate(units, 1):
            user = self._user_text(results, unit)
            if dry_run:
                outcome.dry_run_requests.append({
                    "group_id": unit.group_id, "sentences": len(unit.members), "model": self.cfg.model,
                    "prompt_version": self.spec.version, "user_payload": user})
                continue
            cached = [self.store.get(cache_key(self.cfg.model, self.system_prompt, user, label, self.settings))
                      for label in self._labels(results, unit.members)]
            if all(c is not None for c in cached):
                for idx, cand in zip(unit.members, cached):
                    results[idx].candidate, results[idx].cache_hit = cand, True
                continue
            self.log(f"request {n}/{len(units)}: {len(unit.members)} sentence(s)" + (f" [{unit.group_id}]" if unit.group_id else ""))
            try:
                got = self._translate_unit(results, unit, depth=0)
            except _STOP_RUN as exc:
                outcome.complete, outcome.stop_reason = False, f"{exc.kind}: {exc}"
                break
            ok = 0
            for idx in unit.members:
                if isinstance(got.get(idx), Candidate):
                    results[idx].candidate = got[idx]
                    ok += 1
                else:
                    results[idx].status = FAILED
                    results[idx].error = got.get(idx) or {"kind": "no_result", "message": "No result was returned for this sentence."}
            failed_in_a_row = 0 if ok else failed_in_a_row + 1
            if failed_in_a_row >= MAX_CONSECUTIVE_FAILED_UNITS:
                outcome.complete, outcome.stop_reason = False, "too_many_failed_requests"
                break

        for r in results:  # rules + validation are cheap and deterministic: always (re)run
            if r.candidate is not None and r.status != FAILED:
                self._finish(r)
        outcome.requests_made = getattr(self.client, "requests_made", 0)
        if not dry_run and any(r.status == NOT_ATTEMPTED and r.prepared.translatable for r in results):
            outcome.complete = False
            outcome.stop_reason = outcome.stop_reason or "sentences_not_attempted"
        outcome.assembly = validate_assembly(
            [r.prepared.source.chunk_id for r in results],
            [(g.group_id, list(g.sentence_ids)) for g in groups] if groups is not None else None,
            [r.prepared.source.chunk_id for r in results],
            [r.prepared.source.start_time for r in results],
        )
        return outcome

    # ------------------------------------------------------------------------------------------------------
    def _prepare(self, chunks: list[SourceChunk]) -> list[ChunkResult]:
        from .prepare import prepare_chunk

        results: list[ChunkResult] = []
        for c in chunks:
            try:
                prep = prepare_chunk(c)
            except ValueError as exc:
                prep = PreparedChunk(source=c, llm_text="", skipped=[], unclear_audio_excluded=False)
                results.append(ChunkResult(prepared=prep, status=FAILED, error={"kind": "bad_skip_span", "message": str(exc)},
                                           validation={"status": "failed", "warnings": [
                                               {"code": "bad_skip_span", "severity": "failed", "message": str(exc)}]}))
                continue
            res = ChunkResult(prepared=prep)
            if not prep.translatable:
                res.status = SKIPPED
                warns = [{"code": "entire_chunk_skipped", "severity": "info",
                          "message": "Everything in this sentence was marked skip/excluded; nothing to translate."}]
                if prep.unclear_audio_excluded:
                    warns.append({"code": "unclear_audio_excluded", "severity": "review",
                                  "message": "The sentence consists only of unclear-audio markers."})
                res.validation = {"status": "review_required" if prep.unclear_audio_excluded else "pass", "warnings": warns}
                res.rule_pass_completed = True
            results.append(res)
        return results

    def _units_from_groups(self, results, index, groups: list[Group]) -> list[Unit]:
        seen: set[str] = set()
        units = []
        for g in groups:
            for sid in g.sentence_ids + g.context_before_ids + g.context_after_ids:
                if sid not in index:
                    raise ValueError(f"group {g.group_id} refers to unknown sentence id {sid}")
            for sid in g.sentence_ids:
                results[index[sid]].group_id = g.group_id
                seen.add(sid)
            members = [index[s] for s in g.sentence_ids if results[index[s]].prepared.translatable
                       and results[index[s]].status == NOT_ATTEMPTED]
            if members:
                units.append(Unit(members=members, before=[index[s] for s in g.context_before_ids],
                                  after=[index[s] for s in g.context_after_ids], group_id=g.group_id))
        # sentences missing from the groups must still appear in the output (they will fail assembly validation)
        return units

    def _units_windows(self, results) -> list[Unit]:
        """Old direct mode: fixed windows over the translatable sentences, no grouping, preceding sentences as context."""
        idxs = [i for i, r in enumerate(results) if r.status == NOT_ATTEMPTED and r.prepared.translatable]
        units, cur, chars = [], [], 0
        for i in idxs:
            n = len(results[i].prepared.llm_text)
            if cur and (len(cur) >= self.cfg.batch_size or chars + n > self.cfg.max_batch_chars):
                units.append(cur)
                cur, chars = [], 0
            cur.append(i)
            chars += n
        if cur:
            units.append(cur)
        out = []
        for members in units:
            before, j = [], members[0] - 1
            while j >= 0 and len(before) < self.cfg.context_chunks:
                if results[j].prepared.llm_text:
                    before.append(j)
                j -= 1
            out.append(Unit(members=members, before=list(reversed(before))))
        return out

    def _record_usage(self, resp, seconds: float) -> None:
        usage = getattr(resp, "usage", None) or {}
        prompt = usage.get("prompt_tokens", usage.get("promptTokenCount", 0)) or 0
        completion = usage.get("completion_tokens", usage.get("candidatesTokenCount", 0)) or 0
        total = usage.get("total_tokens", usage.get("totalTokenCount", prompt + completion)) or 0
        self.stats["llm_calls"] += 1
        self.stats["prompt_tokens"] += prompt
        self.stats["completion_tokens"] += completion
        self.stats["total_tokens"] += total
        self.stats["latency_seconds"] = round(self.stats["latency_seconds"] + seconds, 3)

    def _labels(self, results, members: list[int]) -> list:
        """What identifies each target sentence to the model: its real sentence id (prompt 6.0) or a number 1..N (5.0)."""
        if self.spec.id_mode:
            return [results[i].prepared.source.chunk_id for i in members]
        return list(range(1, len(members) + 1))

    @staticmethod
    def _texts(results, idxs: list[int]) -> list[str]:
        return [results[i].prepared.llm_text for i in idxs if results[i].prepared.llm_text]

    def _reference(self, results, unit: Unit) -> tuple[str, list[str]]:
        """(block, entry ids) of the reviewed reference entries relevant to this unit's target sentences ('' when none)."""
        if not len(self.references):
            return "", []
        retrieved = self.references.retrieve(self._texts(results, unit.members), self.cfg.reference_max_entries)
        return self.references.format_block(retrieved), self.references.retrieved_ids(retrieved)

    def _user_text(self, results, unit: Unit, retry_note: str = "") -> str:
        items = list(zip(self._labels(results, unit.members), self._texts(results, unit.members)))
        return self.spec.build_user(self._texts(results, unit.before), items, self._texts(results, unit.after),
                                    reference_block=self._reference(results, unit)[0], retry_note=retry_note)

    def _retry_note(self, results, empties: list[int]) -> str:
        ids = ", ".join(str(label) for label in self._labels(results, empties))
        return (f"your previous reply gave an empty candidate_gloss for: {ids}. Reconsider only those sentences. "
                "If a sentence is understandable (even with an ambiguous pronoun, an unfamiliar name, a complex structure or "
                "some remaining doubt), give your best candidate gloss and state the specific doubt in uncertainty_notes; "
                "ambiguity is never a reason for an empty gloss, and pronouns stay pronouns. If a sentence is genuinely garbled "
                "or unintelligible, keep candidate_gloss empty, set needs_human_review to true and give a specific reason in "
                "uncertainty_notes; do not invent a translation. Return a result only for the target sentences.")

    def _retry_empty_glosses(self, results, unit: Unit, out: dict) -> None:
        """One bounded re-ask for sentences that came back with an empty gloss. The model decides again; nothing is rewritten
        here. The retry is kept only if it produced a gloss, or a stated reason where the first reply gave none."""
        empties = [i for i in unit.members if isinstance(out.get(i), Candidate) and not out[i].tokens]
        if not empties:
            return
        sub = self._subunit(unit, empties)
        note = self._retry_note(results, empties)
        self.stats["empty_gloss_retries"] += 1
        try:
            retried = self._translate_unit(results, sub, MAX_SPLIT_DEPTH, retry_note=note, allow_retry=False)
        except _STOP_RUN:
            raise
        for idx in empties:
            new, old = retried.get(idx), out[idx]
            if not isinstance(new, Candidate):
                outcome = "retry_failed_kept_first_reply"
            elif new.tokens:
                out[idx], outcome = new, "recovered"
                self.stats["empty_gloss_recovered"] += 1
            elif not (old.notes or "").strip() and (new.notes or "").strip():
                out[idx], outcome = new, "still_empty_reason_given"
            else:
                outcome = "still_empty"
            results[idx].response_warnings.append({
                "code": "empty_gloss_retried", "severity": "info",
                "message": f"The first reply had an empty gloss; one targeted retry was made (outcome: {outcome})."})

    def _translate_unit(self, results, unit: Unit, depth: int, retry_note: str = "", allow_retry: bool = True) -> dict:
        """Returns {result_index: Candidate | error-dict}. Raises only for run-stopping errors."""
        members = unit.members
        labels = self._labels(results, members)
        user = self._user_text(results, unit, retry_note)
        try:
            started = time.monotonic()
            resp = self.client.complete(self.system_prompt, user, self.spec.schema_for(labels))
            self._record_usage(resp, time.monotonic() - started)
            rows = parse_results_json(resp.text)
        except _STOP_RUN:
            raise
        except (TruncatedResponse, ValueError) as exc:
            return self._split_or_fail(results, unit, depth, "malformed_response", str(exc))
        except LLMError as exc:
            return {i: {"kind": exc.kind, "message": str(exc)} for i in members}

        key_field = self.spec.key_field
        position = {label: pos for pos, label in enumerate(labels)}
        by_label: dict = {}
        issues: dict = {}
        order: list[int] = []
        stray = 0
        for row in rows:
            label = row.get(key_field)
            cand = self._candidate_from_row(row)
            valid = isinstance(label, (str, int)) and not isinstance(label, bool) and label in position
            if not valid:
                stray += 1
                continue
            if label in by_label:
                issues.setdefault(label, []).append({
                    "code": "duplicate_result_for_sentence", "severity": "review",
                    "message": "The model returned more than one result for this sentence; the first valid one was used."})
                continue
            if cand is None:
                continue
            by_label[label] = cand
            order.append(position[label])
        misordered = order != sorted(order)
        by = "id" if self.spec.id_mode else "number"
        out: dict = {}
        missing = []
        for k, idx in enumerate(members):
            label = labels[k]
            if label not in by_label:
                missing.append(idx)
                continue
            cand = by_label[label]
            out[idx] = cand
            warn = results[idx].response_warnings
            warn.extend(issues.get(label, []))
            if misordered:
                warn.append({"code": "results_out_of_order", "severity": "info",
                             "message": f"The model returned the sentence results out of order; matched by sentence {by}."})
            if stray:
                warn.append({"code": "stray_results_ignored", "severity": "info",
                             "message": f"{stray} result(s) with an unusable sentence {by} were ignored."})
            if self.spec.id_mode:  # the model must copy the sentence back; a different text means the gloss may be misaligned
                sent = " ".join(results[idx].prepared.llm_text.split())
                if cand.echoed_source_text is None:
                    warn.append({"code": "source_text_not_echoed", "severity": "review",
                                 "message": "The model did not return source_text, so alignment with the sentence could not be checked."})
                elif " ".join(cand.echoed_source_text.split()) != sent:
                    warn.append({"code": "source_text_mismatch", "severity": "review",
                                 "message": "The source_text the model returned differs from the sentence it was given; "
                                            "the gloss may belong to a different sentence."})
        ref_ids = self._reference(results, unit)[1]
        if ref_ids and not retry_note:
            self.stats["reference_entries_sent"] += len(ref_ids)
            for idx in out:
                results[idx].response_warnings.append({
                    "code": "reference_entries_supplied", "severity": "info",
                    "message": "Reviewed reference entries were included in the request: " + ", ".join(ref_ids)})
        if self.retry_empty and allow_retry:
            self._retry_empty_glosses(results, unit, out)
        # store what we got, keyed by the exact request text, so a resumed run does not pay for it again
        for k, idx in enumerate(members):
            if idx in out:
                self.store.put(cache_key(self.cfg.model, self.system_prompt, user, labels[k], self.settings), out[idx], self.cfg.model)
        if missing:  # one bounded repair attempt for just the missing sentences, with the rest as read-only context
            sub = self._subunit(unit, missing)
            if depth < MAX_SPLIT_DEPTH and len(missing) < len(members):
                out.update(self._translate_unit(results, sub, depth + 1))
            else:
                out.update(self._split_or_fail(results, sub, depth, "missing_result",
                                               "The model reply did not include a usable result for this sentence."))
        return out

    @staticmethod
    def _subunit(unit: Unit, subset: list[int]) -> Unit:
        first = subset[0]
        rest = [m for m in unit.members if m not in subset]
        return Unit(members=list(subset), before=unit.before + [m for m in rest if m < first],
                    after=[m for m in rest if m > first] + unit.after, group_id=unit.group_id)

    def _split_or_fail(self, results, unit: Unit, depth: int, kind: str, message: str) -> dict:
        members = unit.members
        if len(members) > 1 and depth < MAX_SPLIT_DEPTH:
            mid = len(members) // 2
            out = self._translate_unit(results, self._subunit(unit, members[:mid]), depth + 1)
            out.update(self._translate_unit(results, self._subunit(unit, members[mid:]), depth + 1))
            return out
        return {i: {"kind": kind, "message": message} for i in members}

    @staticmethod
    def _candidate_from_row(row: dict) -> Optional[Candidate]:
        """Parse one reply row. Prompt 6.0 fields are candidate_gloss / uncertainty_notes / needs_human_review; the 5.0
        fields (gloss / notes / status / uncertain) are still understood, so older replies and cached data keep working."""
        gloss = row.get("candidate_gloss", row.get("gloss"))
        if isinstance(gloss, str):  # tolerate a space-separated string; do not guess beyond that
            gloss = gloss.split()
        if not isinstance(gloss, list) or not all(isinstance(t, str) for t in gloss):
            return None
        status = row.get("status") if row.get("status") in STATUS_VALUES else None
        needs = row.get("needs_human_review")
        if not isinstance(needs, bool):
            needs = (status in ("uncertain", "review_required")) if status else bool(row.get("uncertain", False))
        parts = row.get("uncertain_parts")
        parts = [p for p in parts if isinstance(p, str)] if isinstance(parts, list) else []
        notes = row.get("uncertainty_notes", row.get("notes", ""))
        echo = row.get("source_text")
        return Candidate(tokens=[t for t in gloss], notes=str(notes or "")[:300], status=status or "ok",
                         uncertain_parts=parts[:20], needs_human_review=needs,
                         echoed_source_text=echo if isinstance(echo, str) else None)

    def _finish(self, r: ChunkResult) -> None:
        prep, cand = r.prepared, r.candidate
        ctx = RuleContext(source_text=prep.llm_text, skipped_words=skipped_words_from_spans(prep.skipped, prep.llm_text))
        try:
            r.final_tokens, r.applied_rules = self.engine.apply(cand.tokens, ctx)
            r.rule_pass_completed = True
        except RulePassError as exc:
            r.final_tokens, r.applied_rules, r.rule_pass_completed = list(cand.tokens), [], False
            r.error = {"kind": "rule_pass_failed", "message": str(exc)}
        r.status = TRANSLATED
        r.validation = validate_gloss(prep, cand, r.final_tokens, r.applied_rules, r.rule_pass_completed, self.vocabulary)
        if r.response_warnings:
            r.validation["warnings"].extend(r.response_warnings)
            r.validation["status"] = status_from(r.validation["warnings"])

    def translate_text(self, text: str, chunk_id: str = "adhoc-0001") -> ChunkResult:
        """Convenience for one ad-hoc sentence."""
        return self.run([SourceChunk(chunk_id=chunk_id, text=text)]).results[0]
