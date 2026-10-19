"""Backend of the interactive pipeline tester: runs the REAL pipeline on typed text and returns every stage.

    typed text -> cleaning -> sentences -> context groups -> Layer 1 (LLM) -> Layer 2 (spaCy) -> comparison and selection

Nothing here is a mock: it calls the same cleaner, sentence splitter, grouper, translator, spaCy adapter, comparator and
selection policy as the command line. The only extras are limits that protect the free-tier quota and a scrubber that makes
sure an API key can never appear in a response. Output is UNREVIEWED machine output, never verified ISL.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from transcript_cleaner.config import CleaningConfig
from transcript_cleaner.pipeline import process_bytes

from ..config import TranslatorConfig
from ..grouping import group_sentences, group_stats
from ..layer2.pipeline import HONESTY_NOTE, Layer2Config, build_items, summarise
from ..layer2.spacy_translator import SpacyTranslator, dependency_versions
from ..llm import LLMError, MissingApiKey, load_env_file, make_client
from ..prepare import chunks_from_sentences
from ..prompt import PROMPTS
from ..sentences import build_sentences
from ..store import ResultStore
from ..translator import Translator
from ..youtube.align import align_sentences
from ..youtube.jobs import read_json
from ..youtube.pipeline import source_id_for
from ..youtube.transcript import timestamp_to_seconds, to_vtt, validate_transcript

# The page has no option controls: it always runs with these settings (the project-wide defaults used by the command line are unchanged).
WEB_PROMPT_VERSION = "8.0"
WEB_REASONING_EFFORT = "medium"
MAX_CHARS = 6000  # one paragraph or two; protects the free-tier quota
MAX_SENTENCES = 40  # at most about 5 group requests
DEFAULT_VIDEO_SENTENCES = 10  # a video transcript is long: only the first N sentences go on to the LLM stage unless asked otherwise
JOB_ID = re.compile(r"^yt-[A-Za-z0-9_-]{11}$")
REASONS = {
    "end": "reached the end of the text", "turn_change": "a new speaker turn / paragraph started", "speaker_change": "the speaker changed",
    "time_gap": "a long pause in the timestamps", "max_sentences": "the group hit the sentence limit", "max_chars": "the group hit the character limit",
    "max_tokens": "the group hit the estimated-token limit",
}


class DemoError(ValueError):
    """A problem with the user's input; its message is safe to show."""


def _config(opts: dict) -> TranslatorConfig:
    cfg = TranslatorConfig.for_provider("groq")
    cfg.prompt_version = str(opts.get("prompt_version") or WEB_PROMPT_VERSION)
    cfg.reasoning_effort = str(opts.get("reasoning_effort") or WEB_REASONING_EFFORT)
    try:
        cfg.validate()
    except ValueError as exc:
        raise DemoError(str(exc)) from exc
    return cfg


def _int(value, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _job_transcript(job_id: str, jobs_dir: str | Path) -> dict:
    """The saved, valid transcript of a finished YouTube job. The id is validated, so it can never point outside the jobs folder."""
    if not isinstance(job_id, str) or not JOB_ID.match(job_id):
        raise DemoError("That is not a valid video job id.")
    doc = read_json(Path(jobs_dir) / job_id / "transcript" / "transcript.json")
    if not doc or validate_transcript(doc):
        raise DemoError("This video has no valid transcript yet. Run the video steps first.")
    return doc


def _stages(text: str, cfg: TranslatorConfig, opts: Optional[dict] = None, jobs_dir: str | Path = "data/jobs"):
    """Cleaning -> sentences -> groups, from typed text or from a YouTube job's timestamped transcript (through the same cleaner,
    fed the VTT so timestamps and segment ids survive). Returns (result, doc, sentences, groups, source_info)."""
    opts = opts or {}
    transcript = None
    if opts.get("job_id"):
        transcript = _job_transcript(opts["job_id"], jobs_dir)
        vid = transcript["video"]["video_id"]
        ccfg = CleaningConfig()
        ccfg.dedupe_rolling_captions = False  # these cues are ASR segments, not YouTube rolling captions
        res = process_bytes(to_vtt(transcript).encode("utf-8"), f"{vid}.vtt", ccfg, source_id=source_id_for(vid))
    else:
        if not isinstance(text, str) or not text.strip():
            raise DemoError("Type a sentence or a paragraph first.")
        if len(text) > MAX_CHARS:
            raise DemoError(f"That is {len(text)} characters; this tester takes up to {MAX_CHARS} so it cannot burn through the free API quota.")
        res = process_bytes(text.encode("utf-8"), "input.txt", CleaningConfig())
    doc = res.doc
    sentences = build_sentences(doc, cfg)
    info: dict[str, Any] = {"type": "youtube" if transcript else "text", "total_sentences": len(sentences), "total_chunks": len(doc["chunks"])}
    if transcript:
        limit = _int(opts.get("max_sentences"), DEFAULT_VIDEO_SENTENCES, 1, MAX_SENTENCES)
        sentences = sentences[:limit]
        v = transcript["video"]
        info.update(job_id=opts["job_id"], limit=limit, title=v.get("title"), video_id=v.get("video_id"), duration_seconds=v.get("duration_seconds"),
                    timestamp_level=transcript["transcription"]["timestamp_level"], segments=transcript["transcription"]["segments"])
    elif len(sentences) > MAX_SENTENCES:
        raise DemoError(f"That splits into {len(sentences)} sentences; this tester takes up to {MAX_SENTENCES}.")
    groups = group_sentences(sentences, cfg, source_id=doc["source_id"])
    return res, doc, sentences, groups, info


def llm_key_present(env_file: str | Path = ".env") -> bool:
    load_env_file(env_file)
    return bool(os.environ.get("GROQ_API_KEY"))


def youtube_readiness() -> dict[str, Any]:
    """Can a link be processed on this computer? (packages and FFmpeg present; nothing is downloaded or loaded here)"""
    missing = []
    if importlib.util.find_spec("yt_dlp") is None:
        missing.append("yt-dlp (python -m pip install yt-dlp)")
    if importlib.util.find_spec("faster_whisper") is None:
        missing.append("faster-whisper (python -m pip install faster-whisper)")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        missing.append("FFmpeg (winget install Gyan.FFmpeg)")
    return {"available": not missing, "missing": missing}


def status(env_file: str | Path = ".env") -> dict[str, Any]:
    ok, why = _spacy().available()
    return {"llm_key_present": llm_key_present(env_file), "spacy_available": ok, "spacy_problem": None if ok else why,
            "prompt_versions": sorted(PROMPTS), "default_prompt_version": WEB_PROMPT_VERSION, "default_reasoning_effort": WEB_REASONING_EFFORT,
            "limits": {"max_chars": MAX_CHARS, "max_sentences": MAX_SENTENCES}, "versions": dependency_versions(),
            "model": "groq/openai/gpt-oss-120b", "youtube": youtube_readiness(),
            "default_video_sentences": DEFAULT_VIDEO_SENTENCES}


_SPACY: Optional[SpacyTranslator] = None


def _spacy() -> SpacyTranslator:
    global _SPACY
    if _SPACY is None:
        _SPACY = SpacyTranslator()
    return _SPACY


def analyze(text: str, opts: Optional[dict] = None, jobs_dir: str | Path = "data/jobs") -> dict[str, Any]:
    """Stages 1-3 (cleaning, sentences, groups). Instant; makes no LLM call."""
    opts = opts or {}
    cfg = _config(opts)
    res, doc, sentences, groups, source = _stages(text, cfg, opts, jobs_dir)
    index = {s.sentence_id: i + 1 for i, s in enumerate(sentences)}
    kept_chunks = {s.source_metadata.get("parent_chunk_id") for s in sentences}
    chunks = [{"chunk_id": c["chunk_id"], "text": c["text"], "skip_spans": c["skip_translation_spans"],
               "flags": [{"code": f["code"], "text": f["text"], "note": f["note"], "skipped": f["skip_translation"]} for f in c["uncertainty_flags"]],
               "edit_notes": c["edit_notes"]} for c in doc["chunks"] if source["type"] == "text" or c["chunk_id"] in kept_chunks]
    times: dict[int, dict] = {}
    if source["type"] == "youtube":  # real start/end read from the matched words or segments; an unmatched sentence keeps its chunk's interval
        for i, al in enumerate(align_sentences(source["segments"], [s.source_text for s in sentences])):
            if al.status != "unaligned":
                times[i] = {"start": al.start, "end": al.end, "basis": al.timing_basis, "alignment": al.status}
            else:
                times[i] = {"start": timestamp_to_seconds(sentences[i].start_timestamp), "end": timestamp_to_seconds(sentences[i].end_timestamp),
                            "basis": "chunk_interval", "alignment": "unaligned"}
    sent_rows = []
    group_of = {sid: n + 1 for n, g in enumerate(groups) for sid in g.sentence_ids}
    for i, s in enumerate(sentences):
        d = s.to_dict()
        sent_rows.append({"index": i + 1, "sentence_id": s.sentence_id, "text": s.source_text, "skip_spans": d["skip_spans"],
                          "group_index": group_of.get(s.sentence_id), "split_from_run_on": bool(d["source_metadata"].get("split_method")),
                          "incomplete": d["incomplete"], "depends_on_previous": d["depends_on_previous"],
                          "dependency_reasons": d["dependency_reasons"], "time": times.get(i)})
    group_rows = []
    for n, g in enumerate(groups):
        group_rows.append({"index": n + 1, "group_id": g.group_id, "sentences": [index[s] for s in g.sentence_ids],
                           "context_before": [index[s] for s in g.context_before_ids if s in index],
                           "context_after": [index[s] for s in g.context_after_ids if s in index],
                           "boundary_reason": g.boundary_reason, "boundary_text": REASONS.get(g.boundary_reason, g.boundary_reason),
                           "forced_boundary": g.forced_boundary, "oversize": g.oversize,
                           "chars": sum(len(s.source_text) for s in sentences if s.sentence_id in g.sentence_ids)})
    return {"settings": {"prompt_version": cfg.prompt_version, "reasoning_effort": cfg.reasoning_effort},
            "cleaning": {"source_type": doc["source_type"], "status": doc["cleaning_status"], "warnings": doc["warnings"], "chunks": chunks},
            "sentences": sent_rows, "groups": group_rows, "grouping_stats": group_stats(groups),
            "estimated_llm_requests": len(groups), "source": {k: v for k, v in source.items() if k != "segments"}}


def _compact_llm_row(i: int, d: dict) -> dict[str, Any]:
    warnings = [{"code": w["code"], "severity": w["severity"], "message": w["message"]} for w in d["validation"]["warnings"] if w["severity"] != "info"]
    return {"index": i + 1, "sentence_id": d["sentence_id"], "source_text": d["source_text"], "llm_input_text": d["llm_input_text"],
            "status": d["translation_status"], "candidate_tokens": d["candidate_tokens"], "final_tokens": d["final_tokens"],
            "applied_rules": [r["rule_id"] for r in d["applied_rules"]], "uncertainty_notes": d["uncertainty_notes"],
            "uncertain_parts": d["uncertain_parts"], "needs_human_review": d["needs_human_review"],
            "cache_hit": bool((d.get("metadata") or {}).get("cache_hit")), "validation_status": d["validation"]["status"],
            "warnings": warnings, "error": d["error"]}


def translate(text: str, opts: Optional[dict] = None, env_file: str | Path = ".env", data_dir: str | Path = "data/demo",
              client: Any = None, jobs_dir: str | Path = "data/jobs") -> dict[str, Any]:
    """Stages 4-6: Layer 1 (LLM), Layer 2 (spaCy) and the comparison. `client` can be injected (tests); otherwise the real
    Groq client is used when `call_llm` is true and a key exists. A failure of either translator never hides the other."""
    opts = opts or {}
    cfg = _config(opts)
    res, doc, sentences, groups, source = _stages(text, cfg, opts, jobs_dir)
    chunks = chunks_from_sentences(sentences)
    call_llm = bool(opts.get("call_llm", True))
    use_cache = bool(opts.get("use_cache", True))

    # what would be sent to the model: always shown, also when the call is skipped
    preview = Translator(cfg, None).run(chunks, dry_run=True, groups=groups)
    payloads = [{"group_id": p["group_id"], "sentences": p["sentences"], "user_payload": p["user_payload"]} for p in preview.dry_run_requests]
    system_prompt = Translator(cfg, None).system_prompt

    llm: dict[str, Any] = {"enabled": call_llm, "error": None, "requests": payloads, "system_prompt": system_prompt, "usage": None, "counts": None,
                           "complete": None, "stop_reason": None, "model": f"{cfg.provider}/{cfg.model}", "prompt_version": cfg.prompt_version,
                           "reasoning_effort": cfg.reasoning_effort}
    outcome = None
    tr = None
    if call_llm:
        try:
            if client is None:
                load_env_file(env_file)
                client = make_client(cfg)
            cache = Path(data_dir) / "llm_cache.jsonl" if use_cache else None
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
            tr = Translator(cfg, client, ResultStore(cache))
            outcome = tr.run(chunks, groups=groups)
            llm.update(usage=dict(tr.stats), counts=outcome.counts(), complete=outcome.complete, stop_reason=outcome.stop_reason)
            llm["usage"]["requests_made"] = getattr(client, "requests_made", None)
        except MissingApiKey as exc:
            llm["error"] = {"kind": "missing_api_key", "message": str(exc)}
        except LLMError as exc:
            llm["error"] = {"kind": exc.kind, "message": str(exc)[:300]}
        except Exception as exc:  # the page must show what happened, not a blank screen
            llm["error"] = {"kind": type(exc).__name__, "message": str(exc)[:300]}
    if outcome is None:
        outcome, tr = preview, Translator(cfg, None)  # sentences with status "not_attempted"; Layer 2 still runs
    meta = tr.metadata()
    rows = [r.to_dict(meta) for r in outcome.results]
    llm["rows"] = [_compact_llm_row(i, d) for i, d in enumerate(rows)]

    spacy_results, items = build_items(rows, _spacy(), Layer2Config())
    ok, why = _spacy().available()
    spacy = {"available": ok, "problem": None if ok else why, "versions": dependency_versions(),
             "rows": [{"index": i + 1, **r.to_dict()} for i, r in enumerate(spacy_results)]}
    comparison = {"items": [{"index": i + 1, **it} for i, it in enumerate(items)], "summary": summarise(items, spacy_results), "note": HONESTY_NOTE}
    return {"llm": llm, "spacy": spacy, "comparison": comparison}


def scrub(payload: str) -> str:
    """Last line of defence: remove any configured API key value from a serialised response."""
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 8:
            payload = payload.replace(value, "***")
    return payload
