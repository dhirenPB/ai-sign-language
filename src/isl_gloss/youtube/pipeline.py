"""Orchestration of the whole YouTube path:

    URL -> validate -> metadata -> audio-only download -> WAV -> timestamped transcript
        -> [existing] cleaner -> [existing] ISL gloss generation (`translate`) -> timeline mapping

The existing cleaner and translator are called as they are; nothing in them was changed. The LLM is only ever reached through
the existing `translate` command, after a successful transcript, and never during download or transcription. Every stage records
its result in job.json; a re-run skips stages whose outputs are still valid.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from transcript_cleaner.config import CleaningConfig
from transcript_cleaner.export import write_outputs
from transcript_cleaner.pipeline import process_bytes

from .align import align_sentences
from .audio import FFmpeg, wav_info
from .config import IngestConfig
from .downloader import VideoInfo, YouTubeDownloader
from .errors import IngestError, InvalidTranscriptError, JobError, NoAudioError, TranscriptionError
from .jobs import (AUDIO_EXTRACTED, COMPLETED, DOWNLOADING, PROCESSING, TRANSCRIBED, TRANSCRIBING, Job, atomic_write_text, read_json, write_json)
from .transcribe import TranscriptionEngine, make_engine, transcribe_audio
from .transcript import build_transcript, require_valid, timestamp_to_seconds, to_text, to_vtt, validate_transcript
from .urls import YouTubeRef, parse_youtube_url

THROUGH = ("transcript", "clean", "gloss")
KNOWN_STAGES = ("validation", "metadata", "download", "audio", "transcription", "cleaning", "gloss", "timeline")
DEFAULT_GLOSS_LIMIT = 10  # the gloss stage covers only the first N sentences unless asked otherwise: full transcripts are not translated by accident


@dataclass
class Components:
    """The external pieces, injectable so tests (and a future engine) never need the network, FFmpeg or a model."""

    downloader: YouTubeDownloader
    ffmpeg: FFmpeg
    engine_factory: Callable[[IngestConfig], TranscriptionEngine] = make_engine


def default_components(cfg: IngestConfig, log: Callable[[str], None] = print) -> Components:
    return Components(YouTubeDownloader(cfg, log=log), FFmpeg(cfg.ffmpeg_path, cfg.ffprobe_path))


@dataclass
class JobOutcome:
    job: Job
    ok: bool
    error: Optional[dict] = None
    outputs: dict = field(default_factory=dict)


def job_id_for(ref: YouTubeRef) -> str:
    return f"yt-{ref.video_id}"


def source_id_for(video_id: str) -> str:
    """The existing cleaner only accepts lowercase ids (^[a-z0-9][a-z0-9-]*$) and YouTube ids are case-sensitive, so the cleaner and
    every sentence id use this slug; the exact video id is kept in job.json and the transcript."""
    from transcript_cleaner.pipeline import make_source_id
    return "yt-" + make_source_id(video_id)


# ------------------------------------------------------------------------------------------------------------------------------
def run_youtube_job(url: str, cfg: Optional[IngestConfig] = None, components: Optional[Components] = None, *, through: str = "gloss",
                    gloss_limit: Optional[int] = DEFAULT_GLOSS_LIMIT, gloss_args: Sequence[str] = (), force: Sequence[str] = (),
                    log: Callable[[str], None] = print) -> JobOutcome:
    """Run (or resume) the job for one YouTube URL. Never raises for an expected failure: the outcome and job.json carry it."""
    cfg = cfg or IngestConfig()
    cfg.validate()
    if through not in THROUGH:
        raise ValueError(f"through must be one of {THROUGH}")
    ref = parse_youtube_url(url)  # raises InvalidUrlError before anything is created on disk
    components = components or default_components(cfg, log)
    job = Job(Path(cfg.data_dir), job_id_for(ref), echo=log)
    if job.data.get("video_id") not in (None, ref.video_id):
        raise JobError(f"The job folder {job.id} already belongs to a different video ({job.data['video_id']}); on a case-insensitive file "
                       "system two ids that differ only by case collide.", hint="Use a different data_dir for this video.", stage="validation")
    force_set = set(force) | ({"metadata", "audio", "transcript"} if "all" in force else set())
    job.data["video_id"], job.data["url"], job.data["through"] = ref.video_id, ref.original_url, through
    job.save()
    job.log(f"job {job.id} ({ref.canonical_url})")
    stage = "validation"
    try:
        stage = "metadata"
        video = _metadata(job, ref, components, cfg, "metadata" in force_set)
        stage = "download"
        audio = _audio(job, ref, video, components, cfg, "audio" in force_set)
        stage = "transcription"
        transcript = _transcript(job, video, audio, components, cfg, "transcript" in force_set)
        if through == "transcript":
            job.set_state(COMPLETED)
            return JobOutcome(job, True, outputs=dict(job.data["outputs"]))
        stage = "cleaning"
        cleaned_path = _clean(job, transcript)
        if through == "clean":
            job.set_state(COMPLETED)
            return JobOutcome(job, True, outputs=dict(job.data["outputs"]))
        stage = "gloss"
        gloss_path = _gloss(job, cleaned_path, gloss_limit, gloss_args)
        stage = "timeline"
        _timeline(job, transcript, cleaned_path, gloss_path)
        job.set_state(COMPLETED)
        return JobOutcome(job, True, outputs=dict(job.data["outputs"]))
    except IngestError as exc:
        err = exc.to_dict()
        failed = exc.stage if exc.stage in KNOWN_STAGES else stage  # e.g. a missing dependency is reported at the stage that needed it
        err["stage"] = failed
        job.fail(failed, err)
        return JobOutcome(job, False, err, dict(job.data["outputs"]))
    except Exception as exc:  # a bug must be recorded against its stage, not lose the job state
        err = {"kind": "unexpected_error", "stage": stage, "message": f"{type(exc).__name__}: {str(exc)[:300]}", "hint": "See job.log; this is a bug.", "details": {}}
        job.fail(stage, err)
        return JobOutcome(job, False, err, dict(job.data["outputs"]))


# --- stage 1: metadata ----------------------------------------------------------------------------------------------------------
def _metadata(job: Job, ref: YouTubeRef, c: Components, cfg: IngestConfig, force: bool) -> dict:
    path = job.sub("metadata", "video.json")
    cached = None if force else read_json(path)
    if cached and cached.get("video_id") == ref.video_id and job.stage_info("metadata"):
        job.log("metadata: reusing saved video information (no network)")
        return cached
    job.log("fetching video information")
    info: VideoInfo = c.downloader.fetch_info(ref)
    video = {**info.to_dict(), "video_id": ref.video_id, "url": ref.original_url, "canonical_url": ref.canonical_url, "url_type": ref.url_type}
    write_json(path, video)
    job.output("video_metadata", path)
    job.stage_done("metadata")
    job.log(f"  {video.get('title')!r}, {video.get('duration_seconds')}s")
    return video


# --- stage 2: audio ---------------------------------------------------------------------------------------------------------------
def _audio_valid(job: Job, video: dict) -> Optional[dict]:
    meta, wav = read_json(job.sub("audio", "audio.json")), job.sub("audio", "audio.wav")
    if not meta or not wav.is_file() or meta.get("video_id") != video["video_id"]:
        return None
    try:
        return meta if wav_info(wav)["duration_seconds"] > 0 else None
    except IngestError:
        return None


def _audio(job: Job, ref: YouTubeRef, video: dict, c: Components, cfg: IngestConfig, force: bool) -> dict:
    existing = None if force else _audio_valid(job, video)
    if existing and job.stage_info("audio"):
        job.log("audio: reusing the extracted audio (no download)")
        return existing
    job.set_state(DOWNLOADING)
    c.ffmpeg.check()  # fail early, before downloading, if FFmpeg is missing
    src_dir = job.sub("audio", "_download")
    job.log("downloading audio only")
    src = c.downloader.download_audio(ref, src_dir)
    job.log(f"converting {src.name} to {cfg.sample_rate} Hz WAV (timeline preserved: no trimming, no silence removal)")
    wav = job.sub("audio", "audio.wav")
    probe = c.ffmpeg.to_wav(src, wav, cfg.sample_rate, cfg.channels)
    info = wav_info(wav)
    if info["duration_seconds"] <= 0:
        raise NoAudioError("The converted audio has zero length.")
    meta_dur = video.get("duration_seconds")
    warnings = []
    if meta_dur and abs(info["duration_seconds"] - meta_dur) > max(2.0, 0.02 * meta_dur):
        warnings.append(f"Audio is {info['duration_seconds']:.1f}s but the video metadata says {meta_dur:.1f}s; timestamps follow the audio.")
    if probe.get("start_time_seconds") and abs(probe["start_time_seconds"]) > 0.05:
        warnings.append(f"The source audio stream starts at {probe['start_time_seconds']:.3f}s on its own clock; a small offset against the video is possible.")
    meta = {"video_id": video["video_id"], "path": job.rel(wav), "format": "wav", "codec": "pcm_s16le", "sample_rate": info["sample_rate"],
            "channels": info["channels"], "duration_seconds": round(info["duration_seconds"], 3), "source_file": src.name,
            "source_codec": probe.get("codec"), "source_container": probe.get("container"), "source_duration_seconds": probe.get("duration_seconds"),
            "source_start_time_seconds": probe.get("start_time_seconds"), "timeline_preserved": True,
            "processing": "converted only (resample/downmix); no silence removal, trimming or concatenation", "warnings": warnings}
    write_json(job.sub("audio", "audio.json"), meta)
    if not cfg.keep_source_audio:
        for f in src_dir.glob("*"):
            f.unlink(missing_ok=True)
        try:
            src_dir.rmdir()
        except OSError:
            pass
    job.output("audio", wav)
    job.stage_done("audio", duration_seconds=meta["duration_seconds"])
    job.set_state(AUDIO_EXTRACTED)
    for w in warnings:
        job.log(f"  warning: {w}")
    job.log(f"  audio: {meta['duration_seconds']:.1f}s")
    return meta


# --- stage 3: transcript -----------------------------------------------------------------------------------------------------------
def _transcript_valid(job: Job, settings: dict) -> Optional[dict]:
    doc = read_json(job.sub("transcript", "transcript.json"))
    if not doc or (doc.get("transcription") or {}).get("settings") != settings or validate_transcript(doc):
        return None
    return doc


def _write_transcript_files(job: Job, doc: dict) -> None:
    base = job.sub("transcript")
    write_json(base / "transcript.json", doc)  # canonical
    atomic_write_text(base / "transcript.txt", to_text(doc))
    atomic_write_text(base / "transcript.vtt", to_vtt(doc))
    job.output("transcript", base / "transcript.json")
    job.output("transcript_text", base / "transcript.txt")
    job.output("transcript_vtt", base / "transcript.vtt")


def _transcript(job: Job, video: dict, audio: dict, c: Components, cfg: IngestConfig, force: bool) -> dict:
    settings = cfg.transcription_settings()
    existing = None if force else _transcript_valid(job, settings)
    if existing and job.stage_info("transcript"):
        job.log("transcript: reusing the saved transcript (same engine and settings)")
        return existing
    job.set_state(TRANSCRIBING)
    engine = c.engine_factory(cfg)
    wav = job.sub("audio", "audio.wav")
    job.log(f"transcribing with {engine.name} / {getattr(engine, 'model_name', cfg.model)} (language {cfg.language or 'auto'})")
    result = transcribe_audio(wav, cfg, engine, log=job.log, work_dir=job.sub("audio"))
    doc = build_transcript(job.id, {k: video.get(k) for k in ("video_id", "url", "canonical_url", "title", "duration_seconds", "channel")},
                           {k: audio.get(k) for k in ("path", "format", "sample_rate", "channels", "duration_seconds", "timeline_preserved", "processing")},
                           result, engine.describe(), settings)
    require_valid(doc)  # a transcript with an impossible timeline is never saved as good
    _write_transcript_files(job, doc)
    job.stage_done("transcript", segments=len(doc["transcription"]["segments"]), timestamp_level=result.timestamp_level)
    job.set_state(TRANSCRIBED)
    for w in result.warnings:
        job.log(f"  warning: {w['message']}")
    job.log(f"  {len(result.segments)} segment(s), timestamp level: {result.timestamp_level}")
    return doc


# --- stage 4: the EXISTING cleaner ---------------------------------------------------------------------------------------------------
def _clean(job: Job, transcript: dict) -> Path:
    """Reached only with a valid, non-empty transcript. The cleaner is given the VTT derived from the canonical transcript."""
    if not transcript["transcription"]["segments"]:
        raise InvalidTranscriptError("Refusing to continue into the gloss pipeline: the transcript has no segments.")
    job.set_state(PROCESSING)
    job.log("cleaning (existing transcript cleaner)")
    cfg = CleaningConfig()
    cfg.dedupe_rolling_captions = False  # these cues are ASR segments, not YouTube rolling captions
    res = process_bytes(to_vtt(transcript).encode("utf-8"), f"{job.data['video_id']}.vtt", cfg, source_id=source_id_for(job.data["video_id"]))
    if res.status == "failed":
        raise JobError("The existing cleaner rejected the transcript.", stage="cleaning", details={"validation": res.validation})
    paths = write_outputs(res, job.sub("cleaning"))
    job.output("cleaned_json", Path(paths["structured"]))
    job.output("cleaned_text", Path(paths["cleaned"]))
    job.stage_done("cleaning", status=res.status, chunks=len(res.doc.get("chunks", [])))
    job.log(f"  {len(res.doc.get('chunks', []))} cleaned chunk(s), cleaner status: {res.status}")
    return Path(paths["structured"])


# --- stage 5: the EXISTING gloss generation --------------------------------------------------------------------------------------------
def _gloss(job: Job, cleaned: Path, limit: Optional[int], extra: Sequence[str]) -> Path:
    from ..cli import main as translate_main  # the current implementation, called unchanged

    argv = ["translate", str(cleaned), "-o", str(job.sub("gloss")), *([] if not limit else ["--limit", str(limit)]), *extra]
    job.log("generating ISL gloss (existing `translate` pipeline)" + (f", first {limit} sentence(s) only" if limit else ", all sentences"))
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = translate_main(argv)
    for line in (out.getvalue() + err.getvalue()).splitlines():
        if line.strip():
            job.log("  | " + line.rstrip(), quiet=False)
    gloss = job.sub("gloss", f"{cleaned.name.removesuffix('.cleaned.json')}.gloss.json")
    if code != 0 or not gloss.is_file():
        raise JobError(f"The existing gloss pipeline did not finish (exit code {code}).", stage="gloss",
                       hint="See job.log for its message (for example a missing GROQ_API_KEY or an exhausted free-tier quota). Fix it and run the "
                            "same command again: transcription and cleaning are reused and finished sentences are cached.")
    job.output("gloss", gloss)
    job.stage_done("gloss", limit=limit)
    return gloss


# --- stage 6: map everything back to the original timeline --------------------------------------------------------------------------------
def build_timeline(transcript: dict, cleaned: dict, gloss: dict) -> dict[str, Any]:
    """Sentences, groups and glosses with their original-video time. Each sentence is matched back to the transcript's words (or
    segments) by align.align_sentences; its time is then READ from the matched material, never interpolated. A sentence that cannot
    be matched keeps the interval of its cleaned chunk and is labelled so."""
    segments = transcript["transcription"]["segments"]
    segs = {s["id"]: s for s in segments}
    chunks = {c["chunk_id"]: c for c in cleaned.get("chunks", [])}
    results = gloss["results"]
    aligned = align_sentences(segments, [r["source_text"] for r in results])
    sentences, by_group = [], {}
    for r, al in zip(results, aligned):
        meta = r.get("source_metadata") or {}
        chunk = chunks.get(meta.get("parent_chunk_id")) or {}
        cue_ids = meta.get("cue_ids") or chunk.get("cue_ids") or []
        c_start, c_end = timestamp_to_seconds(r.get("start_timestamp")), timestamp_to_seconds(r.get("end_timestamp"))
        if al.status != "unaligned":
            start, end, basis, seg_ids = al.start, al.end, al.timing_basis, al.segment_ids
        else:
            start, end, basis, seg_ids = c_start, c_end, "chunk_interval", [c for c in cue_ids if c in segs]
        row = {"sentence_id": r["sentence_id"], "group_id": r.get("group_id"), "text": r["source_text"], "start": start, "end": end,
               "timing_basis": basis, "alignment": al.status, "segment_ids": seg_ids, "chunk_start": c_start, "chunk_end": c_end,
               "translation_status": r["translation_status"], "candidate_gloss": r.get("candidate_gloss"), "final_gloss": r.get("final_gloss"),
               "needs_human_review": r.get("needs_human_review")}
        sentences.append(row)
        by_group.setdefault(r.get("group_id"), []).append(row)
    groups = []
    for gid, rows in by_group.items():
        starts = [x["start"] for x in rows if x["start"] is not None]
        ends = [x["end"] for x in rows if x["end"] is not None]
        groups.append({"group_id": gid, "sentence_ids": [x["sentence_id"] for x in rows], "start": min(starts) if starts else None,
                       "end": max(ends) if ends else None})
    counts = {k: sum(1 for x in sentences if x["alignment"] == k) for k in ("exact", "approximate", "unaligned")}
    return {"schema_version": "1.0", "report_type": "youtube_timeline", "job_id": transcript["job_id"], "review_status": "unreviewed",
            "timeline": "All times are seconds from the start of the original YouTube video.", "sentences": sentences, "groups": groups,
            "coverage": {"sentences_with_gloss": len(sentences), "transcript_segments": len(segs), "alignment": counts,
                         "note": "Only sentences that were sent through gloss generation appear here (see the gloss limit). Times come from "
                                 "the matched words/segments (timing_basis); 'chunk_interval' means the sentence could not be matched and "
                                 "carries its whole cleaned chunk's interval. The matching is a heuristic."}}


def _timeline(job: Job, transcript: dict, cleaned_path: Path, gloss_path: Path) -> None:
    tl = build_timeline(transcript, read_json(cleaned_path) or {}, read_json(gloss_path) or {"results": []})
    path = job.sub("timeline", "timeline.json")
    write_json(path, tl)
    job.output("timeline", path)
    job.stage_done("timeline", sentences=len(tl["sentences"]))
    job.log(f"  timeline: {len(tl['sentences'])} sentence(s) mapped to original-video time")


# --- inspection ----------------------------------------------------------------------------------------------------------------------------
def fmt(seconds: Optional[float]) -> str:
    if seconds is None:
        return "  --:--.---"
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:06.3f}" if h else f"{m:02d}:{s:06.3f}"


def render_job(job_dir: Path, what: str = "status") -> str:
    """Human-readable view of a job's saved outputs; reads files only, runs nothing."""
    job = read_json(Path(job_dir) / "job.json")
    if not job:
        return f"No job found at {job_dir}"
    lines: list[str] = []
    if what in ("status", "all"):
        lines += [f"job {job['job_id']}  state: {job['state']}  (updated {job['updated_at']})"]
        for name, st in job.get("stages", {}).items():
            lines.append(f"  stage {name}: {st.get('status')} at {st.get('at')}")
        if job.get("error"):
            e = job["error"]
            lines += [f"  FAILED at '{e.get('stage')}': {e.get('kind')}: {e.get('message')}"] + ([f"  hint: {e['hint']}"] if e.get("hint") else [])
        for k, v in job.get("outputs", {}).items():
            lines.append(f"  output {k}: {Path(job_dir) / v}")
    if what in ("transcript", "all"):
        t = read_json(Path(job_dir) / "transcript" / "transcript.json")
        if t:
            tr = t["transcription"]
            lines += ["", f"transcript: {tr['engine']} / {tr['settings'].get('model')}, language {tr['language']}, timestamps: {tr['timestamp_level']}-level "
                          f"(seconds from the start of the video)"]
            lines += [f"[{fmt(s['start'])} -> {fmt(s['end'])}] {s['text']}" for s in tr["segments"]]
        elif what == "transcript":
            lines.append("No transcript has been produced for this job yet.")
    if what in ("gloss", "all"):
        tl = read_json(Path(job_dir) / "timeline" / "timeline.json")
        if tl:
            lines += ["", "gloss by original-video time (unreviewed machine output):"]
            for s in tl["sentences"]:
                lines.append(f"[{fmt(s['start'])} -> {fmt(s['end'])}] ({s['timing_basis']}) {s['sentence_id']}: {s['text']}\n      GLOSS: {s['final_gloss'] or '(none: ' + str(s['translation_status']) + ')'}")
        elif what == "gloss":
            lines.append("No gloss output for this job yet.")
    return "\n".join(lines)
