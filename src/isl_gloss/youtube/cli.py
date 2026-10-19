"""Command-line entry points for the YouTube phase (registered into the existing `isl_gloss` CLI)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from importlib import metadata
from pathlib import Path

from .audio import FFmpeg
from .config import IngestConfig
from .errors import DependencyMissingError, IngestError
from .jobs import list_jobs
from .pipeline import DEFAULT_GLOSS_LIMIT, job_id_for, render_job, run_youtube_job
from .urls import parse_youtube_url


def add_subparsers(sub) -> None:
    y = sub.add_parser("youtube", help="YouTube URL -> audio -> timestamped transcript -> existing cleaner -> existing ISL gloss pipeline")
    y.add_argument("url", help="https://www.youtube.com/watch?v=ID, https://youtu.be/ID or a /shorts/ link")
    y.add_argument("--through", choices=("transcript", "clean", "gloss"), default="gloss", help="stop after this stage (default: gloss)")
    y.add_argument("--gloss-limit", type=int, default=DEFAULT_GLOSS_LIMIT,
                   help=f"gloss only the first N sentences (default {DEFAULT_GLOSS_LIMIT}, to protect the free-tier quota); 0 = every sentence")
    y.add_argument("--force", action="append", choices=("metadata", "audio", "transcript", "all"), default=[], help="redo a stage even if its output is valid")
    y.add_argument("--ingest-config", type=Path, help="JSON file overriding IngestConfig fields")
    y.add_argument("--data-dir", help="where job folders are kept (default data/jobs)")
    y.add_argument("--engine", help="transcription engine (default faster-whisper)")
    y.add_argument("--model", help="speech model name (default small.en)")
    y.add_argument("--language", help="spoken language code, or 'auto' (default en)")
    y.add_argument("--device", choices=("cpu", "cuda"), help="default cpu")
    y.add_argument("--chunk-seconds", type=float, help="transcribe long audio in windows of this length (default 600)")
    y.add_argument("--no-word-timestamps", action="store_true", help="segment-level timestamps only (faster)")
    y.add_argument("--no-vad", action="store_true", help="do not skip silence inside the engine")
    y.add_argument("--keep-source-audio", action="store_true", help="keep the downloaded original audio file")
    g = y.add_argument_group("gloss stage (passed to the existing `translate` command)")
    g.add_argument("--llm-config", type=Path, help="JSON file for the translator (same as translate --config)")
    g.add_argument("--provider", choices=("groq", "gemini"))
    g.add_argument("--prompt-version", choices=("5.0", "6.0", "7.0", "8.0"))
    g.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    g.add_argument("--env-file", help="file with GROQ_API_KEY=...")
    g.add_argument("--layer2", action="store_true", help="also run Layer 2 (spaCy candidate + comparison) on the gloss output")

    s = sub.add_parser("youtube-show", help="inspect a job: status, timestamped transcript, gloss by original-video time")
    s.add_argument("job", help="a job id, a YouTube URL, or the path of a job folder")
    s.add_argument("--what", choices=("status", "transcript", "gloss", "all"), default="all")
    s.add_argument("--data-dir", help="default data/jobs")
    sub.add_parser("youtube-jobs", help="list the jobs in the data folder").add_argument("--data-dir")
    c = sub.add_parser("youtube-check", help="check that yt-dlp, FFmpeg and the transcription engine are installed")
    c.add_argument("--ingest-config", type=Path)


def build_config(args) -> IngestConfig:
    cfg = IngestConfig.from_json(args.ingest_config) if getattr(args, "ingest_config", None) else IngestConfig()
    over = {}
    for attr in ("data_dir", "engine", "model", "device", "chunk_seconds"):
        v = getattr(args, attr, None)
        if v is not None:
            over[attr] = v
    if getattr(args, "language", None) is not None:
        over["language"] = None if args.language.lower() == "auto" else args.language
    if getattr(args, "no_word_timestamps", False):
        over["word_timestamps"] = False
    if getattr(args, "no_vad", False):
        over["vad_filter"] = False
    if getattr(args, "keep_source_audio", False):
        over["keep_source_audio"] = True
    cfg = replace(cfg, **over)
    cfg.validate()
    return cfg


def _gloss_args(args) -> list[str]:
    out: list[str] = []
    for flag, attr in (("--config", "llm_config"), ("--provider", "provider"), ("--prompt-version", "prompt_version"),
                       ("--reasoning-effort", "reasoning_effort"), ("--env-file", "env_file")):
        v = getattr(args, attr, None)
        if v:
            out += [flag, str(v)]
    if getattr(args, "layer2", False):
        out.append("--layer2")
    return out


def _print_error(err: dict) -> None:
    print(f"Error ({err.get('stage')}): {err.get('message')}")
    if err.get("hint"):
        print(f"  {err['hint']}")


def cmd_youtube(args) -> int:
    try:
        cfg = build_config(args)
        outcome = run_youtube_job(args.url, cfg, through=args.through, gloss_limit=args.gloss_limit or None, gloss_args=_gloss_args(args),
                                  force=args.force, log=print)
    except IngestError as exc:  # raised before a job exists (bad URL)
        _print_error(exc.to_dict())
        return 2
    job = outcome.job
    if not outcome.ok:
        print(f"\nJob {job.id} FAILED. Inspect it with: isl_gloss youtube-show {job.id}")
        _print_error(outcome.error or {})
        return 1
    print(f"\nJob {job.id} COMPLETED (through '{args.through}'). Folder: {job.dir}")
    print(f"Inspect: isl_gloss youtube-show {job.id}")
    return 0


def _resolve_job_dir(ref: str, data_dir: str | None) -> Path:
    p = Path(ref)
    if (p / "job.json").is_file():
        return p
    root = Path(data_dir or IngestConfig().data_dir)
    try:
        return root / job_id_for(parse_youtube_url(ref))
    except IngestError:
        return root / ref


def cmd_youtube_show(args) -> int:
    d = _resolve_job_dir(args.job, args.data_dir)
    text = render_job(d, args.what)
    print(text)
    return 0 if (d / "job.json").is_file() else 1


def cmd_youtube_jobs(args) -> int:
    rows = list_jobs(Path(args.data_dir or IngestConfig().data_dir))
    for r in rows:
        print(f"{r['job_id']:<24} {r['state']:<30} {r['updated_at']}  {r['error'] or ''}")
    if not rows:
        print("No jobs yet.")
    return 0


def cmd_youtube_check(args) -> int:
    cfg = IngestConfig.from_json(args.ingest_config) if getattr(args, "ingest_config", None) else IngestConfig()
    ok = True

    def ver(*names):
        for n in names:
            try:
                return metadata.version(n)
            except metadata.PackageNotFoundError:
                continue
        return None

    report = {"yt-dlp": ver("yt-dlp"), "faster-whisper": ver("faster-whisper"), "ctranslate2": ver("ctranslate2")}
    try:
        report["ffmpeg"] = FFmpeg(cfg.ffmpeg_path, cfg.ffprobe_path).check()
    except DependencyMissingError as exc:
        report["ffmpeg"] = f"MISSING: {exc.message}"
        ok = False
    print(json.dumps(report, indent=2))
    if not report["yt-dlp"]:
        print("yt-dlp is missing: python -m pip install yt-dlp")
        ok = False
    if not report["faster-whisper"]:
        print("faster-whisper is missing: python -m pip install faster-whisper   (the model, e.g. small.en, downloads on first use)")
        ok = False
    return 0 if ok else 1


HANDLERS = {"youtube": cmd_youtube, "youtube-show": cmd_youtube_show, "youtube-jobs": cmd_youtube_jobs, "youtube-check": cmd_youtube_check}
