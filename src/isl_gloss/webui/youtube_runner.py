"""Background YouTube jobs for the tester page.

The page cannot wait for a download and a transcription inside one request, so the server runs the EXISTING `run_youtube_job`
(through="transcript": URL -> metadata -> audio -> transcript, no cleaning, no LLM) in a thread and the page polls `status()`.
`status()` is built only from the job's own saved files (job.json, metadata, audio.json, transcript.json), so what the page
shows is what was really written, and a page reload or a re-run of the same link simply picks the saved job up again.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..youtube.config import IngestConfig
from ..youtube.errors import IngestError
from ..youtube.jobs import FAILED, read_json
from ..youtube.pipeline import Components, default_components, job_id_for, run_youtube_job
from ..youtube.transcript import validate_transcript
from ..youtube.urls import parse_youtube_url

MODELS = ("base.en", "small.en")
WEB_SPEECH_MODEL = "small.en"  # the page has no model control: always this one
MAX_SEGMENTS_SHOWN = 300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class YouTubeRunner:
    def __init__(self, jobs_dir: str | Path = "data/jobs", max_video_minutes: float = 30.0,
                 components_factory: Optional[Callable[[IngestConfig, Callable[[str], None]], Components]] = None):
        self.jobs_dir, self.max_video_minutes = Path(jobs_dir), max_video_minutes
        self._factory = components_factory or (lambda cfg, log: default_components(cfg, log))
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------------------------------------------------------
    def config(self, model: Optional[str] = None) -> IngestConfig:
        cfg = IngestConfig(data_dir=str(self.jobs_dir), max_duration_seconds=self.max_video_minutes * 60, model=WEB_SPEECH_MODEL)
        if model:
            if model not in MODELS:
                raise IngestError(f"Unknown model {model!r}; choose one of {', '.join(MODELS)}.", stage="validation")
            cfg = replace(cfg, model=model)
        return cfg

    def start(self, url: str, model: Optional[str] = None, force: bool = False) -> dict[str, Any]:
        """Validate the link (raises InvalidUrlError on a bad one) and start the job in the background."""
        ref = parse_youtube_url(url)
        cfg = self.config(model)
        job_id = job_id_for(ref)
        with self._lock:
            if any(r["running"] for r in self._runs.values()):
                active = next(j for j, r in self._runs.items() if r["running"])
                if active == job_id:
                    return {"job_id": job_id, "started": False, "message": "This video is already being processed."}
                raise IngestError("Another video is still being processed. Wait for it to finish.", stage="validation")
            run = {"running": True, "started_at": _now(), "outcome": None, "lines": deque(maxlen=6), "url": ref.original_url}
            self._runs[job_id] = run

        def log(message: str) -> None:
            run["lines"].append(message)

        def work() -> None:
            try:
                comps = self._factory(cfg, log)
                run["outcome"] = run_youtube_job(ref.original_url, cfg, comps, through="transcript", force=("all",) if force else (), log=log)
            except Exception as exc:  # recorded for the page; never lost in a thread
                run["error"] = {"kind": type(exc).__name__, "stage": "validation", "message": str(exc)[:300], "hint": None}
            finally:
                run["running"] = False

        threading.Thread(target=work, daemon=True, name=f"yt-{ref.video_id}").start()
        return {"job_id": job_id, "started": True}

    # ------------------------------------------------------------------------------------------------------------------
    def status(self, job_id: str) -> dict[str, Any]:
        run = self._runs.get(job_id)
        job_dir = self.jobs_dir / job_id
        job = read_json(job_dir / "job.json") or {}
        stages = job.get("stages", {})
        started = run["started_at"] if run else None

        def done(name: str) -> bool:
            return name in stages and stages[name].get("status") == "completed"

        def reused(name: str) -> bool:
            return bool(started and done(name) and stages[name].get("at", "") < started)

        err = (job.get("error") if job.get("state") == FAILED else None) or (run or {}).get("error")
        if run and run["running"]:
            err = None  # a stale error from an earlier attempt is not the current state
        running = bool(run and run["running"])
        steps = []
        video = read_json(job_dir / "metadata" / "video.json") or {}
        audio = read_json(job_dir / "audio" / "audio.json") or {}
        tdoc = read_json(job_dir / "transcript" / "transcript.json")
        t = (tdoc or {}).get("transcription", {})
        failed_stage = (err or {}).get("stage")
        stage_of = {"metadata": "video", "download": "audio", "audio": "audio", "transcription": "transcript", "validation": "video"}.get(failed_stage or "", None)

        def step(key: str, label_done: str, label_run: str, ok: bool, prev_ok: bool, detail: dict) -> None:
            if ok:
                state = "done"
            elif err and stage_of == key:
                state = "failed"
            elif err:
                state = "pending"
            elif prev_ok and running:
                state = "running"
            elif key == "video" and running:
                state = "running"
            else:
                state = "pending"
            steps.append({"key": key, "state": state, "label": label_done if state == "done" else label_run, "reused": reused({"video": "metadata", "audio": "audio", "transcript": "transcript"}[key]) if ok else False,
                          "detail": detail if state == "done" else {}})

        v_ok, a_ok, t_ok = done("metadata"), done("audio"), done("transcript")
        step("video", "Video detected successfully", "Detecting the video", v_ok, True,
             {"title": video.get("title"), "duration_seconds": video.get("duration_seconds"), "channel": video.get("channel"), "video_id": video.get("video_id")})
        step("audio", "Audio extracted successfully", "Extracting the audio", a_ok, v_ok,
             {"duration_seconds": audio.get("duration_seconds"), "sample_rate": audio.get("sample_rate"), "channels": audio.get("channels"), "warnings": audio.get("warnings", [])})
        step("transcript", "Audio transcribed successfully", "Transcribing the audio", t_ok, a_ok,
             {"segments": len(t.get("segments", [])), "timestamp_level": t.get("timestamp_level"), "language": t.get("language"), "engine": t.get("engine"),
              "model": (t.get("settings") or {}).get("model"), "warnings": [w.get("message") for w in t.get("warnings", [])]})
        ready = bool(tdoc) and not validate_transcript(tdoc) and t_ok and not running
        out: dict[str, Any] = {"job_id": job_id, "running": running, "state": job.get("state"), "steps": steps, "ready": ready, "error": err,
                               "progress": list(run["lines"])[-1:] if run and running else []}
        if ready:
            segs = t["segments"]
            out["transcript"] = {"segments": [{"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in segs[:MAX_SEGMENTS_SHOWN]],
                                 "total_segments": len(segs), "timestamp_level": t.get("timestamp_level"), "language": t.get("language")}
        return out
