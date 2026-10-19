"""Phase F: a small, file-based job record. No database, queue or server: a job is a folder.

    <data_dir>/<job_id>/
        job.json            state, history, per-stage status, error (stage + message + hint), output paths
        job.log             timestamped log lines
        metadata/video.json
        audio/audio.wav, audio/audio.json
        transcript/transcript.json (canonical), transcript.txt, transcript.vtt
        cleaning/...        the existing cleaner's folders (raw, cleaned, structured, reports, review_queue)
        gloss/...           the existing translator's output
        timeline/timeline.json   sentences / groups / glosses mapped back to original-video seconds

job.json is rewritten atomically, so an interruption never leaves a half-written record, and a re-run can skip stages that finished.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

PENDING, DOWNLOADING, AUDIO_EXTRACTED, TRANSCRIBING, TRANSCRIBED = "PENDING", "DOWNLOADING", "AUDIO_EXTRACTED", "TRANSCRIBING", "TRANSCRIBED"
PROCESSING, COMPLETED, FAILED = "PROCESSING_EXISTING_PIPELINE", "COMPLETED", "FAILED"
STATES = (PENDING, DOWNLOADING, AUDIO_EXTRACTED, TRANSCRIBING, TRANSCRIBED, PROCESSING, COMPLETED, FAILED)
JOB_SCHEMA_VERSION = "1.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


REPLACE_RETRIES, REPLACE_PAUSE = 60, 0.05  # about 3 s in total


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file and an atomic replace. On Windows, replacing a file that another thread (for example the web page's status
    polling) happens to have open raises PermissionError for a moment; retrying briefly is the standard remedy and keeps a running job
    from being killed by a reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        for attempt in range(REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == REPLACE_RETRIES - 1:
                    raise
                time.sleep(REPLACE_PAUSE)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Job:
    def __init__(self, root: Path, job_id: str, echo: Callable[[str], None] = print):
        self.root, self.id, self.echo = Path(root), job_id, echo
        self.dir = self.root / job_id
        self.path = self.dir / "job.json"
        data = read_json(self.path)
        self.data: dict[str, Any] = data or {"schema_version": JOB_SCHEMA_VERSION, "job_id": job_id, "state": PENDING, "created_at": now(),
                                              "updated_at": now(), "history": [], "stages": {}, "error": None, "outputs": {}}

    # --- paths (always inside the job folder) ---
    def sub(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.dir.resolve()).as_posix()

    # --- state ---
    @property
    def state(self) -> str:
        return self.data["state"]

    def save(self) -> None:
        self.data["updated_at"] = now()
        write_json(self.path, self.data)

    def set_state(self, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown job state {state!r}")
        self.data["state"] = state
        self.data["history"].append({"state": state, "at": now()})
        if state != FAILED:
            self.data["error"] = None
        self.save()
        self.log(f"state -> {state}")

    def stage_done(self, name: str, **info: Any) -> None:
        self.data["stages"][name] = {"status": "completed", "at": now(), **info}
        self.save()

    def stage_info(self, name: str) -> Optional[dict]:
        s = self.data["stages"].get(name)
        return s if s and s.get("status") == "completed" else None

    def clear_stage(self, name: str) -> None:
        self.data["stages"].pop(name, None)

    def output(self, name: str, path: Path) -> None:
        self.data["outputs"][name] = self.rel(path)
        self.save()

    def fail(self, stage: str, error: dict) -> None:
        self.data["error"] = {"stage": stage, **{k: v for k, v in error.items() if k != "stage"}, "at": now()}
        self.data["state"] = FAILED
        self.data["history"].append({"state": FAILED, "at": now(), "stage": stage})
        self.save()
        self.log(f"FAILED at stage '{stage}': {error.get('kind')}: {error.get('message')}")

    # --- log ---
    def log(self, message: str, *, quiet: bool = False) -> None:
        line = f"{now()}  {message}"
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "job.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if not quiet:
            self.echo(message)


def list_jobs(root: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(root).glob("*/job.json")):
        d = read_json(p)
        if d:
            rows.append({"job_id": d.get("job_id"), "state": d.get("state"), "updated_at": d.get("updated_at"),
                         "error": (d.get("error") or {}).get("message")})
    return rows
