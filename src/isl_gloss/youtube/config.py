"""Configuration of the YouTube ingestion phase. Secrets are not stored here; nothing in this phase needs one."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass
class IngestConfig:
    # --- storage ---
    data_dir: str = "data/jobs"  # jobs live in <data_dir>/<job_id>/ ; relative paths resolve against the working directory
    keep_source_audio: bool = False  # keep the originally downloaded audio file (it is large and not needed after conversion)

    # --- video limits and download behaviour ---
    max_duration_seconds: float = 3 * 3600  # refuse longer videos (guard against accidents); raise it deliberately
    download_retries: int = 3
    socket_timeout_seconds: float = 30.0
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # --- audio (the intermediate file is 16 kHz mono 16-bit PCM WAV: what speech models consume, exact to seek in) ---
    sample_rate: int = 16000
    channels: int = 1

    # --- transcription ---
    engine: str = "faster-whisper"  # see transcribe.ENGINES; an unknown or unavailable engine is an error, never substituted
    model: str = "small.en"  # English-only Whisper model; use a multilingual name (e.g. "small") with language null for other speech
    language: str = "en"  # null = let the model detect
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    word_timestamps: bool = True  # falls back to segment-level only if the engine returns no words; the output says which
    vad_filter: bool = True  # skips silence inside the engine WITHOUT changing the timeline (timestamps stay original)
    condition_on_previous_text: bool = False  # off: avoids a repetition loop carrying across a long window
    allow_model_download: bool = True  # False: use only models already on this computer
    model_dir: str | None = None  # where models are cached (None = the library default)
    chunk_seconds: float = 600.0  # long audio is transcribed in windows of this length so memory stays bounded
    overlap_seconds: float = 15.0  # windows overlap by this much so no word is cut at a boundary; duplicates are removed on merge

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown ingest config keys: {', '.join(unknown)}")
        cfg = replace(cls(), **data)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "IngestConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.chunk_seconds < 30:
            raise ValueError("chunk_seconds must be at least 30")
        if not 0 <= self.overlap_seconds < self.chunk_seconds / 2:
            raise ValueError("overlap_seconds must be >= 0 and less than half of chunk_seconds")
        if self.sample_rate < 8000 or self.channels not in (1, 2):
            raise ValueError("sample_rate must be >= 8000 and channels 1 or 2")
        if self.max_duration_seconds <= 0 or self.download_retries < 0:
            raise ValueError("max_duration_seconds must be positive and download_retries >= 0")

    def transcription_settings(self) -> dict[str, Any]:
        """Everything that changes the transcript. A stored transcript is reused only if these are identical."""
        keys = ("engine", "model", "language", "device", "compute_type", "beam_size", "word_timestamps", "vad_filter",
                "condition_on_previous_text", "chunk_seconds", "overlap_seconds", "sample_rate", "channels")
        return {k: getattr(self, k) for k in keys}
