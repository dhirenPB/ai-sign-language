"""Errors of the YouTube ingestion phase. Every error carries the stage it happened in, a stable `kind`, and an actionable hint,
so a failed job can say exactly what went wrong and what to do. Nothing here is ever retried silently or worked around."""

from __future__ import annotations

from typing import Any, Optional


class IngestError(Exception):
    kind = "ingest_error"
    stage = "ingest"
    hint: Optional[str] = None

    def __init__(self, message: str, *, hint: Optional[str] = None, details: Optional[dict[str, Any]] = None,
                 stage: Optional[str] = None):
        super().__init__(message)
        self.message = message
        if hint:
            self.hint = hint
        self.details = details or {}
        if stage:
            self.stage = stage

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "stage": self.stage, "message": self.message, "hint": self.hint, "details": self.details}


class InvalidUrlError(IngestError):
    kind, stage = "invalid_url", "validation"


class VideoUnavailableError(IngestError):
    """Private, deleted, removed, region-blocked or otherwise inaccessible. Access restrictions are never bypassed."""
    kind, stage = "video_unavailable", "metadata"


class AuthRequiredError(IngestError):
    """The video needs a signed-in account (age check, members-only, bot check). This tool does not sign in or bypass it."""
    kind, stage = "authentication_required", "metadata"
    hint = "This tool does not sign in or bypass access checks. Use a video that is publicly viewable without an account."


class LiveStreamError(IngestError):
    kind, stage = "live_stream", "metadata"
    hint = "Wait until the live stream has ended and the recording is available, then try again."


class VideoTooLongError(IngestError):
    kind, stage = "video_too_long", "metadata"
    hint = "Raise max_duration_seconds in the ingest config if you really want to process it."


class RateLimitedError(IngestError):
    kind, stage = "rate_limited", "download"
    hint = "YouTube is limiting requests. Wait a while and run the same command again; finished stages are kept."


class NetworkError(IngestError):
    kind, stage = "network_error", "download"
    hint = "Check the internet connection and run the same command again; finished stages are kept."


class DownloadFailedError(IngestError):
    kind, stage = "download_failed", "download"


class NoAudioError(IngestError):
    kind, stage = "no_audio_stream", "audio"
    hint = "The downloaded file has no audio track, so there is nothing to transcribe."


class AudioConversionError(IngestError):
    kind, stage = "audio_conversion_failed", "audio"


class DependencyMissingError(IngestError):
    kind, stage = "dependency_missing", "setup"


class TranscriptionError(IngestError):
    kind, stage = "transcription_failed", "transcription"


class NoSpeechError(TranscriptionError):
    kind = "no_speech_detected"
    hint = "The audio is silent or contains no recognisable speech; nothing was passed to the gloss pipeline."


class InvalidTranscriptError(TranscriptionError):
    kind = "invalid_transcript"


class JobError(IngestError):
    kind, stage = "job_error", "job"
