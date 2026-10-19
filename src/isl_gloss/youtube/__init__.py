"""YouTube ingestion phase: URL -> audio -> timestamped transcript, then the EXISTING cleaner and gloss pipeline.

Importing this package needs neither yt-dlp, FFmpeg nor a speech model; they are loaded only when a stage runs.
"""

from .config import IngestConfig
from .errors import IngestError
from .pipeline import Components, JobOutcome, run_youtube_job
from .urls import YouTubeRef, parse_youtube_url

__all__ = ["IngestConfig", "IngestError", "Components", "JobOutcome", "run_youtube_job", "YouTubeRef", "parse_youtube_url"]
