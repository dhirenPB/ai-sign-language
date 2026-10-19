"""Phase A (metadata) and the first part of Phase B (audio-only download), through yt-dlp.

* Only publicly accessible videos. Nothing here signs in, uses cookies, or works around an access check: a private, removed,
  age-restricted, members-only or bot-checked video is reported as such and the job stops.
* Audio-only download (`bestaudio`): no video stream is fetched.
* yt-dlp is imported lazily and injectable, so tests never touch the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import IngestConfig
from .errors import (AuthRequiredError, DependencyMissingError, DownloadFailedError, IngestError, LiveStreamError, NetworkError,
                     RateLimitedError, VideoTooLongError, VideoUnavailableError)
from .urls import YouTubeRef

YTDLP_HELP = "Install yt-dlp: python -m pip install yt-dlp   (or: pip install -e .[youtube])"

# (pattern, factory) tried in order against yt-dlp's message. The wording is yt-dlp's, which changes; unmatched messages
# become a generic DownloadFailedError that still carries the original text.
_RULES: list[tuple[re.Pattern, Callable[[str], IngestError]]] = [
    (re.compile(r"private video", re.I), lambda m: VideoUnavailableError("This video is private.", details={"reason": "private", "yt_dlp": m})),
    (re.compile(r"(sign in|log in).*(age|confirm your age)|age[- ]restricted|inappropriate for some users", re.I),
     lambda m: AuthRequiredError("This video is age-restricted and needs a signed-in account.", details={"reason": "age_restricted", "yt_dlp": m})),
    (re.compile(r"members[- ]only|join this channel|available to this channel's members", re.I),
     lambda m: AuthRequiredError("This video is for channel members only.", details={"reason": "members_only", "yt_dlp": m})),
    (re.compile(r"sign in to confirm you.re not a bot|confirm you.re not a bot|cookies.*authentication", re.I),
     lambda m: AuthRequiredError("YouTube is asking for a sign-in / bot check before serving this video.", details={"reason": "bot_check", "yt_dlp": m})),
    (re.compile(r"live event will begin|premieres in|this live event|is upcoming|currently live", re.I),
     lambda m: LiveStreamError("This is a live or upcoming stream, not a finished video.", details={"yt_dlp": m})),
    (re.compile(r"not available in your country|blocked it in your country|geo.?restrict|uploader has not made this video available in your country", re.I),
     lambda m: VideoUnavailableError("This video is not available in your region.", details={"reason": "region_blocked", "yt_dlp": m})),
    (re.compile(r"removed by the uploader|has been removed|account associated with this video has been terminated|copyright", re.I),
     lambda m: VideoUnavailableError("This video has been removed.", details={"reason": "removed", "yt_dlp": m})),
    (re.compile(r"video unavailable|this video is (not available|unavailable)|does not exist|no longer available|incomplete youtube id", re.I),
     lambda m: VideoUnavailableError("This video is unavailable (it may be deleted or never existed).", details={"reason": "unavailable", "yt_dlp": m})),
    (re.compile(r"http error 429|too many requests|rate.?limit", re.I), lambda m: RateLimitedError("YouTube rate-limited the request.", details={"yt_dlp": m})),
    (re.compile(r"getaddrinfo|name or service not known|network is unreachable|timed out|timeout|connection (reset|refused|aborted)|"
                r"urlopen error|temporary failure in name resolution|unable to connect|ssl", re.I),
     lambda m: NetworkError("A network error stopped the request.", details={"yt_dlp": m})),
]


def classify_ytdlp_error(message: str) -> IngestError:
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(message)).strip()
    one_line = " ".join(text.split())[:400]
    for pattern, make in _RULES:
        if pattern.search(text):
            return make(one_line)
    return DownloadFailedError("yt-dlp could not download this video.", details={"yt_dlp": one_line})


@dataclass
class VideoInfo:
    video_id: str
    url: str
    canonical_url: str
    title: Optional[str]
    duration_seconds: Optional[float]
    channel: Optional[str] = None
    upload_date: Optional[str] = None
    availability: Optional[str] = None
    live_status: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _import_ytdlp():
    try:
        import yt_dlp  # noqa: WPS433
        return yt_dlp
    except ImportError as exc:
        raise DependencyMissingError("The yt-dlp package is not installed.", hint=YTDLP_HELP) from exc


class YouTubeDownloader:
    """`ydl_factory(options) -> context manager with extract_info/download` can be injected; the default is yt_dlp.YoutubeDL."""

    def __init__(self, cfg: IngestConfig, ydl_factory: Optional[Callable[[dict], Any]] = None, log: Callable[[str], None] = print):
        self.cfg, self.log = cfg, log
        self._factory = ydl_factory

    def _ydl(self, options: dict):
        if self._factory is not None:
            return self._factory(options)
        return _import_ytdlp().YoutubeDL(options)

    def _base_options(self) -> dict:
        return {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True, "retries": self.cfg.download_retries,
                "socket_timeout": self.cfg.socket_timeout_seconds, "ignoreerrors": False}

    def fetch_info(self, ref: YouTubeRef) -> VideoInfo:
        """Metadata without downloading. Raises a classified IngestError for unavailable / restricted / live / too-long videos."""
        try:
            with self._ydl(self._base_options()) as ydl:
                info = ydl.extract_info(ref.canonical_url, download=False)
        except IngestError:
            raise
        except Exception as exc:  # yt_dlp.utils.DownloadError and friends
            if type(exc).__name__ in ("ImportError", "ModuleNotFoundError"):
                raise DependencyMissingError("The yt-dlp package is not installed.", hint=YTDLP_HELP) from exc
            raise classify_ytdlp_error(str(exc)) from exc
        if not isinstance(info, dict):
            raise DownloadFailedError("yt-dlp returned no information for this video.")
        status = info.get("live_status")
        if info.get("is_live") or status in ("is_live", "is_upcoming"):
            raise LiveStreamError("This is a live or upcoming stream, not a finished video.", details={"live_status": status})
        availability = info.get("availability")
        if availability in ("private", "needs_auth", "subscriber_only", "premium_only"):
            err = VideoUnavailableError if availability == "private" else AuthRequiredError
            raise err(f"This video is not publicly available (availability: {availability}).", details={"availability": availability})
        duration = info.get("duration")
        duration = float(duration) if isinstance(duration, (int, float)) else None
        if duration is not None and duration > self.cfg.max_duration_seconds:
            raise VideoTooLongError(f"The video is {duration / 60:.0f} minutes long; the limit is {self.cfg.max_duration_seconds / 60:.0f} minutes.",
                                    details={"duration_seconds": duration, "limit_seconds": self.cfg.max_duration_seconds})
        vid = info.get("id") or ref.video_id
        return VideoInfo(video_id=vid, url=ref.original_url, canonical_url=ref.canonical_url, title=info.get("title"), duration_seconds=duration,
                         channel=info.get("channel") or info.get("uploader"), upload_date=info.get("upload_date"), availability=availability,
                         live_status=status)

    def download_audio(self, ref: YouTubeRef, dest_dir: Path, progress: Optional[Callable[[float], None]] = None) -> Path:
        """Download the best audio-only stream into dest_dir as source.<ext> and return its path."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for old in dest_dir.glob("source.*"):  # a stale partial from an earlier attempt must not be mistaken for this download
            old.unlink(missing_ok=True)
        last = {"pct": -10.0}

        def hook(d: dict) -> None:
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if d.get("status") == "downloading" and total:
                pct = 100.0 * (d.get("downloaded_bytes") or 0) / total
                if pct - last["pct"] >= 10:
                    last["pct"] = pct
                    (progress or (lambda p: self.log(f"  downloading audio: {p:.0f}%")))(pct)

        options = {**self._base_options(), "format": "bestaudio/best", "outtmpl": str(dest_dir / "source.%(ext)s"), "progress_hooks": [hook]}
        try:
            with self._ydl(options) as ydl:
                ydl.download([ref.canonical_url])
        except IngestError:
            raise
        except Exception as exc:
            if type(exc).__name__ in ("ImportError", "ModuleNotFoundError"):
                raise DependencyMissingError("The yt-dlp package is not installed.", hint=YTDLP_HELP) from exc
            raise classify_ytdlp_error(str(exc)) from exc
        files = sorted(p for p in dest_dir.glob("source.*") if p.suffix not in (".part", ".ytdl", ".tmp"))
        if not files:
            raise DownloadFailedError("The download finished but no audio file was produced.")
        return files[0]
