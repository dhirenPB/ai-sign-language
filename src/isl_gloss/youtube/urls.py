"""Phase A, step 1: accept only real YouTube video URLs and extract a valid video id. Pure string handling: no network."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from .errors import InvalidUrlError

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_WATCH_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_NOCOOKIE_HOSTS = {"youtube-nocookie.com", "www.youtube-nocookie.com"}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}
_PATH_KINDS = {"shorts": "short", "embed": "embed", "live": "live", "v": "watch"}


@dataclass(frozen=True)
class YouTubeRef:
    video_id: str
    url_type: str  # watch | short | youtu.be | embed | live
    original_url: str

    @property
    def canonical_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def parse_youtube_url(url: object) -> YouTubeRef:
    """Return the video reference, or raise InvalidUrlError with a message saying what is wrong."""
    if not isinstance(url, str) or not url.strip():
        raise InvalidUrlError("No URL was given.", hint="Pass a link such as https://www.youtube.com/watch?v=VIDEO_ID")
    original = url.strip()
    parts = urlsplit(original)
    if parts.scheme.lower() not in ("http", "https"):
        raise InvalidUrlError(f"{original!r} is not an http(s) URL.", hint="Include the full link, starting with https://")
    host = (parts.hostname or "").lower()
    if parts.username or parts.password:
        raise InvalidUrlError("URLs with a user name or password are not accepted.")
    if parts.port not in (None, 80, 443):
        raise InvalidUrlError(f"Unexpected port {parts.port} in the URL.")
    path = parts.path or "/"

    if host in _SHORT_HOSTS:
        candidate, kind = path.strip("/").split("/")[0], "youtu.be"
        if not candidate:
            raise InvalidUrlError("This youtu.be link has no video id.", hint="Use https://youtu.be/VIDEO_ID")
    elif host in _WATCH_HOSTS or host in _NOCOOKIE_HOSTS:
        first = path.strip("/").split("/")[0].lower() if path.strip("/") else ""
        if first == "watch":
            values = parse_qs(parts.query).get("v", [])
            if not values:
                raise InvalidUrlError("This watch link has no video id (no 'v=' value).", hint="Use https://www.youtube.com/watch?v=VIDEO_ID")
            candidate, kind = values[0], "watch"
        elif first in _PATH_KINDS:
            segments = path.strip("/").split("/")
            if len(segments) < 2 or not segments[1]:
                raise InvalidUrlError(f"This /{first}/ link has no video id.")
            candidate, kind = segments[1], _PATH_KINDS[first]
        elif first in ("playlist", "channel", "c", "user", "results", "feed", "hashtag") or first.startswith("@") or not first:
            what = "a playlist" if first == "playlist" else "a channel, search or page" if first else "the YouTube home page"
            raise InvalidUrlError(f"{original!r} is {what}, not a single video.", hint="Pass the link of one video: https://www.youtube.com/watch?v=VIDEO_ID")
        else:
            raise InvalidUrlError(f"Unsupported YouTube link type: /{first}.", hint="Supported: watch?v=, youtu.be/, /shorts/, /embed/, /live/")
    else:
        raise InvalidUrlError(f"{host or original!r} is not a YouTube address.", hint="Only youtube.com and youtu.be links are supported.")

    if not VIDEO_ID.match(candidate):
        raise InvalidUrlError(f"{candidate!r} is not a valid YouTube video id.", hint="A video id is exactly 11 letters, digits, '-' or '_'.")
    return YouTubeRef(video_id=candidate, url_type=kind, original_url=original)
