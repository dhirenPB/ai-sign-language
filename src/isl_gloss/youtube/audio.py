"""Phase B: audio extraction. Independent of any command line or web page.

* Converts the downloaded audio to 16 kHz mono 16-bit PCM WAV with FFmpeg. The conversion keeps the original timeline: no
  silence removal, no trimming, no concatenation. (Speech models consume this format, and a WAV can be read at exact offsets.)
* Windows of the WAV are cut in pure Python with the `wave` module, so a long recording is never loaded into memory whole.
* FFmpeg is called through an injectable runner so tests never need the real binary.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import AudioConversionError, DependencyMissingError, NoAudioError

Runner = Callable[[list], "subprocess.CompletedProcess"]
FFMPEG_HELP = "Install FFmpeg (https://ffmpeg.org/download.html; on Windows: winget install Gyan.FFmpeg) and make sure `ffmpeg` and `ffprobe` are on PATH, or set ffmpeg_path/ffprobe_path in the ingest config."
_NO_AUDIO = re.compile(r"does not contain any stream|Output file #0 does not contain|matches no streams|Stream map '.*' matches no streams|no audio", re.I)


def _default_runner(cmd: list) -> "subprocess.CompletedProcess":
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)


class FFmpeg:
    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", runner: Optional[Runner] = None):
        self.ffmpeg, self.ffprobe, self.run = ffmpeg, ffprobe, runner or _default_runner

    def _call(self, cmd: list) -> "subprocess.CompletedProcess":
        try:
            return self.run(cmd)
        except FileNotFoundError as exc:
            raise DependencyMissingError(f"{cmd[0]!r} was not found.", hint=FFMPEG_HELP) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioConversionError(f"{Path(cmd[0]).name} timed out.", details={"command": Path(cmd[0]).name}) from exc

    def check(self) -> dict[str, Optional[str]]:
        """Is FFmpeg usable? Returns the version line, or raises DependencyMissingError."""
        out = {}
        for name, exe in (("ffmpeg", self.ffmpeg), ("ffprobe", self.ffprobe)):
            if self.run is _default_runner and shutil.which(exe) is None:
                raise DependencyMissingError(f"{exe!r} was not found on PATH.", hint=FFMPEG_HELP)
            res = self._call([exe, "-version"])
            if res.returncode != 0:
                raise DependencyMissingError(f"{exe!r} did not run correctly.", hint=FFMPEG_HELP)
            out[name] = (res.stdout or "").splitlines()[0] if res.stdout else None
        return out

    def probe(self, path: Path) -> dict[str, Any]:
        """Duration, audio presence and start offset of a media file (ffprobe)."""
        res = self._call([self.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
        if res.returncode != 0:
            raise AudioConversionError(f"ffprobe could not read {Path(path).name}: {(res.stderr or '').strip()[-300:]}",
                                       hint="The file may be corrupt or in an unsupported format.")
        try:
            data = json.loads(res.stdout or "{}")
        except ValueError as exc:
            raise AudioConversionError("ffprobe returned unreadable output.") from exc
        audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
        fmt = data.get("format", {})

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return {"has_audio": bool(audio), "codec": audio[0].get("codec_name") if audio else None,
                "duration_seconds": num((audio[0].get("duration") if audio else None) or fmt.get("duration")),
                "start_time_seconds": num((audio[0].get("start_time") if audio else None) or fmt.get("start_time")),
                "container": fmt.get("format_name")}

    def to_wav(self, src: Path, dst: Path, sample_rate: int = 16000, channels: int = 1) -> dict[str, Any]:
        """Convert to PCM WAV. Raises NoAudioError for a file without an audio stream, AudioConversionError otherwise."""
        src, dst = Path(src), Path(dst)
        info = self.probe(src)
        if not info["has_audio"]:
            raise NoAudioError(f"{src.name} has no audio stream.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part.wav")
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-map", "0:a:0", "-ac", str(channels),
               "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)]
        res = self._call(cmd)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            err = (res.stderr or "").strip()
            if _NO_AUDIO.search(err):
                raise NoAudioError(f"{src.name} has no usable audio stream.", details={"stderr": err[-300:]})
            raise AudioConversionError(f"FFmpeg could not convert {src.name} to WAV.", details={"stderr": err[-500:]},
                                       hint="The codec may be unsupported by this FFmpeg build; update FFmpeg and run again.")
        tmp.replace(dst)  # a finished file only: an interrupted conversion never leaves a half-written audio.wav
        return info


# --- WAV helpers (pure Python) ------------------------------------------------------------------------------------------
def wav_info(path: Path) -> dict[str, Any]:
    """Real duration of a PCM WAV from its header."""
    try:
        with wave.open(str(path), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            return {"sample_rate": rate, "channels": w.getnchannels(), "sample_width_bytes": w.getsampwidth(), "frames": frames,
                    "duration_seconds": frames / rate if rate else 0.0}
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioConversionError(f"{Path(path).name} is not a readable WAV file: {exc}") from exc


def extract_wav_window(src: Path, start_seconds: float, duration_seconds: float, dst: Path) -> dict[str, float]:
    """Copy [start, start+duration) of a WAV into a new WAV, reading in small blocks (memory stays bounded).
    Returns the exact start and length that were written, in seconds on the ORIGINAL timeline."""
    with wave.open(str(src), "rb") as r:
        rate, total = r.getframerate(), r.getnframes()
        first = max(0, min(total, int(round(start_seconds * rate))))
        last = max(first, min(total, first + int(round(duration_seconds * rate))))
        r.setpos(first)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as w:
            w.setnchannels(r.getnchannels())
            w.setsampwidth(r.getsampwidth())
            w.setframerate(rate)
            left, block = last - first, rate * 30
            while left > 0:
                data = r.readframes(min(block, left))
                if not data:
                    break
                w.writeframes(data)
                left -= len(data) // (r.getnchannels() * r.getsampwidth())
    return {"start_seconds": first / rate, "duration_seconds": (last - first) / rate}
