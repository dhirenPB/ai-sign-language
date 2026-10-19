"""Test doubles for the YouTube phase: a fake yt-dlp, a fake FFmpeg runner and a scripted transcription engine."""

import json
import math
import struct
import subprocess
import wave
from pathlib import Path

from isl_gloss.youtube.audio import FFmpeg
from isl_gloss.youtube.config import IngestConfig
from isl_gloss.youtube.downloader import YouTubeDownloader
from isl_gloss.youtube.pipeline import Components
from isl_gloss.youtube.transcribe import EngineResult, Segment, Word

VIDEO_ID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def make_wav(path, seconds, rate=16000, tone=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        for start in range(0, n, rate):
            count = min(rate, n - start)
            w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * (start + i) / rate)) if tone else 0) for i in range(count)))
    return path


def words(spec):
    """[(text, start, end), ...] -> Word list (texts get the leading space Whisper uses)."""
    return [Word(s, e, " " + t, 0.9) for t, s, e in spec]


def seg(start, end, text, word_spec=None):
    return Segment(start, end, text, words(word_spec) if word_spec else None)


class FakeYDL:
    """yt-dlp stand-in. `info` is returned by extract_info; download() writes a small file. Records every call."""

    def __init__(self, info=None, extract_error=None, download_error=None, log=None):
        self.info = info if info is not None else {"id": VIDEO_ID, "title": "Test talk", "duration": 12.0, "channel": "Test channel", "availability": "public",
                                                   "live_status": "not_live", "upload_date": "20240101"}
        self.extract_error, self.download_error = extract_error, download_error
        self.calls = log if log is not None else []

    def __call__(self, options):
        self.options = options
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        self.calls.append(("extract_info", url, download))
        if self.extract_error:
            raise self.extract_error
        return self.info

    def download(self, urls):
        self.calls.append(("download", tuple(urls)))
        if self.download_error:
            raise self.download_error
        out = Path(self.options["outtmpl"].replace("%(ext)s", "webm"))
        out.write_bytes(b"fake-audio")
        for hook in self.options.get("progress_hooks", []):
            hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
        return 0


class FakeRunner:
    """FFmpeg/ffprobe stand-in. ffprobe reports `probe`; ffmpeg writes a real silent/tone WAV of `seconds` to its output path."""

    def __init__(self, seconds=12.0, has_audio=True, fail_convert=None, probe_fail=False, missing=False):
        self.seconds, self.has_audio, self.fail_convert, self.probe_fail, self.missing = seconds, has_audio, fail_convert, probe_fail, missing
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        if self.missing:
            raise FileNotFoundError(cmd[0])
        name = Path(cmd[0]).name
        if "-version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, f"{name} version fake\n", "")
        if name.startswith("ffprobe"):
            if self.probe_fail:
                return subprocess.CompletedProcess(cmd, 1, "", "Invalid data found when processing input")
            streams = [{"codec_type": "audio", "codec_name": "opus", "duration": str(self.seconds), "start_time": "0.000000"}] if self.has_audio else [
                {"codec_type": "video", "codec_name": "vp9"}]
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"streams": streams, "format": {"duration": str(self.seconds), "format_name": "matroska,webm"}}), "")
        if self.fail_convert:
            return subprocess.CompletedProcess(cmd, 1, "", self.fail_convert)
        make_wav(cmd[-1], self.seconds)
        return subprocess.CompletedProcess(cmd, 0, "", "")


class ScriptedEngine:
    """Returns the given EngineResults in order (one per window) and records the windows it was given."""

    name = "scripted"
    model_name = "scripted-model"

    def __init__(self, results=None, error=None):
        self.results, self.error, self.windows = list(results or []), error, []
        self.calls = 0

    def transcribe_window(self, wav_path, language):
        self.calls += 1
        with wave.open(str(wav_path), "rb") as w:
            self.windows.append(w.getnframes() / w.getframerate())
        if self.error:
            raise self.error
        return self.results.pop(0) if self.results else EngineResult([])

    def describe(self):
        return {"engine": self.name, "model": self.model_name}


def components(ydl=None, runner=None, engine=None, cfg=None):
    cfg = cfg or IngestConfig()
    ydl = ydl or FakeYDL()
    runner = runner or FakeRunner()
    eng = engine or ScriptedEngine([EngineResult([seg(0.4, 3.2, "We will not go outside unless it stops raining.",
                                                      [("We", 0.4, 0.6), ("will", 0.6, 0.8), ("not", 0.8, 1.0), ("go", 1.0, 1.2), ("outside", 1.2, 1.8),
                                                       ("unless", 1.8, 2.2), ("it", 2.2, 2.4), ("stops", 2.4, 2.8), ("raining.", 2.8, 3.2)]),
                                                 seg(5.0, 8.1, "Why do we get angry so quickly?"),
                                                 seg(9.0, 11.5, "If you practise every day, your mind will become calm.")], "en", 0.99)])
    return Components(YouTubeDownloader(cfg, ydl_factory=ydl, log=lambda m: None), FFmpeg(runner=runner), engine_factory=lambda c: eng), ydl, runner, eng
