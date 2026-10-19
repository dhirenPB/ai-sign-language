"""Phase C: timestamped speech transcription behind a replaceable engine interface.

Timeline rules (the point of this module)
* Every timestamp it returns is in SECONDS FROM THE START OF THE ORIGINAL AUDIO (= the video), never relative to a window.
* Long audio is cut into windows (`chunk_seconds`) that overlap (`overlap_seconds`) so no word is split at a boundary. A segment
  is accepted from the window in which it STARTS; whatever is left of a duplicate across the boundary is removed on merge:
  word by word when the engine gives word timings, otherwise the duplicate segment is dropped or flagged. Timings are never
  interpolated, evenly spread, or invented.
* Word-level timings are used only if the engine returns them. A segment without words keeps `words = None`, and the overall
  `timestamp_level` says "word", "segment" or "mixed" truthfully.
* No engine is substituted silently: a missing or failing engine is an error.
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .audio import extract_wav_window, wav_info
from .config import IngestConfig
from .errors import DependencyMissingError, NoSpeechError, TranscriptionError

EPS = 0.05  # seconds of timing noise tolerated when comparing neighbouring timestamps


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: Optional[float] = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[list[Word]] = None


@dataclass
class EngineResult:
    segments: list[Segment]  # times LOCAL to the window the engine was given
    language: Optional[str] = None
    language_probability: Optional[float] = None


class TranscriptionEngine(Protocol):
    name: str
    model_name: str

    def transcribe_window(self, wav_path: Path, language: Optional[str]) -> EngineResult: ...

    def describe(self) -> dict[str, Any]: ...


# --- engines ------------------------------------------------------------------------------------------------------------
FASTER_WHISPER_HELP = ("Install faster-whisper: python -m pip install faster-whisper   (or: pip install -e .[youtube]). "
                       "The model is downloaded on first use unless allow_model_download is false.")


class FasterWhisperEngine:
    """faster-whisper (CTranslate2 Whisper): good English accuracy on CPU, bounded memory, word timestamps via cross-attention.
    Word timestamps are an estimate (typically within a fraction of a second), not forced alignment."""

    name = "faster-whisper"

    def __init__(self, cfg: IngestConfig, model_factory: Optional[Callable[..., Any]] = None):
        self.cfg, self._factory, self._model = cfg, model_factory, None
        self.model_name = cfg.model

    def _load(self):
        if self._model is not None:
            return self._model
        factory = self._factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel  # noqa: WPS433
            except ImportError as exc:
                raise DependencyMissingError("The faster-whisper package is not installed.", hint=FASTER_WHISPER_HELP) from exc
            factory = WhisperModel
        kwargs: dict[str, Any] = {"device": self.cfg.device, "compute_type": self.cfg.compute_type,
                                  "local_files_only": not self.cfg.allow_model_download}
        if self.cfg.model_dir:
            kwargs["download_root"] = self.cfg.model_dir
        try:
            self._model = factory(self.cfg.model, **kwargs)
        except Exception as exc:
            raise TranscriptionError(f"Could not load the Whisper model {self.cfg.model!r}: {str(exc)[:300]}",
                                     hint="If this computer is offline, the model must already be downloaded; otherwise check the model name, "
                                          "disk space and internet connection.", stage="transcription") from exc
        return self._model

    def transcribe_window(self, wav_path: Path, language: Optional[str]) -> EngineResult:
        model = self._load()
        try:
            segments, info = model.transcribe(str(wav_path), language=language, beam_size=self.cfg.beam_size, word_timestamps=self.cfg.word_timestamps,
                                              vad_filter=self.cfg.vad_filter, condition_on_previous_text=self.cfg.condition_on_previous_text)
            out = []
            for s in segments:  # a generator: segments are produced one at a time
                words = [Word(float(w.start), float(w.end), str(w.word), getattr(w, "probability", None)) for w in (getattr(s, "words", None) or [])]
                out.append(Segment(float(s.start), float(s.end), str(s.text), words or None))
        except Exception as exc:
            raise TranscriptionError(f"The transcription engine failed: {str(exc)[:300]}", stage="transcription") from exc
        return EngineResult(out, getattr(info, "language", None), getattr(info, "language_probability", None))

    def describe(self) -> dict[str, Any]:
        def ver(*names):
            for n in names:
                try:
                    return metadata.version(n)
                except metadata.PackageNotFoundError:
                    continue
            return None
        return {"engine": self.name, "model": self.cfg.model, "device": self.cfg.device, "compute_type": self.cfg.compute_type,
                "faster_whisper_version": ver("faster-whisper"), "ctranslate2_version": ver("ctranslate2")}


ENGINES: dict[str, Callable[[IngestConfig], TranscriptionEngine]] = {"faster-whisper": FasterWhisperEngine}


def make_engine(cfg: IngestConfig) -> TranscriptionEngine:
    try:
        factory = ENGINES[cfg.engine]
    except KeyError:
        raise TranscriptionError(f"Unknown transcription engine {cfg.engine!r}.", hint=f"Available engines: {', '.join(sorted(ENGINES))}. "
                                 "No other engine is substituted automatically.", stage="setup") from None
    return factory(cfg)


# --- windows --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    index: int
    accept_start: float  # segments that START in [accept_start, accept_end) belong to this window
    accept_end: float
    audio_start: float  # what the engine is actually given (includes the overlap)
    audio_end: float


def plan_windows(total: float, chunk: float, overlap: float) -> list[Window]:
    if total <= chunk:
        return [Window(0, 0.0, math.inf, 0.0, total)]
    n = math.ceil(total / chunk)
    out = []
    for i in range(n):
        a, b = i * chunk, (i + 1) * chunk
        last = i == n - 1
        out.append(Window(i, a, math.inf if last else b, max(0.0, a - overlap), total if last else min(total, b + overlap)))
    return out


# --- helpers ----------------------------------------------------------------------------------------------------------------
def join_words(words: list[Word]) -> str:
    sep = "" if any(w.text[:1].isspace() for w in words) else " "
    return " ".join(sep.join(w.text for w in words).split())


def _shift(seg: Segment, offset: float) -> Segment:
    words = [Word(w.start + offset, w.end + offset, w.text, w.probability) for w in seg.words] if seg.words else None
    return Segment(seg.start + offset, seg.end + offset, seg.text, words)


def _sanitise(seg: Segment, total: float, warnings: list[dict]) -> Optional[Segment]:
    """Drop what is invalid; never repair a time by guessing."""
    text = " ".join(seg.text.split())
    if not text:
        return None
    if not (math.isfinite(seg.start) and math.isfinite(seg.end)) or seg.end < seg.start - EPS:
        warnings.append({"code": "segment_dropped_invalid_timing", "message": f"A segment {seg.start}-{seg.end} had invalid timing and was dropped: {text[:60]!r}"})
        return None
    if seg.start > total + 1.0:
        warnings.append({"code": "segment_dropped_beyond_audio", "message": f"A segment starting at {seg.start:.2f}s is beyond the {total:.2f}s of audio and was dropped."})
        return None
    words = seg.words
    if words and any((not (math.isfinite(w.start) and math.isfinite(w.end))) or w.end < w.start - EPS for w in words):
        warnings.append({"code": "words_dropped_invalid_timing", "message": f"Word timings of a segment at {seg.start:.2f}s were invalid; this segment keeps segment-level timing only."})
        words = None
    start, end = max(0.0, seg.start), max(seg.start, seg.end)
    if end > total + EPS:
        warnings.append({"code": "end_clamped_to_audio_length", "message": f"A segment ending at {end:.2f}s was clamped to the audio length {total:.2f}s."})
        end = total
    return Segment(start, end, text, words)


def merge_segments(segments: list[Segment], warnings: list[dict]) -> list[Segment]:
    """Remove what the overlapping windows duplicated. Keeps chronological order and never invents a timing."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        if out and seg.start < out[-1].end - EPS:
            prev = out[-1]
            if seg.words:
                kept = [w for w in seg.words if w.start >= prev.end - EPS]
                if not kept:
                    warnings.append({"code": "boundary_duplicate_dropped", "message": f"A duplicate segment at {seg.start:.2f}s was dropped after a window boundary."})
                    continue
                if len(kept) < len(seg.words):
                    warnings.append({"code": "boundary_words_trimmed", "message": f"{len(seg.words) - len(kept)} duplicated word(s) at {seg.start:.2f}s were removed after a window boundary."})
                    seg = Segment(kept[0].start, max(seg.end, kept[-1].end), join_words(kept), kept)
            elif seg.end <= prev.end + EPS:
                warnings.append({"code": "boundary_duplicate_dropped", "message": f"A segment at {seg.start:.2f}s lay inside the previous one and was dropped."})
                continue
            else:  # partial overlap without word timings: keep as is, tell the reader; do not cut text by guessing
                warnings.append({"code": "segment_overlaps_previous", "message": f"Segment at {seg.start:.2f}s overlaps the previous segment (ends {prev.end:.2f}s); "
                                 "without word timings it cannot be trimmed, so both are kept unchanged."})
        out.append(seg)
    return out


# --- the transcription run ---------------------------------------------------------------------------------------------------
@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: Optional[str]
    language_probability: Optional[float]
    timestamp_level: str  # word | segment | mixed
    audio_seconds: float
    windows: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)


def transcribe_audio(wav_path: Path, cfg: IngestConfig, engine: TranscriptionEngine, log: Callable[[str], None] = print,
                     work_dir: Optional[Path] = None) -> TranscriptionResult:
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise TranscriptionError(f"The audio file {wav_path.name} does not exist.")
    info = wav_info(wav_path)
    total = info["duration_seconds"]
    if total <= 0:
        raise TranscriptionError("The audio is empty (zero length), so there is nothing to transcribe.",
                                 hint="Check that the video has an audible track; delete the job folder's audio/ and run again.")
    windows = plan_windows(total, cfg.chunk_seconds, cfg.overlap_seconds)
    warnings: list[dict] = []
    accepted: list[Segment] = []
    detected: list[tuple[str, float]] = []
    reports: list[dict] = []
    tmp_ctx = tempfile.TemporaryDirectory(dir=str(work_dir) if work_dir else None)
    with tmp_ctx as tmp:
        for w in windows:
            single = len(windows) == 1
            path = wav_path
            if not single:
                path = Path(tmp) / f"window_{w.index:04d}.wav"
                extract_wav_window(wav_path, w.audio_start, w.audio_end - w.audio_start, path)
            log(f"  transcribing window {w.index + 1}/{len(windows)}  ({w.audio_start:7.1f}s - {w.audio_end:7.1f}s of {total:.1f}s)")
            res = engine.transcribe_window(path, cfg.language)
            if not single:
                Path(path).unlink(missing_ok=True)
            if res.language:
                detected.append((res.language, res.language_probability or 0.0))
            kept = 0
            for seg in res.segments:
                g = _shift(seg, w.audio_start)  # window-local time -> original video time
                g = _sanitise(g, total, warnings)
                if g is None or not (w.accept_start <= g.start < w.accept_end):
                    continue
                accepted.append(g)
                kept += 1
            reports.append({"index": w.index, "audio_start": round(w.audio_start, 3), "audio_end": round(w.audio_end, 3),
                            "accepted_from_start": round(w.accept_start, 3), "accepted_until": None if w.accept_end == math.inf else round(w.accept_end, 3),
                            "segments_from_engine": len(res.segments), "segments_accepted": kept})
    merged = merge_segments(accepted, warnings)
    if not merged:
        raise NoSpeechError("No speech was found in the audio.")
    levels = {bool(s.words) for s in merged}
    level = "word" if levels == {True} else "segment" if levels == {False} else "mixed"
    if cfg.word_timestamps and level != "word":
        warnings.append({"code": "word_timestamps_incomplete", "message": f"Word timestamps were requested but the engine returned them for "
                         f"{sum(bool(s.words) for s in merged)} of {len(merged)} segments; the transcript is marked '{level}'."})
    language = cfg.language or (Counter(l for l, _ in detected).most_common(1)[0][0] if detected else None)
    prob = max((p for l, p in detected if l == language), default=None) if detected else None
    return TranscriptionResult(merged, language, prob, level, total, reports, warnings)
