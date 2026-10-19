"""YouTube phase building blocks: URL validation, downloader, audio extraction, transcription and the transcript schema."""

import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from isl_gloss.youtube.audio import FFmpeg, extract_wav_window, wav_info
from isl_gloss.youtube.config import IngestConfig
from isl_gloss.youtube.downloader import YouTubeDownloader, classify_ytdlp_error
from isl_gloss.youtube.errors import (AudioConversionError, AuthRequiredError, DependencyMissingError, DownloadFailedError, InvalidUrlError,
                                      LiveStreamError, NetworkError, NoAudioError, NoSpeechError, RateLimitedError, TranscriptionError,
                                      VideoTooLongError, VideoUnavailableError)
from isl_gloss.youtube.transcribe import (ENGINES, EngineResult, FasterWhisperEngine, Segment, Word, make_engine, merge_segments, plan_windows,
                                          transcribe_audio)
from isl_gloss.youtube.transcript import build_transcript, format_timestamp, to_text, to_vtt, validate_transcript
from isl_gloss.youtube.urls import parse_youtube_url
from tests.youtube_helpers import VIDEO_ID, FakeRunner, FakeYDL, ScriptedEngine, make_wav, seg, words

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class UrlTests(unittest.TestCase):
    def test_accepted_formats(self):
        cases = [(f"https://www.youtube.com/watch?v={VIDEO_ID}", "watch"), (f"https://youtube.com/watch?v={VIDEO_ID}&t=42s", "watch"),
                 (f"https://m.youtube.com/watch?v={VIDEO_ID}", "watch"), (f"https://youtu.be/{VIDEO_ID}", "youtu.be"),
                 (f"https://youtu.be/{VIDEO_ID}?si=abc", "youtu.be"), (f"https://www.youtube.com/shorts/{VIDEO_ID}", "short"),
                 (f"https://www.youtube.com/embed/{VIDEO_ID}", "embed"), (f"  https://www.youtube.com/watch?v={VIDEO_ID}  ", "watch"),
                 (f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PL123", "watch")]
        for url, kind in cases:
            ref = parse_youtube_url(url)
            self.assertEqual((ref.video_id, ref.url_type), (VIDEO_ID, kind), url)
            self.assertEqual(ref.canonical_url, f"https://www.youtube.com/watch?v={VIDEO_ID}")

    def test_rejected_with_a_clear_message(self):
        bad = {"": "No URL", None: "No URL", "not a url": "http(s)", "ftp://www.youtube.com/watch?v=" + VIDEO_ID: "http(s)",
               "https://vimeo.com/12345": "not a YouTube", "https://youtube.com.evil.com/watch?v=" + VIDEO_ID: "not a YouTube",
               "https://www.youtube.com/watch": "no video id", "https://www.youtube.com/watch?v=short": "not a valid YouTube video id",
               "https://youtu.be/": "no video id", "https://www.youtube.com/watch?v=" + VIDEO_ID + "X": "not a valid",
               "https://www.youtube.com/watch?v=abc$%^&*()": "not a valid", "https://www.youtube.com/playlist?list=PL123": "playlist",
               "https://www.youtube.com/@somechannel": "channel", "https://www.youtube.com/": "home page",
               "https://www.youtube.com/results?search_query=x": "search", "https://www.youtube.com/shorts/": "no video id",
               "https://user:pw@www.youtube.com/watch?v=" + VIDEO_ID: "user name", "https://www.youtube.com/feed/trending": "page"}
        for url, expected in bad.items():
            with self.assertRaises(InvalidUrlError, msg=str(url)) as cm:
                parse_youtube_url(url)
            self.assertIn(expected.lower(), (cm.exception.message + " " + (cm.exception.hint or "")).lower(), url)
            self.assertEqual(cm.exception.stage, "validation")


class ErrorClassificationTests(unittest.TestCase):
    def test_messages_map_to_specific_errors(self):
        cases = [("ERROR: [youtube] x: Private video. Sign in if you've been granted access", VideoUnavailableError, "private"),
                 ("ERROR: Video unavailable", VideoUnavailableError, "unavailable"),
                 ("ERROR: This video has been removed by the uploader", VideoUnavailableError, "removed"),
                 ("ERROR: The uploader has not made this video available in your country", VideoUnavailableError, "region_blocked"),
                 ("ERROR: Sign in to confirm your age. This video may be inappropriate for some users.", AuthRequiredError, "age_restricted"),
                 ("ERROR: Join this channel to get access to members-only content", AuthRequiredError, "members_only"),
                 ("ERROR: Sign in to confirm you're not a bot", AuthRequiredError, "bot_check"),
                 ("ERROR: This live event will begin in 2 hours", LiveStreamError, None),
                 ("HTTP Error 429: Too Many Requests", RateLimitedError, None),
                 ("<urlopen error [Errno 11001] getaddrinfo failed>", NetworkError, None),
                 ("something nobody has seen before", DownloadFailedError, None)]
        for message, cls, reason in cases:
            err = classify_ytdlp_error(message)
            self.assertIsInstance(err, cls, message)
            if reason:
                self.assertEqual(err.details.get("reason"), reason)
            self.assertIn("yt_dlp", err.details) if cls is not LiveStreamError else None

    def test_auth_errors_say_that_access_is_not_bypassed(self):
        self.assertIn("does not sign in or bypass", classify_ytdlp_error("Sign in to confirm your age").hint)


class DownloaderTests(unittest.TestCase):
    def dl(self, ydl, **cfg):
        return YouTubeDownloader(IngestConfig(**cfg), ydl_factory=ydl, log=lambda m: None)

    def test_metadata(self):
        info = self.dl(FakeYDL()).fetch_info(parse_youtube_url(f"https://youtu.be/{VIDEO_ID}"))
        self.assertEqual((info.video_id, info.title, info.duration_seconds, info.canonical_url), (VIDEO_ID, "Test talk", 12.0, f"https://www.youtube.com/watch?v={VIDEO_ID}"))
        self.assertEqual(info.url, f"https://youtu.be/{VIDEO_ID}")

    def test_inaccessible_videos_are_classified_and_not_worked_around(self):
        ref = parse_youtube_url(f"https://youtu.be/{VIDEO_ID}")
        for exc, cls in ((Exception("ERROR: Private video"), VideoUnavailableError), (Exception("Video unavailable"), VideoUnavailableError),
                         (Exception("Sign in to confirm your age"), AuthRequiredError), (Exception("HTTP Error 429"), RateLimitedError),
                         (Exception("getaddrinfo failed"), NetworkError)):
            with self.assertRaises(cls):
                self.dl(FakeYDL(extract_error=exc)).fetch_info(ref)
        for info, cls in (({"id": VIDEO_ID, "is_live": True}, LiveStreamError), ({"id": VIDEO_ID, "live_status": "is_upcoming"}, LiveStreamError),
                          ({"id": VIDEO_ID, "availability": "private"}, VideoUnavailableError), ({"id": VIDEO_ID, "availability": "needs_auth"}, AuthRequiredError),
                          ({"id": VIDEO_ID, "duration": 999999}, VideoTooLongError)):
            with self.assertRaises(cls):
                self.dl(FakeYDL(info=info)).fetch_info(ref)

    def test_options_never_carry_credentials_and_download_is_audio_only(self):
        ydl = FakeYDL()
        with tempfile.TemporaryDirectory() as d:
            path = self.dl(ydl).download_audio(parse_youtube_url(f"https://youtu.be/{VIDEO_ID}"), Path(d))
            self.assertTrue(path.is_file())
        self.assertEqual(ydl.options["format"], "bestaudio/best")
        self.assertTrue(ydl.options["noplaylist"])
        for forbidden in ("cookiefile", "cookiesfrombrowser", "username", "password", "proxy", "geo_bypass", "geo_bypass_country"):
            self.assertNotIn(forbidden, ydl.options)

    def test_download_failure_and_no_file(self):
        ref = parse_youtube_url(f"https://youtu.be/{VIDEO_ID}")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NetworkError):
                self.dl(FakeYDL(download_error=Exception("Connection reset by peer"))).download_audio(ref, Path(d))

            class Empty(FakeYDL):
                def download(self, urls):
                    return 0
            with self.assertRaises(DownloadFailedError):
                self.dl(Empty()).download_audio(ref, Path(d))

    def test_missing_yt_dlp_is_an_actionable_error(self):
        import sys
        sys.modules["yt_dlp"] = None  # makes `import yt_dlp` raise ImportError
        try:
            with self.assertRaises(DependencyMissingError) as cm:
                YouTubeDownloader(IngestConfig(), log=lambda m: None).fetch_info(parse_youtube_url(f"https://youtu.be/{VIDEO_ID}"))
        finally:
            del sys.modules["yt_dlp"]
        self.assertIn("pip install yt-dlp", cm.exception.hint)


class AudioTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.dir / "in.webm").write_bytes(b"x")

    def test_successful_conversion_keeps_the_timeline(self):
        runner = FakeRunner(seconds=12.0)
        info = FFmpeg(runner=runner).to_wav(self.dir / "in.webm", self.dir / "audio.wav", 16000, 1)
        self.assertTrue(info["has_audio"])
        self.assertAlmostEqual(wav_info(self.dir / "audio.wav")["duration_seconds"], 12.0, places=2)
        cmd = next(c for c in runner.calls if "-ar" in c)
        for removed in ("silenceremove", "-t", "-ss", "atrim", "concat"):  # nothing that would shift or cut the timeline
            self.assertNotIn(removed, " ".join(map(str, cmd)).split() if removed.startswith("-") else " ".join(map(str, cmd)))
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
        self.assertFalse((self.dir / "audio.wav.part.wav").exists())

    def test_missing_audio_stream(self):
        with self.assertRaises(NoAudioError):
            FFmpeg(runner=FakeRunner(has_audio=False)).to_wav(self.dir / "in.webm", self.dir / "a.wav")
        with self.assertRaises(NoAudioError):
            FFmpeg(runner=FakeRunner(fail_convert="Output file #0 does not contain any stream")).to_wav(self.dir / "in.webm", self.dir / "a.wav")
        self.assertFalse((self.dir / "a.wav").exists())

    def test_conversion_failure_leaves_no_half_written_file(self):
        with self.assertRaises(AudioConversionError) as cm:
            FFmpeg(runner=FakeRunner(fail_convert="Unknown encoder 'xyz'")).to_wav(self.dir / "in.webm", self.dir / "a.wav")
        self.assertIn("Unknown encoder", cm.exception.details["stderr"])
        self.assertFalse((self.dir / "a.wav").exists())

    def test_unreadable_media_and_missing_ffmpeg(self):
        with self.assertRaises(AudioConversionError):
            FFmpeg(runner=FakeRunner(probe_fail=True)).to_wav(self.dir / "in.webm", self.dir / "a.wav")
        with self.assertRaises(DependencyMissingError) as cm:
            FFmpeg(runner=FakeRunner(missing=True)).to_wav(self.dir / "in.webm", self.dir / "a.wav")
        self.assertIn("FFmpeg", cm.exception.hint)
        with self.assertRaises(DependencyMissingError):
            FFmpeg(runner=FakeRunner(missing=True)).check()

    def test_wav_window_is_exact_and_bounded(self):
        make_wav(self.dir / "long.wav", 10.0)
        meta = extract_wav_window(self.dir / "long.wav", 2.5, 3.0, self.dir / "w.wav")
        self.assertEqual(meta, {"start_seconds": 2.5, "duration_seconds": 3.0})
        self.assertAlmostEqual(wav_info(self.dir / "w.wav")["duration_seconds"], 3.0, places=3)
        tail = extract_wav_window(self.dir / "long.wav", 9.0, 5.0, self.dir / "t.wav")  # asks past the end: clipped, not padded
        self.assertAlmostEqual(tail["duration_seconds"], 1.0, places=3)

    def test_invalid_wav(self):
        (self.dir / "bad.wav").write_bytes(b"not a wav")
        with self.assertRaises(AudioConversionError):
            wav_info(self.dir / "bad.wav")

    @unittest.skipUnless(HAVE_FFMPEG, "FFmpeg is not installed")
    def test_real_ffmpeg_converts_and_detects_a_missing_audio_stream(self):
        def run(*args):
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True, timeout=120)
        run("-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "libopus", str(self.dir / "tone.webm"))
        run("-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=5", "-c:v", "libvpx", str(self.dir / "silent_video.webm"))
        ff = FFmpeg()
        ff.check()
        info = ff.to_wav(self.dir / "tone.webm", self.dir / "tone.wav", 16000, 1)
        w = wav_info(self.dir / "tone.wav")
        self.assertEqual((w["sample_rate"], w["channels"]), (16000, 1))
        self.assertAlmostEqual(w["duration_seconds"], 3.0, delta=0.15)  # same length as the source: timeline preserved
        self.assertTrue(info["has_audio"])
        with self.assertRaises(NoAudioError):
            ff.to_wav(self.dir / "silent_video.webm", self.dir / "none.wav")
        (self.dir / "corrupt.webm").write_bytes(b"this is not media")
        with self.assertRaises(AudioConversionError):
            ff.to_wav(self.dir / "corrupt.webm", self.dir / "c.wav")


class WindowTests(unittest.TestCase):
    def test_single_window_for_short_audio(self):
        self.assertEqual(len(plan_windows(500, 600, 15)), 1)

    def test_windows_tile_the_timeline_with_overlap(self):
        ws = plan_windows(25, 10, 2)
        self.assertEqual([(w.accept_start, w.audio_start, w.audio_end) for w in ws], [(0, 0, 12), (10, 8, 22), (20, 18, 25)])
        self.assertEqual(ws[0].accept_end, 10)
        self.assertEqual(ws[-1].accept_end, float("inf"))


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.cfg = IngestConfig(chunk_seconds=30, overlap_seconds=2)

    def run_transcribe(self, seconds, results, cfg=None, tone=True):
        wav = make_wav(self.dir / "a.wav", seconds, tone=tone)
        eng = ScriptedEngine(results)
        return transcribe_audio(wav, cfg or self.cfg, eng, log=lambda m: None, work_dir=self.dir), eng

    def test_short_audio_keeps_global_times_and_word_level(self):
        res, eng = self.run_transcribe(12, [EngineResult([seg(1.0, 2.0, "Hello world.", [("Hello", 1.0, 1.4), ("world.", 1.5, 2.0)])], "en", 0.98)])
        self.assertEqual((len(eng.windows), res.timestamp_level, res.language), (1, "word", "en"))
        self.assertEqual((res.segments[0].start, res.segments[0].end), (1.0, 2.0))

    def test_segment_level_is_reported_as_segment_level(self):
        res, _ = self.run_transcribe(12, [EngineResult([seg(1.0, 2.0, "No word times here.")])])
        self.assertEqual(res.timestamp_level, "segment")
        self.assertIsNone(res.segments[0].words)  # words are never invented
        self.assertIn("word_timestamps_incomplete", [w["code"] for w in res.warnings])

    def test_mixed_level(self):
        res, _ = self.run_transcribe(12, [EngineResult([seg(1, 2, "A b.", [("A", 1, 1.4), ("b.", 1.5, 2)]), seg(3, 4, "C d.")])])
        self.assertEqual(res.timestamp_level, "mixed")

    def test_chunk_local_times_become_global_and_nothing_is_lost_or_duplicated(self):
        # 65 s audio, windows of 30 s with 2 s overlap: windows start at 0, 28, 58
        w0 = EngineResult([seg(1, 3, "First sentence."), seg(29.0, 31.5, "Crosses the boundary.", [("Crosses", 29.0, 29.9), ("the", 30.0, 30.4), ("boundary.", 30.5, 31.5)])])
        # window 1 starts at 28 s: it re-hears the boundary words (global 29-31.5) and continues
        w1 = EngineResult([seg(1.0, 3.5, "Crosses the boundary.", [("Crosses", 1.0, 1.9), ("the", 2.0, 2.4), ("boundary.", 2.5, 3.5)]),
                           seg(4.0, 6.0, "Second window speech.", [("Second", 4.0, 4.5), ("window", 4.5, 5.0), ("speech.", 5.0, 6.0)]),
                           seg(31.0, 33.0, "Third window overlap.", [("Third", 31.0, 31.6), ("window", 31.6, 32.2), ("overlap.", 32.2, 33.0)])])
        w2 = EngineResult([seg(1.0, 2.0, "Third window overlap.", [("Third", 1.0, 1.4), ("window", 1.4, 1.7), ("overlap.", 1.7, 2.0)]),
                           seg(5.0, 7.0, "The very end.", [("The", 5.0, 5.3), ("very", 5.3, 5.8), ("end.", 5.8, 7.0)])])
        res, eng = self.run_transcribe(65, [w0, w1, w2])
        self.assertEqual(len(eng.windows), 3)
        self.assertAlmostEqual(eng.windows[0], 30.0 + 2.0, places=2)  # first window: accept 0-30 plus right overlap
        self.assertEqual([(round(s.start, 2), s.text) for s in res.segments],
                         [(1.0, "First sentence."), (29.0, "Crosses the boundary."), (32.0, "Second window speech."), (59.0, "Third window overlap."),
                          (63.0, "The very end.")])
        joined = " ".join(s.text for s in res.segments)
        for phrase in ("First sentence.", "Crosses the boundary.", "Second window speech.", "Third window overlap.", "The very end."):
            self.assertEqual(joined.count(phrase), 1, f"{phrase!r} lost or duplicated in: {joined}")
        starts = [s.start for s in res.segments]
        self.assertEqual(starts, sorted(starts))  # chronological
        by = {s.text: s for s in res.segments}
        self.assertAlmostEqual(by["Second window speech."].start, 32.0)  # local 4.0 + window start 28.0
        self.assertAlmostEqual(by["The very end."].start, 63.0)  # local 5.0 + window start 58.0
        self.assertAlmostEqual(by["The very end."].words[0].start, 63.0)

    def test_boundary_word_duplicates_are_trimmed_not_invented(self):
        a = seg(9.5, 11.0, "alpha beta", [("alpha", 9.5, 10.0), ("beta", 10.1, 11.0)])
        b = seg(10.2, 13.0, "beta gamma", [("beta", 10.2, 11.0), ("gamma", 11.2, 13.0)])
        warnings = []
        merged = merge_segments([a, b], warnings)
        self.assertEqual([s.text for s in merged], ["alpha beta", "gamma"])
        self.assertEqual((merged[1].start, merged[1].end), (11.2, 13.0))  # taken from the word, not computed
        self.assertIn("boundary_words_trimmed", [w["code"] for w in warnings])

    def test_overlap_without_word_times_is_flagged_not_cut(self):
        warnings = []
        merged = merge_segments([Segment(1, 5, "one two three"), Segment(3, 7, "two three four")], warnings)
        self.assertEqual([s.text for s in merged], ["one two three", "two three four"])
        self.assertIn("segment_overlaps_previous", [w["code"] for w in warnings])
        dup = merge_segments([Segment(1, 5, "one two"), Segment(2, 4, "two")], [])
        self.assertEqual(len(dup), 1)

    def test_invalid_engine_timestamps_are_dropped_with_a_warning(self):
        res, _ = self.run_transcribe(12, [EngineResult([Segment(5, 3, "backwards"), Segment(float("nan"), 2, "nan"), Segment(50, 51, "after the end"),
                                                        Segment(1, 2, "   "), seg(2, 3, "valid one")])])
        self.assertEqual([s.text for s in res.segments], ["valid one"])
        codes = [w["code"] for w in res.warnings]
        self.assertIn("segment_dropped_invalid_timing", codes)
        self.assertIn("segment_dropped_beyond_audio", codes)

    def test_bad_word_times_fall_back_to_segment_level_for_that_segment(self):
        bad = Segment(1, 3, "ab cd", [Word(2.0, 1.0, " ab", 0.9), Word(2.0, 3.0, " cd", 0.9)])
        res, _ = self.run_transcribe(12, [EngineResult([bad])])
        self.assertIsNone(res.segments[0].words)
        self.assertIn("words_dropped_invalid_timing", [w["code"] for w in res.warnings])

    def test_segment_end_is_clamped_to_the_audio_length(self):
        res, _ = self.run_transcribe(10, [EngineResult([seg(8, 10.3, "tail")])])
        self.assertEqual(res.segments[0].end, 10.0)
        self.assertIn("end_clamped_to_audio_length", [w["code"] for w in res.warnings])

    def test_no_speech_empty_and_invalid_audio(self):
        with self.assertRaises(NoSpeechError):
            self.run_transcribe(12, [EngineResult([])])
        (self.dir / "empty.wav").write_bytes(b"")
        with self.assertRaises(Exception) as cm:
            transcribe_audio(self.dir / "empty.wav", self.cfg, ScriptedEngine(), log=lambda m: None)
        self.assertIsInstance(cm.exception, AudioConversionError)
        with wave.open(str(self.dir / "zero.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        with self.assertRaises(TranscriptionError) as cm2:
            transcribe_audio(self.dir / "zero.wav", self.cfg, ScriptedEngine(), log=lambda m: None)
        self.assertIn("empty", cm2.exception.message)
        with self.assertRaises(TranscriptionError):
            transcribe_audio(self.dir / "missing.wav", self.cfg, ScriptedEngine(), log=lambda m: None)

    def test_engine_failure_is_raised_not_swallowed(self):
        wav = make_wav(self.dir / "a.wav", 12)
        with self.assertRaises(TranscriptionError) as cm:
            transcribe_audio(wav, self.cfg, ScriptedEngine(error=TranscriptionError("boom")), log=lambda m: None)
        self.assertEqual(cm.exception.message, "boom")

    def test_window_files_are_cleaned_up(self):
        self.run_transcribe(65, [EngineResult([seg(1, 2, "x")]), EngineResult([]), EngineResult([])])
        self.assertEqual([p.name for p in self.dir.glob("window_*.wav")], [])


class EngineSelectionTests(unittest.TestCase):
    def test_unknown_engine_is_an_error_not_a_substitution(self):
        with self.assertRaises(TranscriptionError) as cm:
            make_engine(IngestConfig(engine="made-up"))
        self.assertIn("No other engine is substituted", cm.exception.hint)

    def test_faster_whisper_missing_is_actionable(self):
        import sys
        sys.modules["faster_whisper"] = None
        try:
            with self.assertRaises(DependencyMissingError) as cm:
                FasterWhisperEngine(IngestConfig()).transcribe_window(Path("x.wav"), "en")
        finally:
            del sys.modules["faster_whisper"]
        self.assertIn("pip install faster-whisper", cm.exception.hint)

    def test_model_load_failure_is_reported_and_not_replaced_by_another_model(self):
        calls = []

        def factory(name, **kw):
            calls.append(name)
            raise OSError("no such model offline")
        with self.assertRaises(TranscriptionError) as cm:
            FasterWhisperEngine(IngestConfig(model="small.en", allow_model_download=False), model_factory=factory).transcribe_window(Path("x.wav"), "en")
        self.assertEqual(calls, ["small.en"])
        self.assertIn("offline", cm.exception.hint)

    def test_faster_whisper_adapter_maps_results_and_passes_settings(self):
        class W:
            def __init__(self, s, e, w, p): self.start, self.end, self.word, self.probability = s, e, w, p

        class S:
            def __init__(self, s, e, t, ws): self.start, self.end, self.text, self.words = s, e, t, ws

        class Info: language, language_probability = "en", 0.97

        class Model:
            def __init__(self, name, **kw): self.name, self.kw = name, kw

            def transcribe(self, path, **kw):
                self.seen = kw
                return iter([S(0.5, 1.5, " Hi there.", [W(0.5, 0.9, " Hi", 0.9), W(1.0, 1.5, " there.", 0.8)]), S(2.0, 3.0, " No words.", None)]), Info()

        holder = {}

        def factory(name, **kw):
            holder["m"] = Model(name, **kw)
            return holder["m"]
        cfg = IngestConfig(model="base.en", device="cpu", compute_type="int8", language="en", vad_filter=True, allow_model_download=False)
        res = FasterWhisperEngine(cfg, model_factory=factory).transcribe_window(Path("x.wav"), "en")
        self.assertEqual((res.language, res.language_probability), ("en", 0.97))
        self.assertEqual([s.text for s in res.segments], [" Hi there.", " No words."])
        self.assertEqual(len(res.segments[0].words), 2)
        self.assertIsNone(res.segments[1].words)
        self.assertEqual(holder["m"].kw["local_files_only"], True)
        self.assertEqual((holder["m"].seen["word_timestamps"], holder["m"].seen["vad_filter"], holder["m"].seen["language"]), (True, True, "en"))

    def test_engine_registry_is_replaceable(self):
        ENGINES["scripted"] = lambda cfg: ScriptedEngine()
        try:
            self.assertEqual(make_engine(IngestConfig(engine="scripted")).name, "scripted")
        finally:
            del ENGINES["scripted"]


class TranscriptSchemaTests(unittest.TestCase):
    def build(self, segments, level="word"):
        from isl_gloss.youtube.transcribe import TranscriptionResult
        res = TranscriptionResult(segments, "en", 0.99, level, 30.0, [], [])
        return build_transcript("yt-x", {"video_id": VIDEO_ID, "duration_seconds": 30.0}, {"duration_seconds": 30.0}, res, {"engine": "scripted"}, {"chunk_seconds": 600, "overlap_seconds": 15})

    def test_structure_and_derived_files(self):
        doc = self.build([seg(12.4, 13.1, "I want", [("I", 12.4, 12.6), ("want", 12.7, 13.1)]), seg(13.1, 15.8, "to explain this concept.", [("to", 13.1, 13.3), ("explain", 13.3, 14.0),
                                                                                                                                            ("this", 14.0, 14.4), ("concept.", 14.4, 15.8)])])
        t = doc["transcription"]
        self.assertEqual((doc["job_id"], doc["review_status"], t["engine"], t["timestamp_level"], t["language"]), ("yt-x", "unreviewed", "scripted", "word", "en"))
        self.assertEqual([s["id"] for s in t["segments"]], ["seg-0001", "seg-0002"])
        self.assertEqual((t["segments"][0]["start"], t["segments"][1]["end"]), (12.4, 15.8))
        self.assertEqual(t["segments"][0]["words"][1], {"start": 12.7, "end": 13.1, "text": "want", "probability": 0.9})
        self.assertIn("seconds from the start of the original YouTube video", t["timeline"])
        self.assertEqual(validate_transcript(doc), [])
        vtt = to_vtt(doc)
        self.assertTrue(vtt.startswith("WEBVTT\n\nseg-0001\n00:00:12.400 --> 00:00:13.100\nI want\n"))
        self.assertEqual(to_text(doc), "I want\nto explain this concept.\n")
        json.dumps(doc)  # serialisable

    def test_validation_catches_timeline_problems(self):
        doc = self.build([seg(5, 6, "a b", [("a", 5, 5.5), ("b", 5.5, 6)]), seg(1, 2, "late", [("late", 1, 2)])])
        problems = " | ".join(validate_transcript(doc))
        self.assertIn("not chronological", problems)
        doc = self.build([seg(5, 4, "backwards")])
        self.assertIn("invalid interval", " ".join(validate_transcript(doc)))
        doc = self.build([seg(25, 99, "past the end")])
        self.assertIn("after the audio ends", " ".join(validate_transcript(doc)))
        doc = self.build([seg(1, 2, "x")], level="word")
        self.assertIn("timestamp_level", " ".join(validate_transcript(doc)))
        self.assertTrue(validate_transcript({"transcription": {"segments": []}}))
        self.assertIn("outside or out of order", " ".join(validate_transcript(self.build([seg(5, 6, "a b", [("a", 5, 5.5), ("b", 9, 9.5)])]))))

    def test_timestamp_format(self):
        self.assertEqual(format_timestamp(0), "00:00:00.000")
        self.assertEqual(format_timestamp(3725.4567), "01:02:05.457")


if __name__ == "__main__":
    unittest.main()
