# YouTube ingestion: URL → audio → timestamped transcript → existing pipeline

**Status:** implemented. Routine tests use doubles for YouTube and the speech model, and the *real* cleaner, `translate` command and FFmpeg.
On top of that, the real chain (real FFmpeg, real `faster-whisper` `small.en`, real cleaner, real `translate` with a live Groq request) was run
on **locally synthesised English speech** (Windows voice; only the YouTube download was replaced by a file copy): see "Measured so far". It has
**not** yet been run on a real YouTube video with real `yt-dlp` downloading: see "Manual smoke test". Everything it outputs is unreviewed machine output.

## Measured so far (synthetic speech, not real-world audio)

| Check | Result |
|---|---|
| Short run: 23 s, 5 sentences, 3 pauses of 1.5 s | 3 Whisper segments, word-level timestamps; real `translate` glossed all 5 (1 live request); a re-run reused every stage and took 3 s |
| Long run: 71 s of near-continuous speech, forced into three 30 s windows (boundaries fell mid-speech) | 16 of 16 sentences present **exactly once**, in order; no loss or duplication at either boundary |
| Word times against the audio itself (energy): 207 words in total | 206 of 207 have sound inside their interval (the other at least partly); under 1% of sound lies more than 150 ms from any word |
| First word | stamped 0.00 s while sound begins about 0.12 s: Whisper tends to start the first word at 0 |
| Speed (CPU, model already downloaded) | 71 s of audio in about 59 s end to end (slightly faster than real time); a cold first run also downloads the ~480 MB model |
| Spelling/numbers | Whisper normalises: "practise" came out as "practice" and "3" as "three" |

This shows the timeline and windowing logic works on real engine output. It does **not** show accuracy on real speech (accents, noise, music, overlapping voices).

## Architecture

```
YouTube URL
 → validate (urls.py)                 id extracted, malformed / unsupported links rejected, no network
 → metadata (downloader.py, yt-dlp)   title, duration; private / removed / age-gated / live / too long are classified, never bypassed
 → audio-only download + FFmpeg        16 kHz mono WAV, timeline untouched (audio.py)
 → timestamped transcript              faster-whisper in overlapping windows, back on the global timeline (transcribe.py)
 → transcript.json (CANONICAL) + transcript.txt + transcript.vtt             (transcript.py)
 → existing cleaner          (reads the VTT: timestamps + segment ids become chunk times + cue_ids)
 → existing `translate`      (cleaning → sentences → groups → LLM → rules → validation; optional --layer2)
 → timeline.json             every sentence/group/gloss mapped back to seconds of the original video (align.py, pipeline.py)
```

What already existed and was **not** modified: the cleaner (it already parses timestamped VTT/SRT and records `cue_ids`), the sentence
splitter, grouping, the LLM translator and prompts, the rule pass, spaCy Layer 2, the selection policy and gloss notation. The only
edit to existing code is registering the new commands in `cli.py`. A transcript supplied directly (the old path) is untouched
and still works (tested).

## Files (`src/isl_gloss/youtube/`)

| File | Purpose |
|---|---|
| `urls.py` | Accept `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`; reject everything else with a specific message |
| `downloader.py` | yt-dlp wrapper (injectable): metadata, audio-only download, error classification |
| `audio.py` | FFmpeg conversion/probe (injectable runner); pure-Python exact WAV windows |
| `transcribe.py` | Engine interface + faster-whisper engine, windowing, local→global times, boundary de-duplication |
| `transcript.py` | Canonical schema, validator, VTT and plain-text derivation |
| `align.py` | Match cleaned sentences back to transcript words/segments (heuristic, labelled) |
| `jobs.py` | File-based job record (`job.json`, `job.log`), states, atomic writes |
| `pipeline.py` | The orchestration, resume logic, timeline builder, job viewer |
| `cli.py`, `config.py`, `errors.py` | Commands, `IngestConfig`, typed errors with stage + hint |

## Library choices

* **yt-dlp** (downloader): the maintained successor of youtube-dl; audio-only `bestaudio`; no cookies, accounts or workarounds are configured.
* **FFmpeg** (already installed here, 8.1.1): conversion to WAV. WAV is read at exact offsets, so long audio is windowed in pure Python without loading it whole.
* **faster-whisper** (default engine): Whisper accuracy on CPU through CTranslate2 (int8) with bounded memory, and *word timestamps* (Whisper's cross-attention estimates; not forced alignment). Alternatives weighed: `openai-whisper` (needs PyTorch, much heavier on a CPU-only machine with limited disk), `whisper.cpp` (needs a build step). No paid API is used.
* **Default model `small.en`** (about 480 MB, downloaded on first use): a sensible accuracy/CPU trade-off for English. Use `base.en` for speed or `medium.en` for accuracy. Real-world speed on this machine has **not** been measured.
* The engine is behind an interface (`transcribe.ENGINES`); a failing or missing engine is an error, never silently replaced.

## Install

```bash
.venv/Scripts/python.exe -m pip install yt-dlp faster-whisper      # or: pip install -e .[youtube]
winget install Gyan.FFmpeg                                         # FFmpeg: already present on this machine
.venv/Scripts/python.exe run_translator.py youtube-check           # shows what is installed
```
The Whisper model downloads automatically on first use (`allow_model_download: false` forbids that). yt-dlp must be kept up to date.

## Run

```bash
# the whole path (gloss stage limited to the first 10 sentences by default)
.venv/Scripts/python.exe run_translator.py youtube "https://www.youtube.com/watch?v=VIDEO_ID"
# stop earlier / gloss everything / pick a model / add the spaCy candidate
... youtube URL --through transcript
... youtube URL --gloss-limit 0 --model base.en --layer2
# inspect (reads saved files only)
... youtube-show URL            # status + timestamped transcript + gloss by original-video time
... youtube-show URL --what transcript
... youtube-jobs
```
Exit codes: 0 completed, 1 job failed (stage and hint printed; `job.json` has them), 2 bad input. Re-running the same command resumes:
finished stages are reused, a failed stage is retried, and `--force audio|transcript|all` redoes one deliberately. `--through transcript`
never calls the LLM; the LLM is reached only by the existing `translate` command after a valid transcript.

## Configuration (`IngestConfig`, JSON via `--ingest-config`, or flags)

`data_dir` (default `data/jobs`), `keep_source_audio`, `max_duration_seconds` (3 h), `download_retries`, `socket_timeout_seconds`,
`ffmpeg_path`/`ffprobe_path`, `sample_rate` (16000), `channels` (1), `engine` (`faster-whisper`), `model` (`small.en`), `language`
(`en`, or null to detect), `device` (`cpu`), `compute_type` (`int8`), `beam_size`, `word_timestamps` (true), `vad_filter` (true; skips
silence inside the engine and restores the original times), `condition_on_previous_text` (false), `allow_model_download`, `model_dir`,
`chunk_seconds` (600) and `overlap_seconds` (15). The LLM settings are the existing ones (`--llm-config`, `--prompt-version`, `.env`).

## Job folder (`data/jobs/yt-<VIDEO_ID>/`, git-ignored)

`job.json` (state, per-stage status, error with stage/kind/hint, output paths) · `job.log` · `metadata/video.json` · `audio/audio.wav` + `audio.json` ·
`transcript/transcript.json` (canonical) `.txt` `.vtt` · `cleaning/` (the cleaner's own folders) · `gloss/` (the translator's own output) · `timeline/timeline.json`.
States: `PENDING → DOWNLOADING → AUDIO_EXTRACTED → TRANSCRIBING → TRANSCRIBED → PROCESSING_EXISTING_PIPELINE → COMPLETED`, or `FAILED` with the stage recorded.
The cleaner and every sentence id use a lowercase slug of the video id (`yt-dqw4w9wgxcq-s0001`), because the cleaner's schema requires lowercase; the exact id is in `job.json`.

## Transcript schema (excerpt of a real file produced by the pipeline with a scripted engine, not a real ASR run)

```json
{
  "schema_version": "1.0", "report_type": "youtube_transcript", "job_id": "yt-dQw4w9WgXcQ", "review_status": "unreviewed",
  "video": {"video_id": "dQw4w9WgXcQ", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Test talk", "duration_seconds": 12.0},
  "audio": {"path": "audio/audio.wav", "format": "wav", "sample_rate": 16000, "channels": 1, "duration_seconds": 12.0, "timeline_preserved": true},
  "transcription": {
    "engine": "faster-whisper", "language": "en", "timestamp_level": "mixed",
    "timeline": "All times are seconds from the start of the original YouTube video.",
    "chunking": {"windows": [{"index": 0, "audio_start": 0.0, "audio_end": 12.0, "segments_accepted": 3}], "window_seconds": 600.0, "overlap_seconds": 15.0},
    "warnings": [{"code": "word_timestamps_incomplete", "message": "..."}],
    "segments": [
      {"id": "seg-0001", "start": 0.4, "end": 3.2, "text": "We will not go outside unless it stops raining.",
       "words": [{"start": 0.4, "end": 0.6, "text": "We", "probability": 0.9}, "..."]},
      {"id": "seg-0002", "start": 5.0, "end": 8.1, "text": "Why do we get angry so quickly?", "words": null}
    ]
  }
}
```
`timestamp_level` is `word`, `segment` or `mixed` and states what is really there; a segment without word times has `"words": null` and times are never spread over its words.
The timeline entry for a sentence (`timeline/timeline.json`) carries `start`, `end`, `timing_basis` (`word`/`segment`/`mixed`/`chunk_interval`), `alignment` (`exact`/`approximate`/`unaligned`), `segment_ids`, and the gloss.

## Tests

`tests/test_youtube.py` (URLs, error classification, downloader, audio incl. real FFmpeg, windows, transcription, merge, schema),
`tests/test_youtube_pipeline.py` (jobs, resume/cache, failures, existing cleaner + `translate` against a mock LLM, timestamps, alignment, CLI).
No test downloads a video or loads a speech model.

## Manual smoke test (needs network, yt-dlp, faster-whisper and about 0.5 GB for the model)

1. Install as above and run `youtube-check` until it reports everything present.
2. Choose a **short** (under 2 minutes) video you are permitted to download and transcribe: your own upload, or one marked Creative Commons / public domain. Downloading is subject to YouTube's terms and the uploader's rights.
3. `youtube URL --through transcript --model base.en` then `youtube-show URL --what transcript`: check the first and last lines against the video by ear (play it and compare times).
4. `youtube URL --gloss-limit 5` then `youtube-show URL --what gloss` (needs `GROQ_API_KEY` in `.env`; 1 request).
5. Re-run the same command: it must report "reusing" for every finished stage and spend no new LLM request.

## Known limitations

* **Not yet verified on a real YouTube download.** faster-whisper and FFmpeg were exercised for real (on synthetic speech); the yt-dlp path is exercised through a double, with its option names accepted by the installed yt-dlp 2026.8.19. Some error patterns (age check, bot check, live event) are written from yt-dlp/YouTube's usual messages and are unconfirmed until one occurs.
* Word timestamps: the first word of a recording tends to start at 0.00 s although sound begins a little later (about 0.12 s measured).
* Transcription accuracy is Whisper's: errors in names, accents, noise and music are expected. Nothing here corrects them (no LLM correction stage).
* Word timestamps are estimates (often within a fraction of a second, sometimes worse around pauses and overlapping speech); segment times are what the engine reports. VAD can drop very quiet speech.
* Sentence timing is recovered by a heuristic match of the cleaned text to transcript words; a sentence that cannot be matched keeps its cleaned chunk's interval and is labelled `unaligned`. Sentences are shorter than chunks, so sentence times are only as good as the matched words/segments.
* The audio stream's own start offset is recorded (and warned about) but not corrected; a video whose audio does not start at 0 may drift slightly against the picture.
* `small.en` is English only. Other languages need a multilingual model and `language: null`; the rest of the pipeline is English-only.
* CPU transcription speed is unmeasured here; a 3-hour limit (configurable) guards against accidents. The WAV is about 1.9 MB per minute.
* The gloss stage is limited to the first 10 sentences by default so a full video is not translated by accident; `--gloss-limit 0` lifts it (free-tier quota applies).
* yt-dlp can break when YouTube changes; update it first when downloads fail. Access restrictions are reported, never bypassed.
* Diarisation (who is speaking) is not done.
* The pipeline tester page takes a YouTube link too (see `pipeline_tester.md`); the web page runs a job only through the transcript and then feeds the saved transcript to the rest of the page's pipeline.
