# Interactive pipeline tester

A small local web page that runs your **real** pipeline on any English text you type and shows every stage.
It is a testing aid. Everything it shows is unreviewed machine output, not verified ISL.

## Start it

```bash
.venv/Scripts/python.exe run_webui.py
```

Then open <http://127.0.0.1:8765/> (use `--port 9000` to change the port). Use the project's `.venv` so spaCy is available.
The Groq key is read on the server from `.env` (`GROQ_API_KEY`); it is never sent to the page.

## Two ways in: type text, or paste a YouTube link

The page has two tabs. **Type text** works as described below. **YouTube link** takes a link and first shows three steps, ticking each off as it really finishes:

1. **Video detected successfully**: the link is valid and the video is publicly available (title, length shown).
2. **Audio extracted successfully**: the audio-only download is converted to 16 kHz mono WAV with the timeline untouched.
3. **Audio transcribed successfully**: Whisper's timestamped transcript (segments, timestamp level, model); a collapsible list shows every line with its time.

Only after all three succeed does the rest run exactly as for typed text (stages 1 to 6 below), fed from the saved timestamped transcript so **every sentence shows its time in the video**. If a step fails, the page stops there with the reason and a hint, the later steps stay pending, and **nothing is sent to the LLM**. Running the same link again reuses the saved results ("reused saved result"). The only option is **Gloss the first N sentences** (default 10, up to 40, so a whole video is not translated by accident). Everything else is fixed on the page: **prompt 8.0**, **medium** reasoning (Groq `openai/gpt-oss-120b`, temperature 0), the **small.en** speech model, and saved LLM answers are reused for identical requests. (The command line is unchanged and still defaults to prompt 7.0.) To redo a video from scratch, delete its folder in `data/jobs/`.

Limits: public videos only (nothing is signed in or bypassed), up to 30 minutes by default (`--max-video-minutes`), one video at a time. Jobs are kept in `data/jobs/` (`--jobs-dir`); see `youtube_ingestion.md`.

## What you see

1. **Cleaned text**: the cleaner's output. Highlighted words (fillers, non-English, unclear audio) are kept but skipped in translation.
2. **Sentences**: numbered `#1, #2, …` with their ids; run-on splits and dependencies are tagged.
3. **Context groups**: which sentences travel together (one LLM request per group) and why each group ended. The first three stages appear instantly and make no LLM call.
4. **Layer 1 · LLM**: candidate gloss per sentence, the model's doubts, and (collapsed) the exact message and system prompt that were sent.
5. **Layer 2 · spaCy**: the rule-based translator's output on the same cleaned sentence.
6. **Comparison and output**: both candidates side by side, the candidate the policy carried forward, and why. The LLM is the primary candidate by policy default; spaCy is an independent alternative.


## Limits and safety

* Typed text: up to 6,000 characters and 40 sentences per run (about 5 LLM requests at most) so the free tier is not burned by accident.
* One LLM run at a time. A failing LLM (quota, bad key) is shown on the page; spaCy still runs.
* Binds to `127.0.0.1` only; requests with a foreign `Host` or `Origin` are refused, so another website cannot use your key through your browser. Any key value is scrubbed from responses.
* The standard library only: no new dependencies.

Tests: `tests/test_webui.py`.
