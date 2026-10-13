# English transcript -> ISL gloss: Phase 1 (transcript cleaning)

Implements **Phase 1 only** of `English_to_ISL_Gloss_Project_Requirements_and_Phase1.pdf`: turn English transcripts
into clean, segmented, traceable text plus an audit trail. There is **no gloss translation and no sign generation** here.

Pure Python 3.9+, standard library only, no install required.

## Run it

```bash
python run_cleaner.py data                 # every .txt/.srt/.vtt directly inside data/
python run_cleaner.py data/some_file.txt   # one file
python run_cleaner.py data -o out_dir      # write outputs somewhere else
python run_cleaner.py data --keep-fillers      # fillers are removed by default
python -m unittest discover -s tests -t .  # run the tests (all)
```

Exit code: `0` all files processed, `1` at least one file failed, `2` bad arguments. A file that needs human review is
**not** a failure. One bad file never stops the batch; it gets a `failed` report and the next file is processed.

CLI options: `--keep-fillers`, `--collapse-repeats` and `--drop-sound-tags` (the last two **off** by default),
`--max-chunk-words N`, `--language`, `--source-type`, `--config file.json` (any `CleaningConfig` field, see
`src/transcript_cleaner/config.py`; unknown keys are rejected).

## Outputs (per source, named by `source_id`)

`source_id` = lower-cased file stem with non-alphanumerics turned into `-`
(`01_CSU-Bakersfield-English-Session.txt` -> `01-csu-bakersfield-english-session`). Two files that collapse to the same
id are rejected rather than overwritten.

| Folder | File | Content |
|---|---|---|
| `raw/` | `<id>.raw.<ext>` | Byte-for-byte copy of the input. **Never overwritten**: if different content already exists at that path the file fails with a clear message. |
| `cleaned/` | `<id>.cleaned.txt` | Spoken words only, one chunk per line. No title, source, speaker labels or timestamps (those live in the JSON). |
| `structured/` | `<id>.cleaned.json` | Chunks + metadata. Schema: `src/transcript_cleaner/schemas/cleaned_transcript.schema.json` (checked on every run). |
| `reports/` | `<id>.cleaning_report.json` | Stats, config used, full edit log, flag counts, warnings, validation result. Written last; its presence marks a finished run. |
| `reports/` | `batch_report.json` | Per-file status for the whole run. |
| `review_queue/` | `<id>.review.json` / `.csv` | Every item a human must look at, with a blank `decision` column (CSV opens in Excel). |

### `structured/*.cleaned.json`

Top level: `schema_version, source_id, source_filename, source_type, source_language, source_format, source_sha256,
raw_artifact, offset_basis, source_metadata (title, source_url, video_id...), timestamps_available, speakers_available,
cleaning_status (complete | needs_review | failed), review_required, warnings[], chunks[]`.

Each chunk: `chunk_id, text, start_time, end_time, speaker_id, turn_index, source_start_offset, source_end_offset,
cue_indices, cue_ids, uncertainty_flags[], edit_notes[]`.

* `null` means "not in the source". Timestamps, speakers and offsets are never invented.
* Speaker labels are kept verbatim (`"Speaker 1"`). They are diarisation labels, **not identities**; `Speaker 1` in one
  file is not the same person as `Speaker 1` in another. A label applies until the next label.
* **Offsets** (TXT only): Python string indices into the raw file decoded as UTF-8 (BOM removed, original line endings
  kept), i.e. `raw_text[start:end]` is the chunk's source span before whitespace normalisation. Verified on every run.
  SRT/VTT have no stable character offsets, so they are `null` and chunks carry `cue_indices` / `cue_ids` + times
  instead. A chunk that spans several cues gets the first cue's start and the last cue's end.
* `uncertainty_flags[]`: `{code, text, start, end, note, review_required}`; `start/end` index into the chunk text.
* `source_type` stays `unknown` unless you pass `--source-type`: a `watch?v=` URL cannot tell prerecorded from a
  livestream replay.

## Pipeline (one module per stage in `src/transcript_cleaner/`)

`ingest` (decode, TXT/SRT/VTT -> pieces; header, labels, cue numbers, timings become metadata) -> `segment` (sentence
spans on the *raw* text, so offsets stay exact) -> `normalize` (Unicode/whitespace per chunk) -> `clean` (opt-in rules)
-> `detect` (flags) -> `validate` -> `report` / `export`. `pipeline.py` wires them; `cli.py` is the batch front end.

## Cleaning policy as implemented

Always on (no word is changed):
* Header lines (`Source:`, `Title:`...) -> metadata; `Speaker N:` and `>>` -> speaker metadata; SRT numbers/timings,
  VTT headers/NOTE/settings, markup tags, HTML entities -> removed from text.
* Unicode NFC, exotic spaces -> space, zero-width/soft-hyphen/control characters removed (ZWJ/ZWNJ are kept for Indic
  scripts), line breaks and repeated whitespace -> single spaces, clean UTF-8-read-as-cp1252 mojibake repaired (logged).
* YouTube rolling-caption duplicate lines in SRT/VTT removed (logged); a lone repeated one-line cue is kept.
* A chunk that is *entirely* platform boilerplate (e.g. "Subtitles by ...") is dropped (logged).

Removed by default (decision: fillers are handled later through facial expression): `um, uh, uhm, er, erm`. Each
removal is in the report's edit log with its `chunk_id` and original text, so a later stage can place the expression.
"like", "you know", "hmm" and "ooh" are not touched. `--keep-fillers` turns removal off.

Off by default, logged when enabled: repeated-word collapse (every collapse goes to the review queue) and dropping
`[Music]`-style tags. `[inaudible]` and friends are never removed.

**Skip-translation markers** (decision: kept in the text, but the translator must skip them): each chunk has
`skip_translation_spans[]` = `{start, end, text, reason}` (indices into the chunk text).
* `expression_cue`: "hmm", "ooh", "whistle" and similar (`suspect_tokens` in config). Kept in the text on purpose for the
  later facial-expression stage, and skipped by translation.
* `skip_term`: words on the watchlist (currently "Jane", "Janindra", "Jhinendra", "Jin Indra", "Jai Jin"), edit in config.
* `skip_non_english`: runs of romanised Gujarati/Hindi marker words (2+ markers with at most 4 words between them).
  Coverage is approximate: neighbouring Gujarati words that are not in the marker list stay unmarked (see chunk 0260
  of file 02). Extend `non_english_markers`, or swap in a real language detector, if that matters.

Sentence breaks inserted by the ASR (a full stop followed by a lowercase word) are treated as real sentence ends.
Runs with no punctuation at all stay one chunk unless longer than 60 words.

Flag only, never rewritten: `unclear_audio`*, `replacement_character`*, `possible_mojibake`*, `repeated_word`, `repeated_phrase`, `non_speech_tag`, `short_fragment`,
`forced_split`. Flags marked * enter the review queue. `skip_term` and `skip_non_english` do not (already decided).

Validation (a violation marks the file `failed`): word accounting (source words - logged removals + logged additions
must equal output words; catches lost or duplicated words), offsets point at the right text, no speaker label/timestamp
left in text, no stray whitespace, unique ordered ids, valid timestamps, schema conformance. Warnings: consecutive
identical chunks, out-of-order times, very long tokens (possible concatenation), large text loss.

## What the three sample files taught us (see `data/reports/`)

* They are single-line bodies with inline `Speaker N:` labels, no timestamps, and punctuation inserted by ASR.
* Many sentences have no terminal punctuation or have capitals mid-sentence ("Hey Prabhu If you had not..."). The
  splitter cannot reliably recover those boundaries, so such run-ons stay one chunk (split only above 60 words).
* File 02 ends with romanised Gujarati (flagged, kept verbatim) and opens with likely ASR noise ("Om Ooh Hello Done,
  frio"). File 01 has "Jane"/"Janindra"/"Jhinendra" variants of Jain terms (flagged, not corrected).
* Consecutive identical sentences (e.g. "What qualities did he have?" twice) are real repetition; they are kept and
  only warned about.

## Limits and decisions still open

1. **Nothing is verified against audio.** Reports say so explicitly.
2. `source_type` is `unknown` because videos can be prerecorded or live. Pass `--source-type` per run if known.
3. The watchlist and the Gujarati/Hindi marker list are starter lists; extend them via `--config`.
4. LLM-assisted correction is not implemented (spec: optional, off by default).

---

# Part 2: translation module (cleaned chunks -> candidate ISL gloss)

Package `src/isl_gloss/`; brief: `Transcript_to_ISL_Gloss_Hybrid_Translation_Architecture.pdf`; details and decisions:
`docs/translation_implementation_notes.md`. Sequential pipeline, no fallback chain:

`cleaned chunk` -> cut skipped spans -> **LLM candidate** -> **mandatory rule pass** -> validation -> result (all of
source, candidate, final gloss, applied rules, warnings, metadata). Everything is labelled `unreviewed`.

```bash
python run_translator.py translate data/structured --dry-run     # preview requests, no key needed
python run_translator.py translate data/structured --limit 10    # first real test on the free tier
python run_translator.py translate data/structured               # everything; re-run the same command to resume
python run_translator.py evaluate                                # 40-sentence set -> reviewer CSV
python run_translator.py rules                                   # list the rule set
```

Setup: create a free API key in the Groq console (console.groq.com, API Keys), then `copy .env.example .env` and paste it
after `GROQ_API_KEY=` (or set the environment variable). The key is sent in a header only and is scrubbed from logs and errors.
**Default LLM: Groq `openai/gpt-oss-120b`** (strict JSON-schema output, reasoning effort `low`, reasoning hidden). Gemini is
still available: `--provider gemini` with `GEMINI_API_KEY`. Change the model with `--model`.
Groq limits tokens as well as requests (reported free tier for this model: about 30 requests/min, 1,000 requests/day, 8K
tokens/min, 200K tokens/day; check your own plan in the Groq console), so requests are paced by tokens per minute
(`--tpm`, default 7000) and requests per minute (`--rpm`, default 20). Results are cached in
`data/translations/*.llm_cache.jsonl`, so a daily-limit stop is resumable: re-run the same command.

## Sentence chunking and context grouping (new stages between cleaning and the LLM)

`cleaned JSON -> sentences (stable ids) -> context groups -> LLM (one request per group) -> rule pass -> validation`.
Details, config options, data structures and test results: [docs/sentence_chunking_and_grouping.md](docs/sentence_chunking_and_grouping.md).

```bash
python run_translator.py group data/structured                  # stages 2+3 only, no key; writes <id>.groups.json
python run_translator.py translate data/structured --dry-run     # preview the group requests
python run_translator.py translate data/structured --mode direct # the old fixed-window mode, for comparison
python run_translator.py compare                                 # old vs new on 5 passages (live; needs quota)
```

**What the model receives (prompts 6.0 and 7.0; 7.0 is the default):** the target sentences of one group, each on a line `[sentence_id] text`, taken from
the `text` of each chunk in `structured/*.cleaned.json` with skipped spans already cut out. The sentence id is the only
metadata: no speaker labels, timestamps, title or JSON wrapper. Read-only context blocks (no ids) appear only where a group's
first sentence depends on the previous one or it ends mid-thought. Preview it any time with `--dry-run`. The model returns, per
sentence: `sentence_id` (pinned to the ids in the request by the schema), `source_text` (a copy, checked against the input),
`candidate_gloss`, `uncertain_parts`, `uncertainty_notes`, `needs_human_review`.
Layer 1 details, the prompt, and a before/after comparison: [docs/layer1_llm_improvements.md](docs/layer1_llm_improvements.md).
Layer 2 (rule-based secondary pass) improvements are documented as future work only: [docs/layer2_future_requirements.md](docs/layer2_future_requirements.md).
Compare prompt versions on the same inputs: `python run_translator.py compare-prompts` (observation only; see the doc).

Outputs in `data/translations/`: `<id>.gloss.json` (full audit trail), `<id>.review_sheet.csv` (for the ISL reviewer),
`<id>.gloss.txt` (quick look). Run the tests with `python -m unittest discover -s tests -t .`.
