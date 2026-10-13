# Sentence-level chunking and context-based grouping

Implements the change request "Sentence-Level Chunking and Contextual Grouping for ISL Gloss Translation".
Status (13 Oct 2026): implemented and tested (180 tests pass). The live old-vs-new comparison has NOT been run yet: the
free Gemini daily quota was exhausted (HTTP 429) when it was attempted (the project has since moved to Groq, see translation_implementation_notes.md). See "Comparison with the direct pipeline".

## 1. What was inspected (before any change)

| Piece | Where | Finding |
|---|---|---|
| Transcript cleaning | `src/transcript_cleaner/` | Works and is **unchanged**. It already splits at sentence punctuation (abbreviation, initial, decimal, ellipsis aware) and keeps `turn_index`, `speaker_id`, timestamps, skip spans, flags. Its only non-linguistic split is a 60-word cap on unpunctuated run-ons. |
| Point where text reached the LLM | `isl_gloss/cli.py` -> `Translator.run(chunks)` | Cleaned chunks were packed into **fixed windows of 10** (`_make_batches`), regardless of meaning, with no sentence ids and no group concept. |
| LLM integration | `isl_gloss/llm.py`, `prompt.py` | Gemini client, retries, error taxonomy. Kept; only the prompt and the request unit changed. |
| Rule pass, validation | `rules.py`, `validate.py` | Per-sentence, already id-agnostic. Kept; validation gained run-level checks. |

## 2. Smallest change that fits

Two new modules inserted between "load cleaned JSON" and `Translator.run`; the translator now takes a list of groups.

```
Raw transcript -> [existing] transcript_cleaner -> cleaned JSON (structured/*.cleaned.json)
   -> Stage 2  isl_gloss/sentences.py   build_sentences()   Sentence objects, stable sentence_id
   -> Stage 3  isl_gloss/grouping.py    group_sentences()   Group objects, group_id, read-only context
   -> Stage 4  isl_gloss/translator.py  one LLM request per group, numbered sentences in, numbered results out
   -> Stage 5  isl_gloss/rules.py       mandatory rule pass on every valid candidate (unchanged logic)
   -> Stage 6  isl_gloss/validate.py    per-sentence checks + run-level assembly checks -> results in original order
```

The old behaviour is still available as `--mode direct` (cleaned chunks in fixed windows) and is the baseline of the comparison.

## 3. Files

New: `src/isl_gloss/sentences.py`, `grouping.py`, `compare.py`, `timeutil.py`; `evaluation/context_passages.json`;
tests `tests/test_gloss_sentences_grouping.py`, `tests/test_gloss_grouped_pipeline.py`.

Changed: `translator.py` (rewritten around "units"), `prompt.py` (v4.0), `models.py`, `prepare.py`, `store.py` (cache key),
`validate.py` (+ `validate_assembly`, `status_from`), `export.py`, `cli.py`, `config.py`, and the tests that encoded old ids/formats.

Unchanged: everything in `src/transcript_cleaner/`, `llm.py`, `rules.py`.

## 4. Data structures

`Sentence` (Stage 2): `sentence_id` (`<source_id>-s0001`, transcript order), `source_text`, `start_timestamp`, `end_timestamp`,
`speaker_id`, `source_metadata` (source_id, `parent_chunk_id`, `split_method`, `piece`, offsets or cue ids), plus
`turn_index`, `incomplete`, `depends_on_previous`, `dependency_reasons`, `skip_spans`, `excluded_spans`.
Timestamps and speaker are `None` when the source has none.

`Group` (Stage 3): `group_id` (`<source_id>-g0001`), `sentence_ids`, `sentences`, `context_text`, `start_timestamp`,
`end_timestamp`, plus `context_before_ids`, `context_after_ids`, `boundary_reason`, `forced_boundary`, `oversize`.
`sentences` are the ones that get separate gloss outputs. `context_text` is the readable passage including any read-only
context. Read-only context sentences are never translated and never counted as the group's sentences.

Per-sentence result (`<id>.gloss.json` -> `results[]`): `sentence_id`, `group_id`, `source_text`, `llm_input_text`,
`skipped_spans`, `candidate_gloss`, `candidate_tokens`, `final_gloss`, `final_tokens`, `applied_rules`, `validation`,
`uncertainty_notes`, `speaker_id`, `start_timestamp`, `end_timestamp`, `turn_index`, `source_metadata`, `metadata`,
`review_status: "unreviewed"`. **Breaking change:** `chunk_id` is now `sentence_id`; the nested `context` object is gone
(its fields are top level). Top level of the file adds `mode`, `grouping`, `groups`, `assembly_validation`.

## 5. Stage 2: sentence chunking

* Punctuation-based sentences come from the cleaner and are carried over unchanged (text is only ever sliced, never rewritten).
* **Missing punctuation:** a chunk of at least `runon_min_words` (14) words with no internal punctuation is split before a
  capitalised sentence opener (If, But, When, They, It, This, ...), never at word-count intervals. Needs 3+ words on each side,
  never cuts inside a skip span, and is marked `split_method: runon_capital_opener`. "I", "The", "A" are not openers.
  Risk: a genuinely capitalised word in the middle of a sentence can cause a wrong split; `--runon-min-words 0` switches it off.
* **Incomplete sentences / ASR false stops:** kept as they are (earlier decision: ASR full stops are real sentence ends) but
  flagged. `dependency_reasons` records why a sentence probably needs its predecessor: `lowercase_start`,
  `continuation_word` (and, but, so, because...), `pronoun_start` (he, she, it, they, this, that...), `previous_incomplete`
  (no terminal punctuation or ends with , ; :), `very_short` (two words or fewer). These are heuristics, not a parse.
* Real-data effect: 564/444/253 cleaned chunks became 578/463/265 sentences (27/36/23 sentences from run-on splits).

## 6. Stage 3: grouping (baseline: adjacency + limits)

* A group is a run of adjacent sentences; order is never changed and no sentence is dropped or duplicated.
* **Hard breaks** (unrelated passages are not combined to fill a group): new speaker turn/paragraph (`turn_index`), a different
  known speaker, or a pause over `group_max_gap_seconds` when timestamps exist.
* **Limits:** sentences, characters, estimated tokens (characters / 4, no tokenizer dependency). When a limit is hit the cut goes
  at the latest boundary where the next sentence does **not** depend on the previous one. If every boundary is dependent the cut
  is forced and recorded (`forced_boundary`); the next group then gets read-only context from before the cut.
  A sentence is never cut; one that alone exceeds a limit is its own `oversize` group.
* **Read-only context:** a group whose first sentence depends on its predecessor gets up to `context_sentences_before`
  earlier sentences; a group that ends mid-thought, or whose successor depends on it, gets up to `context_sentences_after`
  following ones. Skipped words are cut from context too.
* Not done on purpose: no embeddings or topic-similarity grouping. The request says to add them only if an evaluation justifies it.
  Topic similarity alone does not prove dependence, and nothing so far shows the baseline is insufficient.

## 7. Stage 4: LLM request (prompt v4.0)

One request per group. The model sees only bare numbered sentences, plus optional "Earlier speech" / "Following speech" blocks.
No ids, speakers, timestamps or JSON wrapper. The numbers are the minimum needed to map each reply to a sentence id.
The prompt tells the model that the numbered sentences are one connected passage to be used for resolving pronouns and
references, but that every sentence gets its own gloss containing only its own content, that pronouns stay pronoun tokens,
that unresolvable references or fragments are marked `uncertain` with a note, and that context blocks are never output.
Reply fields: `n`, `gloss`, `uncertain`, `notes`; mapped to `sentence_id`, `candidate_gloss`, `uncertainty_notes` locally.
Same provider, client and error handling as before. The result cache is keyed on model + full system prompt + full request
text + sentence number, so a different context can never reuse an old answer.

## 8. Stage 6: validation

Per sentence (as before) plus reply checks: duplicate result for a sentence, results out of order (matched by number, flagged),
stray results with unusable numbers (ignored, flagged), missing results (one repair request with the rest as context, then
failure recorded on that sentence). Run level (`validate_assembly`, written to `assembly_validation`): every sentence has
exactly one result, in the original order, no unexpected ids, groups cover every sentence exactly once in transcript order,
group ids unique, no empty groups, start times not going backwards. A group that fails never removes its sentences: they stay
in the output as `failed` or `not_attempted`. These checks cannot show linguistic correctness.

## 9. Configuration (all in `TranslatorConfig`, overridable by `--config` JSON or CLI flags)

| Option | Default | CLI flag |
|---|---|---|
| `runon_min_words` | 14 (0 = never split) | `--runon-min-words` |
| `group_max_sentences` | 8 | `--max-group-sentences` |
| `group_max_chars` | 1200 | `--max-group-chars` |
| `group_max_tokens` (chars/4) | 300 | `--max-group-tokens` |
| `group_max_gap_seconds` (null = ignore timing) | 8.0 | `--max-gap-seconds`, `--no-time-gap` |
| `group_break_on_turn_change` | true | config only |
| `context_sentences_before` / `_after` | 1 / 1 | `--context-before`, `--context-after` |
| `context_chunks`, `batch_size` | 0 / 10 | direct mode only |

## 10. Commands

```bash
python run_translator.py group data/structured             # stages 2+3 only: no key, writes <id>.groups.json
python run_translator.py translate data/structured --dry-run   # shows the group requests; nothing is sent
python run_translator.py translate data/structured --limit 20   # new default mode: sentences -> groups -> LLM
python run_translator.py translate data/structured --mode direct # the old fixed-window mode
python run_translator.py compare --dry-run                 # request counts for the 5 comparison passages
python run_translator.py compare                           # live old-vs-new comparison (needs quota)
```

## 11. Real-data numbers (your three transcripts, `group` command)

| | chunks | sentences | groups (max 8) | requests if max 12 |
|---|---|---|---|---|
| 01 | 564 | 578 | 105 | 81 |
| 02 | 444 | 463 | 66 | 44 |
| 03 | 253 | 265 | 39 | 28 |

No forced cuts and no oversize groups. A full run needs 210 requests at the default size (the old windows needed about 128),
so grouping costs more free-tier quota. `--max-group-sentences 12` brings it to 153.

## 12. Tests (180 pass: `python -m unittest discover -s tests -t .`)

New coverage for the ten requested cases: (1) normally punctuated text, (2) missing punctuation, (3) related sentences needing
context sent together, (4) unrelated topics kept in separate groups, (5) long sentences and groups over the character/token
budget (including a single oversize sentence), (6) incomplete sentences and ambiguous pronouns (dependency signals, forced cuts,
read-only context), (7) timestamps and speakers (SRT with cue ids, speaker/time-gap breaks), (8) sentence id <-> LLM output
mapping (including shuffled, duplicate, stray and missing results), (9) chronological order across many groups and a
3000-sentence transcript, (10) ids, speakers and timestamps flowing through LLM, rule pass and validation; plus assembly
validation failures, failed/stopped groups never dropping sentences, cache correctness, the comparison tool and the CLI.

## 13. Comparison with the direct pipeline

`compare` runs five short passages (`evaluation/context_passages.json`: pronouns, speaker turns, unrelated topics, unpunctuated ASR
text, ambiguous references) through the real cleaner, then through the old direct mode and the new grouped mode with separate
caches. It reports per mode: translated/failed counts, uncertain flags, meaning-related warnings (omissions, additions, lost
numbers, negation, names), assembly checks and request counts, and writes a side-by-side sheet for the ISL reviewer.
**Automatic checks can reveal regressions; they cannot show better ISL.** If either mode stops early the verdict says
INCOMPLETE and concludes nothing. Offline, the tool and its verdict logic are tested with fake models.
**Not yet run live** (daily quota). Rerun `python run_translator.py compare` once the quota resets (13 requests).

## 14. Known limits

* Dependency and run-on detection are heuristics; wrong splits or missed dependencies are possible and are visible in `<id>.groups.json`.
* Grouping raises request count; a group that fails fails together (then it is split and retried, bounded).
* Old output files in `data/translations/` (small samples from earlier runs) use the previous schema (`chunk_id`) and prompt versions; regenerate them.
