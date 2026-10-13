# Reviewer guide: LLM candidate vs spaCy candidate

**Nothing in these sheets has been reviewed.** Both glosses are machine-generated. A "selected" candidate has only passed
technical checks; it is not linguistically verified. Your judgements are what will show whether either candidate, or the
combination, is any good.

## Which sheet

| Folder | LLM candidate comes from |
|---|---|
| `prompt_7.0/` | Layer 1 prompt 7.0 (the current default) |
| `prompt_8.0/` | Layer 1 prompt 8.0 (an experimental revision; not the default) |

The spaCy candidate is identical in both (same sentences, same rule-based translator). Review whichever LLM version you want to
assess; the file to open is **`candidates_side_by_side.csv`** (UTF-8; opens in Excel). The same data with full detail is in
`candidates_side_by_side.json`.

## How to fill it in without touching the candidates

1. **Save a copy first** (for example `candidates_side_by_side_reviewed_<yourname>.csv`) and fill in the copy.
2. Only edit the last seven columns. Do not change any other column; they are the record of what each system produced.

| Reviewer column | Allowed entries |
|---|---|
| `preferred_candidate (...)` | `LLM`, `SPACY`, `BOTH_ACCEPTABLE`, `NEITHER_ACCEPTABLE`, `UNCERTAIN` |
| `meaning_preserved_llm` / `meaning_preserved_spacy` | for example `yes` / `partly` / `no` |
| `linguistically_acceptable_llm` / `_spacy` | for example `yes` / `partly` / `no` |
| `missing_or_introduced_meaning` | free text: what is lost or added, and in which candidate |
| `reviewer_comments` | free text; a corrected gloss may go here |

## What the other columns mean

* `cleaned_source_text` is the exact English both systems received. `source_text` is the sentence before skip-markers were applied.
* `llm_gloss` is the model's own gloss (empty means it declined; see `llm_uncertainty_notes` and `llm_stated_review_flag`).
* `spacy_gloss` is the reference translator's output: lower-case lemmas in a dependency-based order. A token can be a merged phrase containing spaces. It has no `FS:` convention.
* `llm_warnings` / `spacy_warnings` list **heuristic flags** (for example a missing negation word). They are prompts to look, not errors; ISL may express these non-manually.
* `candidate_difference` summarises word-level differences. It is not a similarity or quality score.
* `primary_candidate` is always `LLM`: the LLM gloss is carried forward by policy whenever it is usable. **That is a default, not a judgement that it is better than spaCy.** spaCy is an independent alternative.
* `selection_status`: `llm_primary` = LLM usable and carried forward by default; `candidates_agree` = same words after normalising case and separators; `llm_abstained_alternative_available` = the LLM declined the sentence and spaCy is offered but not substituted; `llm_failed_alternative_rejected` = the LLM is unusable and spaCy did not pass the stricter fallback check; `spacy_fallback_selected` = the LLM technically failed and spaCy was used as a stand-in (none so far).
* `alternative_usable_as_fallback` says whether spaCy passed that fallback check (blank if it was not evaluated).
* `needs_human_review` is `True` only where the LLM itself asked for review, a heuristic warning fired, or the LLM failed. It does **not** mean other rows are fine: every row is unreviewed, which is why all rows are in this sheet.
* `FS:NAME` is this project's notation for a fingerspelled name, not a claim about universal ISL practice.
