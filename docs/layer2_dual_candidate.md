# Layer 2: an independent spaCy candidate, compared with the Layer 1 LLM candidate

**Status: implemented and run on the 41-sentence set. Nothing here is ISL-reviewed.** The counts below describe how the two
translators behave and agree. They do not show that either gloss is correct, nor that combining them improves quality.

## What it does

```
cleaned English (the exact text Layer 1 sent to the LLM)
  ├─ Layer 1: LLM gloss                (already saved; reused, never re-run)
  └─ Layer 2: spaCy reference gloss    (independent; never sees the LLM gloss)
        → structural checks, conservative warnings, descriptive difference
        → selection policy (LLM primary; spaCy kept as a checked alternative)
        → side-by-side files for an ISL reviewer
```

Layer 1 does not import Layer 2 (tested). Layer 2 only reads a saved Layer 1 results file, records its SHA-256 before and
after, and makes **no LLM request**.

## Code (`src/isl_gloss/layer2/`)

| File | Role |
|---|---|
| `spacy_reference.py` | Port of `spacy_rules.py` from krishnshyam/VirtualISLInterpreter, commit `f5d9484`. Logic unchanged. |
| `spacy_translator.py` | Adapter: never raises; keeps sentence id, group id, timestamps, raw output, status and error. Records dependency versions. |
| `candidate_comparator.py` | Structural validity, conservative semantic warnings, notation warnings, descriptive difference. Never edits a candidate. |
| `selection.py` | Configurable policy with rule ids `R-SEL-00 … 08`. |
| `pipeline.py` | `run_layer2`, summary, artefacts, report. |

### Compatibility changes to the reference code (only these two)

* **C1** the model was loaded at import (`nlp = load("en_core_web_sm")`), so importing without the model crashed the app. It now loads on first use.
* **C2** `translate_to_tokens` calls the lazy loader.

Parity was checked against the **unmodified** upstream file on 56 inputs (the 41 evaluation sentences and 15 extra cases):
identical output on every input except whitespace-only text, where upstream raises `ValueError` ("ROOT" missing) and the
adapter reports status `empty`. `tests/fixtures/spacy_reference_outputs.json` holds the observed upstream outputs (parity
data, **not** ground truth).

### Reference behaviour preserved on purpose (observed, not fixed)

* `find_syn()` uses an undefined `wordlist` and is never called; the `nltk.corpus.wordnet` import is only needed to import the module (no WordNet data is read).
* Output is lower-case lemmas; only the first DATE is moved to the front (a second date stays in place, and a stray `.` token can appear: `monday on we meet and tuesday 5 june on . on`).
* Merged noun phrases keep their articles (`the cat on the mat sit`): the drop-list only removes whole tokens.
* `whenever` appeared twice in one output; conjunction splitting emits `and` into the gloss.
* A question word is detected only if the first token is a W-word, so `Did you finish the report?` gives `you the report finish` (no marker).
* Sentences are translated independently; no context is used.
* The upstream repository has **no licence file**. Confirm the terms of reuse before redistributing the port.

The root-children indexing was inspected: the `.index(...)` calls and `root_children[...]` use the same child order, so it is consistent.

## Selection policy (`selection.py`, policy version 2.0)

**The LLM is the primary candidate. spaCy is an independent alternative.** Both are always generated and kept; neither is edited.

| Rule | Condition | Result |
|---|---|---|
| R-SEL-00 | nothing translatable in the source | `no_source_text` |
| R-SEL-03 | LLM and spaCy both usable and identical after normalising case / `FS:` / separators | `candidates_agree` (LLM; agreement is not proof) |
| R-SEL-05 | **off by default**: LLM raises a listed concrete warning and spaCy raises none | `selected_by_check` (spaCy, always flagged) |
| R-SEL-06 | LLM technically usable | `llm_primary`: LLM carried forward by **policy default**; spaCy kept as the alternative |
| R-SEL-01 | neither candidate technically usable | `both_invalid` |
| R-SEL-08 | LLM unusable and spaCy fails the stricter fallback check (or fallback is switched off) | `llm_failed_alternative_rejected`, nothing carried forward |
| R-SEL-07 | LLM **declined on purpose** (empty gloss plus its own review flag) and spaCy passes the fallback check | `llm_abstained_alternative_available`: spaCy is offered, **not substituted** |
| R-SEL-02 | LLM technically failed and spaCy passes the fallback check | `spacy_fallback_selected` (spaCy, provisional, flagged) |

* `llm_primary` is a default, not a finding. The record says so (`selection_basis = policy_default_primary`) and no selection currently rests on human-reviewed evidence.
* An empty LLM gloss is **not** enough to select spaCy. spaCy must first pass the **fallback check**: technically usable, no possible-omission warning, no punctuation-only token. A model that deliberately declined a sentence is respected; `fallback_on_llm_abstention` (off) would change that.
* Length, token count, similarity and English-word counts never decide anything.
* The LLM row is flagged for review only for its own reasons (its review flag, a warning, a failure); spaCy merely differing is not a reason, though every row is still in the reviewer sheet.
* Every decision carries `verification = technical_checks_only_not_linguistically_verified`.

## Checks (warnings only)

Structural: missing, empty, malformed output; sentence-id or source-text mismatch; translator failure or unavailability;
Layer 1 validation failure. Semantic (heuristic, can miss and can false-alarm; ISL may mark these non-manually):
negation, condition, wh-question word, cause, time expression, digits, possible names. Missing English function words are
**not** warnings. Notation: tokens outside the project format, `FS:` joining several words. `FS:NAME` is a project
convention, not a universal ISL one; no word is converted into one.

## Setup

Layer 2 needs spaCy, its small English model and NLTK. They are never downloaded automatically.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install spacy nltk
.venv/Scripts/python.exe -m spacy download en_core_web_sm
```

(or `pip install -e .[layer2]` plus the model download). NLTK's WordNet data is **not** needed. The versions used here:
Python 3.11.5, spaCy 3.8.16, en_core_web_sm 3.8.0, NLTK 3.10.3. Check an environment with:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m isl_gloss.cli layer2-check
```

## Running

Independently, on saved Layer 1 output (no API key or request needed):

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m isl_gloss.cli layer2 data/translations/prompt_comparison/results_prompt_7.0.json -o data/translations/layer2_evaluation/prompt_7.0
```

Through the full pipeline (Layer 1 first, then Layer 2 on its saved result, into `<out>/<name>.layer2/`):

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m isl_gloss.cli translate data/structured --limit 10 --layer2
```

Options: `--llm-field final_tokens` (use Layer 1's post-rule gloss as Candidate A), `--layer2-config file.json`
(`{"enabled": false}` disables Layer 2; `policy` sets the selection options). `translate` accepts `--layer2-config` too, so
`--layer2 --layer2-config off.json` runs Layer 1 and skips Layer 2. If Layer 1 stopped early (for example on quota), Layer 2
still runs on what exists, and its manifest and report say the Layer 1 run was incomplete. Re-running rewrites the same files
atomically; the output folder may not be the Layer 1 file's folder.

## Output (`data/translations/layer2_evaluation/prompt_<version>/`)

`spacy_raw_outputs.json`, `candidates_side_by_side.json` and `.csv` (with blank reviewer columns: preferred candidate
`LLM/SPACY/BOTH_ACCEPTABLE/NEITHER_ACCEPTABLE/UNCERTAIN`, meaning preserved, linguistic acceptability, missing or introduced
meaning, comments), `selection_decisions.csv`, `warnings.csv`, `summary.json`, `summary_report.md`, `run_manifest.json`.

## Results on the 41-sentence set (no reviewer labels exist)

| | Layer 1 prompt 7.0 | Layer 1 prompt 8.0 |
|---|---|---|
| spaCy: gloss / empty / failed | 41 / 0 / 0 | 41 / 0 / 0 |
| LLM: gloss / empty / failed | 37 / 4 / 0 | 40 / 1 / 0 |
| LLM kept as primary by policy default | 37 | 40 |
| LLM declined, spaCy offered but not substituted | 3 | 0 |
| LLM unusable and spaCy rejected by the fallback check | 1 | 1 |
| spaCy selected | 0 | 0 |
| Selections resting on reviewed evidence | 0 | 0 |
| Flagged for review (LLM's own flag, warning or failure) | 6 | 2 |

The single rejected case is the garbled ASR line `Om Ooh Hello Done frio`: spaCy's output (`hello frio om`) raises a possible-name warning, so it is not accepted as a stand-in. spaCy never replaced the LLM here; its glosses are a retained alternative for the reviewer. Whether they are any good is unanswered until an ISL reviewer fills in the sheet.
(The 8.0 column was computed with the current code; the on-disk `prompt_8.0/` folder is refreshed by re-running its command once its CSV is closed in any viewer.)

## Verification performed

* `translate --layer2` was run as a real subprocess against a local mock LLM server: one LLM request in total, both candidate paths ran, results saved, Layer 1 output identical to a run without `--layer2`.
* Layer 2 disabled through the CLI (`--layer2-config` with `enabled: false`, and the standalone command) writes nothing.
* A genuinely missing spaCy package and a genuinely missing model were exercised in subprocesses (the model hidden from import and package metadata, so spaCy's own `spacy.load` raised `[E050]`); each time the LLM result was kept and the error named `python -m spacy download en_core_web_sm`. Tests: `tests/test_layer2_cli.py`.
* Reviewer instructions: `data/translations/layer2_evaluation/README_for_reviewers.md`.

## Limitations

* No linguistic evaluation. The reviewer columns are empty on purpose.
* The reference translator is a 2021 dependency-order heuristic. Its output is English-lemma order with leaked function words and punctuation, so it differs from the LLM on every usable pair here; that difference says little about which is closer to ISL.
* The comparator's semantic checks are word-list heuristics (English-centric) and unvalidated; R-SEL-05 stays off for that reason.
* Results depend on the spaCy and model versions (recorded in `run_manifest.json`).
* No context: spaCy translates each sentence alone.
* Not done: reviewing the 8.0 prompt (separate task), a learned selector, any correction of either candidate.
