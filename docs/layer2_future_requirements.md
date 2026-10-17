# Layer 2 (rule-based secondary pass): requirements for a FUTURE phase

> **Update:** a different Layer 2 (an independent spaCy candidate, compared with the LLM candidate, never rewriting either) has since been implemented and evaluated: see `layer2_dual_candidate.md`. The rule-based *correction* pass described below is still not implemented.

**Status: documented only. Nothing in this file is implemented.** Recorded 13 Oct 2026 as Part B of the "Improve LLM
Translation Layer" request. The Layer 1 task deliberately adds no Layer 2 logic, no rule-based corrections, no
post-rule validation and no transformation logging.

## What already exists in the repository (and was not touched by the Layer 1 task)

Before this request the repository already contained a small rule pass and per-sentence validation, built under the earlier
architecture brief: rules `R-FMT-01` (token format), `R-ART-01` (drop THE/AN), `R-SKIP-01` (remove words the cleaner marked
skip), three rules that are off by default (`R-COP-01`, `R-ORD-WH-01`, `R-ORD-TIME-01`), rule-set version `0.1.0`, and the
validation checks in `validate.py` that predate this task (empty gloss, token format, possible omission/addition, number,
negation, possible missing name, unknown vocabulary, assembly integrity). The Layer 1 task did not modify them. The
requirements below describe what the real Layer 2 should become; the existing rules are a starting point, not a
substitute, and the three off-by-default ordering/copula rules should be treated as unreviewed.

## Pieces that were briefly added and then removed (so they can be rebuilt properly)

Last turn I added, beyond what was asked, one rule and four validator checks. They fall under Layer 2 as defined here, so
they were **removed** when this boundary was set:

| Removed item | What it did | Why it must not come back as-is |
|---|---|---|
| Rule `R-FS-01` | Rewrote `FS:JONATHAN_YOUNG` into `FS:JONATHAN FS:YOUNG` | A blind rewrite; needs a reviewed convention for multi-word names and for institutions first |
| Check `name_not_fingerspelled` | Flagged a capitalised mid-sentence word written as an ordinary label | "Do not blindly transform every capitalised word into a name": it also hit "Jainism", "Though", titles |
| Check `condition_may_be_lost` | Flagged `if/unless/whenever` with no IF/UNLESS/WHENEVER token | Token lists are an English-centric approximation of ISL conditional marking |
| Checks `cause_may_be_lost`, `time_reference_may_be_lost` | Same idea for because/therefore and yesterday/today/tomorrow/tonight | Same limitation |

The detectors survive only inside the evaluation tool (`prompt_eval.py`), where they observe LLM output for the prompt
comparison and never change anything.

## Intended responsibilities

Layer 2 must run **after the LLM on every valid candidate gloss**. It is not a fallback for LLM failure.

### 1. Conservative rule application
* Apply a rule only when its documented preconditions are satisfied.
* Never rewrite a gloss blindly. Do not apply generic word-order, negation or word-removal rules without linguistic justification.
* If a transformation is uncertain, keep the LLM candidate.
* Prefer flagging a possible issue for review over making an unsupported correction.

### 2. Proper-name consistency checks
* Check whether names follow the documented project-level `FS:NAME` convention; flag violations of the format where applicable.
* Do not turn every capitalised word into a name (sentence starts, conjunctions, titles, domain terms such as "Jainism" need a policy).
* Do not invent signs or name signs. Where a verified lexical sign or approved name-sign convention exists, follow the documented project policy.
* `FS:NAME` is a project notation, not a universally accepted ISL gloss convention.

### 3. Meaning-preservation checks
* Flag potential omissions of conditions, negation, questions, time, quantities and other meaning-bearing elements, using the original English sentence and the group context where needed.
* Do not blindly insert English words such as `IF` into a gloss. Any correction must be justified by an applicable linguistic rule and the project convention; flagging for review is the default.

### 4. Output traceability
Keep the LLM candidate and produce a **separate** post-rule output. Record: the original candidate gloss; the final gloss after
rules; applied rule ids; a concise description of each transformation; warnings and review status.

### 5. Validation after the rule pass
Detect: empty or malformed outputs; missing or duplicate sentence ids; missing sentence results; unexpected changes of
sentence order; potential meaning omissions; inconsistent project-level gloss notation (for example the same concept written
`SAME_PLACE` in one sentence and `SAME PLACE` in another). Technical validation is never proof that a translation is
linguistically correct.

## Before implementation

* Obtain reviewer-approved conventions (multi-word names, institutions, religions and domain terms, conditional and negation marking).
* Agree which checks flag and which, if any, may correct.
* Keep the evaluation set (`evaluation/layer1_inputs.json`, `eval_sentences.json`) and add reviewer corrections as reference glosses.
