# Gloss signability: synonyms and name spelling (Phase 1 follow-up)

Written 2026-10-20. Scope: making Phase 1 gloss easier to *show* with the signs we actually have (Yap avatar words and the ISLRTC dictionary).
It does **not** make the gloss linguistically more correct, and nothing here is reviewed by an ISL signer.

## Real Phase 1 run (2026-10-20): Groq, prompt 8.0

After you set up the key I re-ran Phase 1 on all three transcripts: `python run_translator.py translate data/structured --prompt-version 8.0 --limit 40 -o data/translations_v8` (Groq `openai/gpt-oss-120b`, 24 requests, the first 40 sentences of each transcript = 120 sentences; 120 of 120 translated, 11 flagged `review_required` by the validator).
This replaces my earlier measurements, which used the old stored files (Gemini, prompt 1.0/2.0; those only held the first 5 sentences per transcript and transcript 02 had failed entirely with HTTP 503). The full transcripts are 581 / 463 / 267 sentences (210 requests, roughly 2 hours on the free tier); that run has **not** been done.

What the fresh output looks like (1,174 gloss words):
* Names are handled well by the prompt itself: `PROFESSOR FS:JONATHAN FS:YOUNG`, `FS:JAINISM`, `FS:PRABHU`. A few are still plain tokens (`CALIFORNIA STATE UNIVERSITY`), the known `name_not_fingerspelled` failure.
* **The gloss is close to word-for-word English.** Examples: `BUT IT IS PHILOSOPHY AND WAY_OF_LIVING`, `LET US CLOSE EYES FOR TWO MINUTES`, `I WOULD LIKE APPRECIATE YOUR CURIOSITY TO LEARN SOMETHING ABOUT ...`. The prompt asks to drop empty function words, but the model keeps many.
* **41% of gloss words have no sign** (484 of 1,174: 193 avatar, 423 dictionary video, 74 spelled). Of the 439 unsigned words that are not names, **33% (146) are English function/auxiliary words**: `OF` 16, `AND` 15, `BUT` 14, `TO` 13, `FOR` 12, `NOT` 12, `WITH` 11, `WOULD` 8, `SO` 7, `IF` 6, `THEN` 6, `OR` 5, `IS` 4... Neither Yap nor the dictionary (a glossary of terms) has signs for them, so synonyms cannot help.
  The rest are content words missing from both vocabularies (`IMPRESSION`, `NORMAL`, `FORTUNATE`, `REMAIN`, `GRATITUDE`, ...).

## Limits of what I could check

* **There is no reviewed ground truth.** `references/reference_set.json` is an empty template and the only quality measures are crude detectors, so "better gloss" cannot be proven by a score; the objective, measurable thing is **how many gloss words have a sign**.
* I changed no prompt. Whether the function words should be dropped is an ISL-grammar decision (see "Function words" below), not something I can settle without a reviewer.
* The earlier design decision "synonym mapping: forbidden, silent substitution changes meaning" (`docs/translation_implementation_notes.md`) is **reversed on your instruction**. The safeguards below are what keeps that risk visible.

## The design

```
Phase 1 gloss (unchanged)  ->  signable step (new, deterministic)  ->  resolver / demo
                                 for each word with NO sign of its own:
                                   1. synonym from the curated table  (exact by default, close on request)
                                   2. fingerspell, if consistently capitalised in the source (a name)
                                 then merge a repeat the substitution itself created
```

Safeguards: a word's own sign **always wins** (a synonym is only a fallback); `FS:` tokens are never touched; the original word is kept next to the new one; every change is logged with its reason; the original gloss file is never edited (tested by hashing);
the table is validated against the real vocabulary every time the index is built, so a synonym can never point at a word that has no sign.

### The synonym table: `data/vocab/synonyms.json`
51 entries, curated by me, **not reviewed by an ISL signer**; each says two *English* words mean (nearly) the same, not that ISL signs them alike.
* `exact` (21, applied by default): interchangeable in almost every context. `HI`/`HEY` to `HELLO`, `THANKS` to `THANK_YOU`, `MOM`/`MUM` to `MOTHER`, `DAD` to `FATHER`, `PHYSICIAN` to `DOCTOR`, `ASSIST` to `HELP`, `US` to `WE` and `ME` to `I` (pronoun signs are not case-marked), `TRADITION` to `TRADITIONS` (the dictionary only has the plural), and a few spelling variants.
* `close` (30, off by default): near-synonyms where a nuance can differ: `SAY` to `TELL`, `MANY` to `A_LOT`, `NOT` to `NO`, `CREATE` to `MAKE`, `SESSION` to `MEETING`, `INTRODUCTORY` to `BASIC`, `EVERYONE` to `ALL`... Turn on per request, see below.
* Validation (`src/islrtc_index/vocab.py`): drops and reports an entry that is malformed, repeats a source word, maps to itself, targets a word with no sign, chains through another synonym, or whose source word already has a sign. All 51 pass today.
* Ordered targets: `CELLPHONE` tries `MOBILE`, then `PHONE`; the first available one is used. An ambiguous target (several signs, no fixed default) is never guessed.

### Name spelling
A word with no sign and no synonym is fingerspelled (`FS:WORD`) only if it is capitalised **everywhere** it appears in the source text, at least once mid-sentence, is 4+ letters, and is one plain word. I tried spaCy entity recognition first and rejected it: it does not recognise `Jainism` and reads `Jain` as a person named "Jane".
Auto-captions capitalise unreliably, so any lower-case appearance disqualifies a word. This is a heuristic and is flagged as such in the change log.

## How to use it

```
# 1. Rebuild the index; this validates the synonym table and writes it into data\islrtc\approved_mappings.json
python run_islrtc_index.py --data-dir C:\Dhiren\internship\islrtc --out data\islrtc        # reads data\vocab\synonyms.json and ..\yap-render\lib\islVocabulary.ts

# 2. Make Phase 1 gloss more signable (never edits the gloss files); writes <id>.signable.json with before/after coverage
python run_signable.py --gloss data\translations\*.gloss.json --out data\signable [--level exact|close|off] [--no-spell-names]
```
The demo and `/api/sign-plan` apply synonyms at request time (default `exact`). Open `http://localhost:3000/demo?synonyms=close` for near-synonyms, or `?synonyms=off` to see the raw gloss. The corner tag shows every substitution, e.g. `avatar · HI → HELLO`.
To add a synonym: add an entry to `data/vocab/synonyms.json`, re-run step 1, and read the "problems" list it prints.

## Measured on the fresh Groq output (120 sentences, 1,174 gloss words)

The Python step and the real resolver agree exactly (checked both ways).

| Synonyms | Words with no sign | Avatar | Dictionary video | Spelled | Used a synonym |
|---|---|---|---|---|---|
| off | 484 (41%) | 193 | 423 | 74 | 0 |
| exact (default) | 458 (39%) | 201 | 441 | 74 | 26 |
| close | 413 (35%) | 228 | 459 | 74 | 71 (26 exact + 45 close) |

`run_signable.py` also spelled 17 consistently capitalised words with no sign: `BHAGWAN`, `PARAMATMA`, `PRABHU`, `CALIFORNIA`, `AGAMAS`, `KEVALNAN`, `SADHANA`, `VANDANA`, `JAIN`, `LORD`. My first version also spelled `DONE` and `THOUGH` (ordinary English words that auto-captions had capitalised);
common English words (spaCy's stop-word list) are now excluded and a test covers it. The synonym layer is worth a few points of coverage; it is not the main lever. `HELLO HI` becomes `HELLO` (a repeat created by the substitution is merged; a repeat the speaker really made is not).
The `close` row includes debatable swaps (`SESSION`->`MEETING`, `INTRODUCTORY`->`BASIC`): use it for demos, not as a claim of equivalence.

## Function words (the biggest remaining gap, not changed)

About a third of the unsigned words are English function words the LLM kept. Some carry meaning and need a sign we do not have (`NOT`, `IF`, `BUT`, `OR`, `WITH`); some are usually dropped in ISL gloss (copulas `IS`/`ARE`, infinitive `TO`, `OF`, `WOULD`).
The rule set already has switches for this (`R-COP-01`, and others; all off pending reviewer sign-off). I did not turn any on: which words ISL gloss omits is a linguistic decision for an ISL-proficient reviewer. If you want, I can produce a per-word frequency list from this run for the reviewer to mark keep/drop,
then wire the approved list in as an explicit, logged rule.

## What this does NOT do
* It does not make the gloss correct ISL. ISL grammar, non-manual signals and context are untouched.
* It does not add signs. 35% of words still have no sign even at `close`, mostly function words and content words missing from both vocabularies.
* The table is small and hand-made. WordNet would find more candidates, but its data is not installed here and its senses are noisy; a candidate generator that writes a *review list* (never auto-applied) is the safe next step if you want to scale the table.

## Next steps that would actually improve Phase 1 quality
1. Done for the first 40 sentences of each transcript (above). The full run (210 requests) is still to do if you want it.
2. Get an ISL-proficient reviewer to approve or reject the 51 synonyms and to fill `references/reference_set.json`; only then can quality be measured against something real.
3. Decide how to treat function words (above), then whether `run_signable.py` should become an automatic last step of the YouTube pipeline (it is deliberately separate today).
