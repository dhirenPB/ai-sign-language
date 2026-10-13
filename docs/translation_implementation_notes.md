# Translation module: implementation notes

Module: `src/isl_gloss/` (standard library only). Spec: `Transcript_to_ISL_Gloss_Hybrid_Translation_Architecture.pdf`.
Inspected 12 Oct 2026. Everything the module outputs is labelled **unreviewed**; nothing here proves ISL correctness.

## What was inspected

| Reference | What I read | Findings |
|---|---|---|
| `mitarthpathak/yap-render` (last push 2026-10-06) | `app/api/translate-isl` route (via summary), repo layout, README | Next.js app. Gemini call: model `gemini-2.5-flash`, temperature 0, thinking budget 0, JSON mime type + array-of-strings response schema, input capped at 600 chars, separate HTTP codes for missing key (500), bad JSON (502), rate limit (429), no usable tokens (422). The prompt restricts output to a fixed `SUPPORTED_TOKENS` list tied to their avatar's animations. Offline fallback lives in `lib/islTranslator.ts` and `components/avatar/textToGloss.js` (**not** inspected in detail, not used, per brief section 3C). |
| `krishnshyam/VirtualISLInterpreter` (last push **2021-05-28**) | `src/translate/spacy_rules.py` in full | spaCy (`en_core_web_sm`) + NLTK WordNet. Works from the **English parse tree**, not from gloss. Steps: merge noun chunks/entities; split on "and"; emit DATE first, subject, clauses (xcomp/ccomp/prep/advcl), direct object, root verb, auxiliary, negation, question word last; lemmatise; drop `{be, do, a, the, of, for, from, to}`. Synonym lookup (`find_syn`) is unused. |

**Licences:** neither repository contains a licence file (GitHub reports none). Without one, all rights are reserved, so
**no code or prompt text was copied from either**. Only general ideas were taken, and everything was re-implemented.
If you want to reuse their code verbatim, ask the authors for a licence first.

## Reused / adapted / new / excluded

| Item | Status | Where | Note |
|---|---|---|---|
| Structured-output LLM call at temperature 0 with JSON schema | adapted (idea) | `llm.py`, `prompt.py` | Own REST client and own prompt. |
| Error taxonomy (missing key, malformed JSON, rate limit, empty output) | adapted (idea) | `llm.py`, `translator.py` | Extended: daily-vs-minute quota, truncation, blocked replies, bounded repair. |
| Fixed avatar sign vocabulary in the prompt | **excluded** | | Animation vocabulary is out of scope. Optional `--vocab` file flags unknown tokens instead. |
| Offline rule engine of Yap & Render | **excluded** | | Not a refinement of LLM output; not required by the brief. |
| "Time first", "question word last", article/copula dropping | adapted (idea) | `rules.py` R-ORD-TIME-01, R-ORD-WH-01, R-ART-01, R-COP-01 | Re-expressed as narrow, source-aware rules on gloss tokens. Only R-ART-01 is on. |
| Droplist words `of for from to do` | **excluded** | | Removing them loses meaning ("from London to Delhi", "do homework"). |
| Lemmatisation | **excluded** | | Would alter glosses the LLM chose deliberately. |
| spaCy dependency parse / SOV reorder | **excluded** | | The brief warns against treating a generic parse or simple SOV reorder as ISL grammar. |
| Negation-after-verb rule | **excluded** | | Needs a reliable verb phrase boundary; would require a parser. Candidate for later with reviewer input. |
| Format normalisation (R-FMT-01), skip-word clean-up (R-SKIP-01) | new | `rules.py` | |
| Batching, resumable cache, rate pacing | new | `translator.py`, `store.py`, `llm.py` | Free-tier friendly. |

## Prompt design: comparison with Yap & Render (read 13 Oct 2026) and the merge (prompt v3.0)

Full source of `app/api/translate-isl/route.ts` was read. Still no licence, so our prompt text is our own wording.

| Topic | Yap & Render | Prompt v3.0 (ours) | Decision |
|---|---|---|---|
| Grammar | "Apply strictly, in this order": time first; SOV; drop articles, copulas, auxiliaries | Same decisive numbered order | **Adopted** (replaces our hedged wording) |
| Negation | End of clause, after the verb | Same, explicit rule | **Adopted** |
| Question words | Always last | Same | **Adopted** |
| Auxiliaries | Drops do/does/will/have | Drops only "do"-support; keeps meaning-bearing modals and tense words | **Adapted**: dropping will/have can lose tense; reviewer to confirm |
| Worked examples | Two (one garbled) | Two clean ones, labelled as illustrations, not gold | **Adopted** |
| Vocabulary list | Fixed avatar sign list, mandatory | None by default; if `--vocab` is given it is put in the prompt (use only those tokens or FS:) | **Adapted** as an option for the later sign-generation phase |
| Synonym mapping | Maps to the nearest listed sign (professor -> TEACHER) | Forbidden: fingerspell or mark uncertain | **Kept ours**: silent substitution changes meaning |
| Pronouns | Merges (we -> I, he/she/they -> PERSON) | Kept distinct | **Kept ours** |
| Names | Letter by letter | One `FS:NAME` token per name | **Kept ours** (reviewer to confirm the notation) |
| Output | Bare string array | Objects: n, gloss, uncertain, notes | **Kept ours** |
| Input | One utterance, silently cut at 600 chars | Numbered batches, never truncated | **Kept ours** |
| Reviewer framing | None | "Candidate to be reviewed by a signer" | **Kept ours** |

The strict ordering means the off-by-default rules R-ORD-WH-01 / R-ORD-TIME-01 now act only as a backstop; they stay off
until a reviewer approves them. The result cache is keyed on the full system prompt, so changing the prompt or supplying
a vocabulary can never reuse stale answers.

**Not yet measured:** v3.0 has not been compared against the first prompt on the 40-sentence set. The attempt on
13 Oct hit Google's temporary overload (503) and then the free daily quota (429). The v1.0 outputs are kept in
`data/translations/evaluation/prompt_v1_baseline/`. After the quota resets, run `python run_translator.py evaluate -o
data/translations/evaluation_v3` and compare the two review sheets.

## Rule set 0.1.0

Run `python run_translator.py rules` for the full table (id, version, kind, effect, assumptions, limitation).

* **On:** R-FMT-01 (token format), R-ART-01 (drop THE/AN, drop A except in spelled sequences), R-SKIP-01 (remove tokens
  that equal skipped words).
* **Off, pending reviewer sign-off:** R-COP-01 (copulas), R-ORD-WH-01 (single WH word to the end of a question),
  R-ORD-TIME-01 (single time word to the front). Turn on with `--enable-rule R-COP-01` etc.

Rules are deterministic and idempotent (tested), run in a fixed order, and log `before`/`after` token lists.

## How the cleaner's decisions are honoured

* `skip_translation_spans` (Gujarati runs, mis-recognised Jain terms, expression cues like "hmm") and `[inaudible]` /
  `[Music]`-style markers are **cut out of the text before it reaches the LLM**, recorded in `skipped_spans` with their
  original positions, and any skipped word the model reintroduces is removed by R-SKIP-01.
* A chunk with nothing left gets status `skipped_no_translatable_text` and costs no request.
* `[inaudible]` chunks are translated for the intelligible rest but forced to `review_required`.
* Side effect to be aware of: cutting "Jane" from "the Jane tradition" leaves "the tradition". That follows the skip
  decision but loses the word "Jain" from the gloss input.

## Error handling

No fallback chain. Failed LLM calls yield `translation_status: failed` with an error kind; nothing else translates.
Malformed replies are repaired in a bounded way (strip code fences, take the outermost JSON object), then the batch is
split in halves (max depth 3), then the entries fail. Missing ids get one re-request. Daily quota exhaustion or an auth
error stops the run; progress is cached, so rerunning the same command resumes. Rule-pass errors keep the candidate,
set `rule_pass_completed: false` and validation `failed`.

## LLM provider and free-tier notes

Default: Groq, `openai/gpt-oss-120b` via the OpenAI-compatible endpoint `https://api.groq.com/openai/v1/chat/completions`
(Bearer key in the `Authorization` header, `GROQ_API_KEY`). Switched from Gemini on 13 Oct 2026; Gemini remains available with
`--provider gemini` (`gemini-3.6-flash`; `gemini-2.5-flash` answers HTTP 404 "no longer available to new users").

What was verified in Groq's documentation (not yet by a live call: no Groq key was available): `openai/gpt-oss-120b` supports
JSON Schema structured outputs with `strict: true` (all properties required, `additionalProperties: false`, which our reply
schema satisfies; converted from the shared schema by `to_json_schema`); reasoning is controlled with `reasoning_effort`
(low | medium | high) and hidden with `include_reasoning: false`; `max_completion_tokens` caps output including reasoning;
429 responses carry a `retry-after` header and a "please try again in ..." message; 413 means the request is too large.

Client behaviour: tokens-per-minute pacing (rolling 60 s window, default 7000, below the reported 8K) on top of requests per
minute; per-minute 429s are retried using the server's wait hint; messages naming a per-day limit (RPD/TPD) or waits over 5 min
stop the run as quota exhausted (resumable from the cache); 413 splits the group and retries; 401/403 auth error; 404 model
unavailable; 5xx and 498 retried with backoff; a custom `User-Agent` is sent because some gateways reject the default Python one.
No fallback to another provider ever happens silently.

Rough budget (estimates, not measurements): about 890 tokens of system prompt plus about 100 of group text plus output and low
reasoning is roughly 1,500 tokens per request. At 7K tokens/min that is about 4 requests/min; at 200K tokens/day about 130
requests/day. Your three transcripts make 210 group requests (153 with `--max-group-sentences 12`), so expect two days on the free
tier unless Groq's prompt caching (cached tokens do not count toward limits) helps. Check the real numbers in your Groq console.

## Not done / needs you

* No live call has been made (no API key was available). The HTTP client is tested against a local mock server that
  speaks the `generateContent` protocol. First real call: `python run_translator.py translate data/structured --limit 10`.
* The evaluation set (`evaluation/eval_sentences.json`, 40 sentences) has **no reference glosses** on purpose. Run
  `python run_translator.py evaluate` and give `eval.review_sheet.csv` to an ISL-fluent reviewer.
* LLM-only vs post-rule comparison: both are kept side by side (`candidate_gloss` / `final_gloss`).
