# Reviewed ISL reference set: format and use

**Status: the interface exists; the data does not.** `references/reference_set.json` is an empty template. Nothing in it
was written by the engineering side, and no ISL material has been invented to fill it. Until someone proficient in ISL
supplies reviewed entries, prompt 8.0 runs with no reference block and says so in its system prompt.

## What it is for

Prompt 8.0 can be shown a few reviewed entries *relevant to the sentences in the current request*:

* **notation conventions**: short written rules about how this project writes gloss (e.g. how a condition is marked);
* **vocabulary**: an English word or phrase and the gloss label a reviewer approved for it;
* **sentence pairs**: an English sentence and the gloss a reviewer approved, tagged by the feature it shows.

The model is told these are the only verified ISL material it has. It decides how to use them. The retrieval code never
edits, reorders or corrects a gloss, so this is **not** a rule-based layer and not Layer 2.

## File format (`format_version` "1.0")

```json
{
  "format_version": "1.0",
  "notation_conventions": [
    {"id": "conv-001", "statement": "<one reviewed sentence about notation>",
     "reviewed_by": "<reviewer name>", "reviewed_on": "YYYY-MM-DD", "source": "<where it comes from>"}
  ],
  "vocabulary": [
    {"id": "voc-001", "english": "<word or phrase>", "gloss": "<LABEL>", "notes": "<optional>",
     "reviewed_by": "<reviewer name>", "reviewed_on": "YYYY-MM-DD", "source": "<where it comes from>"}
  ],
  "sentence_pairs": [
    {"id": "pair-001", "english": "<sentence>", "gloss": ["<TOKEN>", "<TOKEN>"],
     "tags": ["condition"], "notes": "<optional>",
     "reviewed_by": "<reviewer name>", "reviewed_on": "YYYY-MM-DD", "source": "<where it comes from>"}
  ]
}
```

Rules enforced at load time (`isl_gloss/references.py`):

* every entry needs a unique `id`, a named `reviewed_by`, a `reviewed_on` date and a `source`;
* an entry that fails any check is **rejected, not repaired**, listed in the load report, and never sent to the model;
* gloss tokens are UPPERCASE (`A-Z`, `0-9`, underscore; an `FS:` prefix is allowed); a sentence-pair gloss is a list;
* pair `tags` come from: `condition negation question number time cause name pronoun comparison emphasis context`.

Check a file without calling a model:

```bash
python -m isl_gloss.cli check-references path/to/reference_set.json
```

## Retrieval

For each request only entries relevant to its **target sentences** are used, capped per category (default 5,
`reference_max_entries`):

* notation conventions: always eligible (they are short and global), up to the cap;
* vocabulary: only when the English word or phrase occurs in a target sentence;
* sentence pairs: ranked by shared meaning features (a condition, a negation, a question, ...) and shared content words,
  ties broken by `id`, so the same input always retrieves the same entries.

When nothing is relevant (or the set is empty) the block is omitted and costs no tokens. Each result records which entry
ids were sent (`reference_entries_supplied` in its validation warnings), so a reviewer can see what the model was shown.

## Use

```bash
python -m isl_gloss.cli compare-prompts --versions 8.0 --references references/reference_set.json -o <new folder>
```

Prompts 5.0, 6.0 and 7.0 ignore `--references`, so their cached results stay valid and comparable.

## What this does not do

* It does not supply ISL knowledge. Its value depends entirely on the reviewed entries someone adds.
* Keyword and tag retrieval is crude. It can miss a relevant entry or return a marginal one.
* Whether reference entries improve the gloss has **not** been measured, because there are none yet.
