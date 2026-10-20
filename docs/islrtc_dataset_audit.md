# ISLRTC dictionary: dataset audit (Phase 2, Step 2)

Generated 2026-10-20 from `C:\Dhiren\internship\islrtc` (read in place; nothing copied, no media in Git). Reproduce with:

```
python run_islrtc_index.py --data-dir C:\Dhiren\internship\islrtc --out data\islrtc --workers 8      # about 6 min cold (ffprobe), under 1 min with the probe cache
python -m unittest tests.test_islrtc_index                                                          # indexer tests
```

Outputs in `data/islrtc/`: `islrtc_index.sqlite` (query/audit), `manifest.json` (what the resolver loads), `review_queue.csv` (blank `decision`
column for a person), `audit.json` (these numbers), `approved_mappings.json` (written only when a review CSV is applied).
Set `ISLRTC_DATA_DIR` (dataset root) and `ISLRTC_MANIFEST` (path to `manifest.json`); see `yap-render/docs/islrtc_setup.md`.

## What the download is

Hugging Face `silentone0725/Indian_Sign_Language_Data.gov_Rencoded`: a re-encode (H.265) of the ISLRTC *Indian Sign Language Dictionary
(till January 2024)* from data.gov.in. Folders `A`..`Z` and `Numbers` (one level deep), 13,665 `.mp4` files, 73 GB on disk.

**Labels: the download has no label table.** The README promises "Hugging Face-compatible Parquet metadata", but no Parquet/CSV/JSON label
file is in the folder (the only `.metadata` files are Hugging Face download bookkeeping). So the **file name is the only label the dataset gives**.
Consequently every status below is derived mechanically from the file name; nothing was inferred semantically and nothing has been checked by
an ISL signer. If you have the Parquet files from the Hugging Face repo, they should replace file names as the primary source.

## Counts (from `audit.json`)

| Item | Count |
|---|---|
| Video files | **13,665** (folder sizes: S 1404, C 1496, A 1158, P 991, B 954, ... Numbers 43, X 7) |
| Missing files / zero-size files at audit | 0 / 0 |
| Unreadable by ffprobe | 0 (all 13,665 have a video stream) |
| Codec | HEVC (H.265) for all 13,665 |
| Resolution | 1920x1080: 12,006; 1280x720: 1,501; 3840x2160: 87; others: 71 |
| Frame rate | 25 fps: 8,789; 24: 2,423; 30: 1,640; 50: 501 |
| Duration | 1.7 s to 269 s, median 16 s (many entries are explanations/long terms, not 1-2 s signs) |
| Unique raw labels (file stems) | 13,665 |
| **Unique normalised headwords** | **12,086** (case, punctuation, `(Sign_n)` / qualifier stripped) |
| Searchable keys (headwords + source-provided aliases, live entries) | 12,376 |
| Headwords with more than one live sign | **1,108** (2,547 files): 821 only `(Sign_n)`/`(n)` alternates, 215 only sense qualifiers, 72 both. Typical: 2 signs (860), up to 7 |
| Explanation / concept entries (`(Explanation)`, `(Example)`, ...) | 143 |
| Duplicate entries (same label, variant and sense as an earlier file) | 29 |
| Weak/missing label | 1 (`C++`) |

Status of each file: `source_label` 12,621 (clean name, not reviewed by a person) | `needs_review` 872 | `concept` 143 | `duplicate` 29.
Nothing is `verified` yet: that status is reserved for entries a person approves through the review CSV.

## How labels are parsed (all literal patterns seen in the names; see `src/islrtc_index/parse.py`)

* `Name_(Sign_2)` / `(Sign_3)`: alternate sign of the same term -> `variant`. `Name_(2)` and `Name (1)`: also a numbered alternate. I checked the
  60 names ending ` (N)`: **none is a byte-copy of the base file** (58 differ, 2 have no base), so they are different clips, not duplicates.
* `Name_(Explanation)`, `(Explanation_2)`, `(Explaination)`, `(Sign_and_Explanation)`, `(Example)`, `(Meaning)`: `concept` (an explanation, not a
  direct sign). Never auto-selected.
* `Name_(Math)`, `(Academic)`, `(Financial_Term)`, etc. (837 files): kept as a `sense_qualifier`, status `needs_review`, and they **count as a competing
  sign** for the headword, so `Bank` plus `Bank (Financial Term)` is ambiguous rather than silently resolved.
* `Name_(AAP)` (all-caps in brackets, 169): the dictionary itself prints the abbreviation, so it is stored as a source-provided alias.
* `A_or_B_or_C` and `A, B` (75): each part is stored as a source-provided alias; the whole stays the label. More than 8 words: `long_label`, needs review (24).
* `Numbers` folder `5_Five`: pairs digit and word (alias `5`, label `Five`) only when the name has the `digits_word` underscore form (32). `120 Volt` and `404 error` are left as written (a test caught my first version mis-pairing these).
* `... - English` (298): language tag kept, headword without it. Trailing numbers (`Lean_1`, `Form_16`): ambiguous (variant or part of the term), so `needs_review` (19).
* 10 files sit in a folder that is not their first letter (e.g. `Gun_Range` in `R`): flagged, informational.

## Known limitations (please read before trusting any number)

1. **Labels are file names, unreviewed.** A clean name is the dictionary headword, but nothing proves the clip signs that exact English word in the sense your
   gloss means. The resolver marks such videos "unreviewed source label" and can be switched to approved-only (`requireReview`).
2. **Ambiguous headwords get an automatic default sign.** 1,108 headwords (including `Good`, `Name`, `Come`, `India`, `Book`, `Bank`) have several signs. By your instruction each gets ONE fixed default so the same word always shows the same
   sign (see "Default signs" below). The file names do not prove the chosen sign is the best or the only correct one for a context; a reviewer can change any default.
3. Isolated, dictionary-style signs, not continuous signing. Concatenating them is not grammatical ISL.
4. Many clips are long (median 16 s) and 1080p-4K HEVC; Chrome here cannot decode HEVC (see the setup doc), so clips are converted to H.264 on first use.
5. Alias sources are limited to the file names. No synonym table is invented: `automobile` will not find `Car`.
6. Licence: the project owner confirmed use is permitted (2026-10-20). The README states MIT for the re-encode of ISLRTC's dictionary from data.gov.in, asks for citation of both the original dataset and the re-encode, and asks for
   ethical use (no surveillance/profiling). I did not independently verify the data.gov.in terms.

## Default signs (one sign per ambiguous word)

For every key with more than one live sign (1,108 headwords plus alias-key clashes: **1,134 keys**) the indexer picks one default and writes it to `approved_mappings.json` (`select`; `auto_default` lists `[chosen, ...alternates]`).
The rule is deterministic and uses only the file names: (1) the entry that is the word itself beats one that merely lists it as a source-provided alias; (2) `verified`, then clean `source_label`, then `needs_review`; (3) no sense qualifier beats `(Financial_Term)`;
(4) the base file, then `(Sign)`, then `(Sign_2)`, `(Sign_3)`... by number; (5) qualifier name, then path. Result on the real data: 1,082 defaults are the plain base file, 52 are a numbered alternate or qualified sense (the word has no base file), 45 are
entries that still need review (no clean candidate exists). Concept, duplicate, unreadable and rejected entries are never defaults. Asset ids come from the relative path and the rule has no randomness, so re-indexing gives identical defaults
(tested); a reviewer's `default` decision overrides the automatic choice for that key. `review_queue.csv` has an `auto_default` column marking the chosen row of each group.

## Review workflow

`review_queue.csv` has 3,569 rows (every non-clean entry plus every file of a multi-sign headword). Fill `decision` with `verified`, `default` (the sign to use
for that headword; written to `approved_mappings.json`), `reject` or `concept`; optional `label_override` and `aliases_add` (`;`-separated). Then
`python run_islrtc_index.py --data-dir ... --reviews data\islrtc\review_queue.csv`. Unknown ids or decisions are reported and not applied.
