# Phase 2 report: ISLRTC dictionary + Yap & Render (2026-10-20)

Nothing is committed in either repo. Phase 1 code is unchanged (only `pyproject.toml` gained an optional `pose` extra and `.gitignore` gained `outputs/`).
Yap & Render work is on the local branch `phase2-islrtc` of the clone at `C:\Dhiren\internship\yap-render` (from `mitarthpathak/yap-render`, base commit `bee74ac`).

## Status by deliverable

| # | Deliverable | Status |
|---|---|---|
| 1 | Architecture report | done: `docs/phase2_architecture_report.md` (Step 1 audit; its "open points" are now answered below) |
| 2 | Dataset audit and manifest/index | done: `docs/islrtc_dataset_audit.md`; `data/islrtc/{islrtc_index.sqlite,manifest.json,review_queue.csv,audit.json}` |
| 3 | Hybrid resolver and mixed-media player | done and run on real data (resolver, `/api/sign-plan`, `/api/islrtc/<id>`, the clean `/demo` page: one input box, one output box). The demo page was driven end to end in headless Chrome (type words, click Sign): avatar, then three dictionary videos, then avatar spelling, then back to idle |
| 4 | Pose-extraction proof of concept and visualisations | done: 20 clips, `docs/islrtc_pose_poc.md`, `docs/islrtc_samples/` |
| 5 | Retargeting feasibility report | done, with an experiment: partial, approximate, not a working 3D sign renderer (`docs/islrtc_pose_poc.md`) |
| 6 | Automated tests and coverage statistics | done (below) |
| 7 | Setup and environment documentation | `yap-render/docs/islrtc_setup.md`, `docs/islrtc_pose_poc.md`, `docs/islrtc_dataset_audit.md` |
| 8 | Changed files, commands, results, limitations | this document |

## Coverage on real Phase 1 output (13 sentences from the 3 sample transcripts; real manifest; no review decisions applied)

106 gloss units (108 tokens; two adjacent-token phrases merged), 77 unique glosses. Occurrences and unique vocabulary are kept apart; nothing is counted twice.

| Source | Occurrences | Unique glosses | Note |
|---|---|---|---|
| Existing Yap 3D avatar animation | 25 | 16 | hand-authored, 164-sign vocabulary |
| ISLRTC dictionary **video** (not avatar) | 42 | 26 | all 42 are *unreviewed source labels* (file names); 4 of them are default signs chosen among alternates |
| Fingerspelling (`FS:` names, avatar) | 6 | 6 | |
| Experimental pose sequence | 0 | 0 | no pose-driven output is wired into playback |
| Unresolved | 33 | 29 | all 33 have no dictionary entry; none is ambiguous any more |

Before the default-sign rule the same input gave 38 dictionary videos and 37 unresolved (4 ambiguous); without the dictionary it had 25 avatar, 6 fingerspelled and 75 unresolved units. Vocabulary view (`scripts/coverage-report.ts`): Yap 164 tokens;
dictionary keys with exactly one sign 10,753; keys with several signs and one fixed default 1,134; keys needing review 489; pose-sequence vocabulary 0. **A successful lookup says nothing about whether the sign is the right one for the context, and playing isolated signs in sequence is
not grammatical, continuous ISL.** The matched pairs are all exact label matches (e.g. `TODAY` to `Today`, `PROFESSOR` to `Professor`) but were only eyeballed, not checked by a signer.

## Tests and checks run

* `python -m unittest discover -s tests -t .` (with `PYTHONPATH=src:.`): **546 tests pass** (the full existing suite plus 69 new, including 27 for the signable-gloss step: parser/normalisation parity, status rules, duplicates, default-sign rule, review application, ffprobe probing and caching, quality metrics, asset selection).
* yap-render `npm test` (Vitest): **78 pass** (3 more real-data tests run when the dataset env is set: 81 pass): resolver (exact, aliases, multiword, ambiguity, review levels, missing/unreadable), Yap animation table consistency (every advertised token has a function), safe serving (invalid ids, traversal, absolute paths, symlink escape when permitted, unsupported type, 503s), byte ranges, a real FFmpeg HEVC-to-H.264 conversion, playback progression through mixed items with failures, coverage counting, gloss input parsing, normalisation parity with Python.
* Real-data end-to-end test (`tests/e2e-real-data.test.ts`, needs `PHASE1_GLOSS_DIR`, `ISLRTC_MANIFEST`, `ISLRTC_DATA_DIR`, optionally `ISLRTC_MAPPINGS`): **3 pass**; every dictionary item from real Phase 1 sentences points at an existing, readable file, no word is ambiguous, and each word keeps the same sign across all sentences; skipped otherwise.
* `npx tsc --noEmit`: clean. `npm run build`: passes (note: the project's build skips type checking). Yap & Render has no lint configuration, so no lint was run.
* Over real HTTP against the real dataset: plan endpoint, `Range` (206/416), HEAD, 6 attack-style ids (all 400/404), H.264 conversion on first request 0.7 to 2.1 s then cached; headless Chrome played 3 real converted clips; HEVC originals are rejected by this Chrome.
* Real MediaPipe run on 20 clips and a rendered comparison of 2 signs on the avatar (see the pose report).

## Files

ai-sign-language (all new unless noted; untracked): `src/islrtc_index/{__init__,__main__,parse,index,cli}.py`, `src/islrtc_pose/{__init__,extract,quality,overlay,cli,retarget,retarget_cli,compare}.py`,
`run_islrtc_index.py`, `run_islrtc_pose.py`, `run_islrtc_retarget.py`, `tests/test_islrtc_index.py`, `tests/test_islrtc_pose.py`, `data/islrtc/*`, `docs/{phase2_architecture_report,islrtc_dataset_audit,islrtc_pose_poc,phase2_final_report}.md`,
`docs/islrtc_samples/*.png`; modified: `pyproject.toml` (`pose` extra), `.gitignore` (your own `isl_out/` change kept; added `outputs/`).

yap-render (branch `phase2-islrtc`, uncommitted): new `lib/sign/{types,normalize,manifest,resolver,range,server,transcode,playback,coverage,input}.ts`, `app/api/sign-plan/route.ts`, `app/api/islrtc/[id]/route.ts`,
`components/sign/GlossDemo.tsx`, `app/demo/{page.tsx,demo.css}`, `app/lab/retarget/page.tsx` (experiment), `scripts/coverage-report.ts`, `tests/*.test.ts` + `tests/fixtures.ts`, `vitest.config.ts`, `docs/islrtc_setup.md`; modified: `package.json` / `package-lock.json` (vitest, `test` and `typecheck` scripts), `.gitignore` (`public/lab/`).
Outside both repos: `C:\isl\venv-pose` (venv), `C:\isl\models\holistic_landmarker.task` (13.7 MB model).

## Main commands run

`python run_islrtc_index.py --data-dir C:\Dhiren\internship\islrtc --out data\islrtc --workers 8` (6 min cold, 51 s with the probe cache); `npx vite-node scripts/coverage-report.ts ...`; `npm test`; `npx tsc --noEmit`; `npm run build`;
`next start` against the real dataset for HTTP and Chrome checks; `run_islrtc_pose.py` (20 clips) and `--reassess`; `run_islrtc_retarget.py` (6 clips); `python -m islrtc_pose.compare` (2 clips).

## Bugs found by the tests/checks during this work (all fixed)

Numbers-folder parsing mis-paired `120 Volt` / `404 error` as digit+word (a Python test caught it); an FFmpeg filter comma mis-escaped in TypeScript (the real FFmpeg test caught it); the `server-only` import broke Vitest (removed);
Turbopack traced the whole project because of env-driven paths (annotated); a needs-review sense could be silently outvoted by a clean entry (now counts as a competing sign); the overlay kept every frame in memory (now streams).
One assumption I checked and dropped: files named `Name (1).mp4` are *not* byte-copies of `Name.mp4` (58 of 60 differ), so they stay alternate signs, not duplicates.

## Limitations (summary)

Labels are file names, not reviewed; nothing in this phase was judged by an ISL signer. The 1,108 words with alternate signs use one fixed default each (consistent, but not shown to be the best sign for every context). HEVC clips need the H.264 conversion in Chrome;
other browsers untested. The avatar cannot show non-manual signals (no face bones or blendshapes). Pose retargeting is approximate and unvalidated. The Yap repository has no licence file. You confirmed the dataset may be used; the README's citation and ethical-use requests are recorded in the setup doc.

## Gloss signability follow-up (2026-10-20)

Synonym fallback (`HI`->`HELLO`, 51 validated entries, `exact` on by default, `close` opt-in) and name spelling, as a deterministic step after Phase 1; see `docs/gloss_signability.md`. New files: `data/vocab/synonyms.json`, `src/islrtc_index/vocab.py`,
`src/isl_gloss/{signable,signable_cli}.py`, `run_signable.py`, `tests/test_signable.py`; Yap side `lib/sign/{types,manifest,resolver,coverage}.ts`, `app/api/sign-plan/route.ts`, `components/sign/GlossDemo.tsx`. Phase 1 was re-run with Groq / prompt 8.0 on the first 40 sentences of each transcript (120 sentences, all translated). On that fresh output (1,174 gloss words) words with no sign went 41% -> 39% (exact) -> 35% (close); 33% of the unsigned non-name words are English function words (OF, AND, BUT, TO, FOR, NOT, WITH...) that the LLM keeps and that neither vocabulary can sign.
I changed no prompt. Details and the open decision on function words: `docs/gloss_signability.md`.

## Decisions and status

1. **Default sign for alternates: done (your instruction).** One deterministic default per ambiguous word, stored in `approved_mappings.json`; the same word shows the same sign everywhere. Details: `docs/islrtc_dataset_audit.md`, "Default signs".
2. **Phrase before avatar: still open, my recommendation is to keep it as is.** Plain version: if the gloss has `I` then `KNOW`, Yap has an avatar animation for each word, but the dictionary also has a single video sign called "I Know". Today the player shows the one dictionary video. The alternative is to show the two avatar animations. Nothing needs approving; tell me which you prefer.
3. **Unreviewed labels:** the default accepts clean file-name labels and marks them "unreviewed source label"; `requireReview` switches to approved-only if you ever want it.
4. **Licence:** you confirmed use is permitted; recorded in the docs.
5. **Commit and push: on hold at your request.** Nothing is committed in either repo; test locally with `scripts\dev-islrtc.ps1` first. Open question for later: whether `data/islrtc/` (about 9 MB of manifest and SQLite) should be committed or ignored, and whether to push the Yap branch to a fork of yours.
