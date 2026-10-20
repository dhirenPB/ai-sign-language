# Phase 2, Step 1: architecture audit of Phase 1 and Yap & Render

Written 2026-10-19 (Step 1 audit, read-only at the time). **Update 2026-10-20: the open points in section 4 are resolved and Steps 2 to 7 are done; see `docs/phase2_final_report.md`.** The ISLRTC dataset is **not on this machine yet**, so Steps 2 onward are
waiting for it. Yap & Render was cloned from `https://github.com/mitarthpathak/yap-render.git` (commit `bee74ac`) to
`C:\Dhiren\internship\yap-render` (outside this repo; nothing was committed or pushed there).

## 1. The real execution flow

```
YouTube URL
  -> src/isl_gloss/youtube/*            Python: yt-dlp -> FFmpeg -> faster-whisper -> transcript.json/.vtt
  -> src/transcript_cleaner             Python: clean + sentence chunks
  -> src/isl_gloss translate            Python: LLM -> rules -> validation  =>  data/translations/<id>.gloss.json   <== PHASE 1 ENDS HERE
  ------------------------------------------------------------------------------------------------------------
  (no link exists today)                Phase 1 output is never handed to the Yap & Render app
  ------------------------------------------------------------------------------------------------------------
  yap-render  app/page.tsx              user types/speaks English  (NOT Phase 1 gloss)
    -> lib/islTranslator.ts             POST /api/translate-isl (Gemini 2.5 Flash, 3 s timeout) else textToGloss.js offline rules
    -> lib/islVocabulary.ts             sanitizeTokens(): unknown words become single letters
    -> components/avatar/AvatarPlayer   enqueuePhrase(): token -> words.js function, else A-Z fingerspelling
    -> three.js ybot.glb                bone rotations stepped every frame
```

Findings that matter for the integration:

1. **The two halves are not connected.** Yap & Render has its own English-to-gloss step (Gemini, not Groq) and never reads Phase 1
   output. The seam to build is: Phase 1 `final_tokens` -> Yap player. Phase 1 stays as it is.
2. **Phase 1 output contract** (`results[].final_tokens`, an array of strings): plain uppercase glosses (`STUDENT`, `GOOD_MORNING`,
   `RELIGIOUS_STUDIES`), and names as one `FS:NAME_PARTS` token (`FS:JONATHAN_YOUNG`). Underscore means multiword. Five of the
   sentences in the three sample files have no tokens (skipped/failed) and must be tolerated.
3. **Yap's player re-translates anything it does not recognise.** `enqueuePhrase()` calls `looksPreTranslated()`; if any token is
   outside the 164 `SUPPORTED_TOKENS` and not a single letter, the whole phrase goes back through `textToGloss()`. Feeding Phase 1 gloss
   straight in would silently rewrite it. The mixed-media path must bypass this.
4. **The player has no per-sign boundaries.** It flattens every sign into one queue of bone-instruction frames (`runtime.animations`)
   and fires `onStateChange('ready')` only when the whole queue is empty. Mixed playback (avatar, then a video, then avatar) needs
   per-token start/end events, which the current design cannot give; the player needs a small refactor, or a new controller above it.
5. **`FS:` is not understood by Yap.** It would be stripped to letters with the underscore removed, so `FS:JONATHAN_YOUNG` becomes
   `JONATHANYOUNG` with no word gap. The resolver must expand `FS:` itself.
6. **Yap vocabulary is small.** 164 word signs (`animations/Words/*.js`, hand-authored bone deltas) + A-Z. Measured on the three sample
   gloss files (108 tokens): 29 (27%) are Yap signs, 6 are `FS:` names, **73 (68%, 54 unique) have no avatar sign**, e.g. `THIS WE
   HAVE EVERYONE TODAY RELIGION GOOD_MORNING SAY JOIN`. Today they would be fingerspelled letter by letter, which is the gap ISLRTC is
   meant to fill. (Also note Yap's `GOOD_MORNING` is two tokens `GOOD MORNING`, and Phase 1 emits `HI` which Yap lacks.)

## 2. Yap & Render facts (verified in the clone)

| Item | Finding |
|---|---|
| Stack | Next.js 16.3 (App Router), React 19, TypeScript 5.7, three.js 0.185 + react-three-fiber, Tailwind 4. `AGENTS.md` says this Next.js has breaking changes; docs are in `node_modules/next/dist/docs/` (route handlers: `01-app/01-getting-started/15-route-handlers.md`; `params` is a **Promise**). |
| Avatar | One rig: `public/models/ybot.glb` (Mixamo Y Bot, 2.2 MB). "Human" mode is a procedural face + colours added at runtime on the same rig. `brunette.glb` is in the repo but unused (removed from use 2026-08-28, CC BY-NC). |
| Animation format | Not clips. Each sign is a JS function pushing frames of `[boneName, "rotation", axis, targetRadians, "+"/"-"]` onto `runtime.animations`; the renderer eases each bone toward its target at a fixed step (0.05 rad/frame) with a 170 ms hold per key pose. Bone names are `mixamorig*`. |
| Rig controls available | Arms, forearms, hands, all 15 finger joints per hand, neck/head (the existing signs use them). Spine and shoulders are not used by the current signs. **Face: the Y Bot has no facial bones or blendshapes;** the "Human" face is a static procedural overlay. So no non-manual (facial) signing is possible today. |
| Fingerspelling | A-Z, right hand only (`animations/Alphabets/*.js`), hand-authored. |
| Translation API | `POST /api/translate-isl {text}` -> `{tokens, mode, model}`; needs `GEMINI_API_KEY`, else the client falls back to offline `textToGloss.js`. |
| Bridgeconn / 2D stick figure | **Not present** in this repo or anywhere on this machine. Nothing to preserve; the old 2D renderer lives elsewhere. |
| Repo health | `tsc --noEmit` passes. `npm run build` passes, but `next.config.mjs` sets `typescript.ignoreBuildErrors: true`, so the build does not type-check. **There is no lint script and no test runner.** Phase 2 must add a test runner (Vitest suggested) and use `tsc` as the real check. |
| Licensing | Mixamo licence for ybot; team credits in README. No repo licence file found, so redistributing or building on it needs the owners' permission. |

## 3. Proposed architecture (to be confirmed once ISLRTC is present)

Decisions I would take, each reversible:

* **Where code lives.** Player, resolver and media endpoint in `yap-render` (they need its animation table and three.js). Dataset
  indexer, pose extraction, tests of those and the audit reports in this repo (Python, where FFmpeg/MediaPipe tooling already is).
  Neither repo is rewritten; Phase 1 files are untouched.
* **Manifest.** Python script builds `islrtc_manifest.sqlite` (audit/query) and a JSON export the resolver loads. Paths relative to
  `ISLRTC_DATA_DIR`. No media is copied into either repo.
* **Resolver** (`lib/signResolver.ts`, pure and unit-tested): per gloss, in this order: Yap animation, verified ISLRTC video,
  fingerspelling, `unresolved`. Result is a discriminated union `yap_animation | islrtc_video | fingerspelling | unresolved`
  (+ `pose_sequence` later), each carrying the source asset id. Exact and approved-alias matching only; multiword before single word;
  fuzzy matches go to a review file, never auto-chosen.
* **Handoff.** A small loader accepts a Phase 1 `gloss.json` (or pasted token list) and feeds the tokens to the resolver, bypassing
  `looksPreTranslated`/`textToGloss`. Needs `FS:` handling (letters with word gaps).
* **Player.** A sequencing controller above `AvatarPlayer`: it walks the resolved list, runs avatar signs through the existing queue,
  waits for the queue to drain (needs a new per-sign completion callback), plays video items in a `<video>` panel, and shows a
  per-item badge (avatar / dictionary video / fingerspelling / unresolved) with the asset id. Errors and missing media advance with a
  visible marker rather than stalling.
* **Media endpoint.** `app/api/islrtc/[id]/route.ts`: id looked up in the manifest, path joined to `ISLRTC_DATA_DIR` and
  re-checked to stay inside it, `Range` support, video MIME only. Optional cached H.264 transcode via FFmpeg (found: ffmpeg 9.0 full
  build on PATH, which can decode HEVC) only for clips that are actually requested, into a cache folder; originals untouched.
* **Pose experiment** (offline, separate): MediaPipe Holistic on ~20 clips -> JSON + overlay videos + failure log -> a feasibility
  report against the rig above. Honest expectation from the audit: the rig has no face control and the signs are authored as per-bone
  Euler deltas, so mapping 2D/relative landmarks onto them needs an inverse-kinematics step that does not exist yet; I will not
  fabricate rotations if that cannot be validated by rendering and comparing.

## 4. Open points I need from you

1. **ISLRTC location.** Path to the dataset root (read in place; C: has 81 GB free, so it is probably elsewhere) and any
   licence/terms file or website it came from.
2. **Repo for the Yap changes.** The clone is the original authors' repo. Should I (a) work in a local branch of this clone, (b) copy
   it into your own repo/fork, or (c) put the Yap code inside `ai-sign-language`? I recommend (a) on a branch `phase2-islrtc`, you push to your own fork.
3. **Test runner.** OK to add Vitest to yap-render (dev dependency)?

Not done and not claimed: no ISLRTC audit, resolver, player change, media endpoint or pose extraction yet.
