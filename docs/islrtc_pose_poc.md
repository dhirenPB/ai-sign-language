# ISLRTC pose extraction and avatar retargeting: proof of concept (Phase 2, Steps 5 and 6)

Written 2026-10-20. **Verdict in one paragraph:** landmark extraction from ISLRTC clips works and is documented (20 clips, 15 judged usable by a strict rule).
An experimental retargeting of those landmarks onto the Yap & Render Y Bot (arms, palm orientation, finger flexion) transfers the *gross* arm movement and its
timing, and I inspected the renders side by side with the source clips. It does **not** reproduce signs accurately: hand placement misses face contact, handshapes are
approximate, thumbs, face and torso are not driven, and an out-of-frame arm drifts. This is not a working 3D sign generator, nothing here has been judged by an ISL signer,
and the product still uses the hand-authored Yap animations and the dictionary-video fallback.

## Setup (separate from the main app; nothing downloaded silently)

```
python -m venv C:\isl\venv-pose
C:\isl\venv-pose\Scripts\python -m pip install mediapipe opencv-python-headless numpy        # mediapipe 1.1.1, numpy 2.4.6, opencv-python-headless 5.0.0.93 were used
curl -o C:\isl\models\holistic_landmarker.task https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

The model is 13,683,609 bytes, sha256 `e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8` (recorded in every output file). MediaPipe 1.1.1 no longer ships the old
`solutions.holistic`, so this uses the Tasks API `HolisticLandmarker` in VIDEO mode. FFmpeg and ffprobe must be on PATH (HEVC is decoded by FFmpeg through a pipe, with real per-frame
timestamps from ffprobe). `pip install -e .[pose]` lists the same Python packages.

```
set ISLRTC_DATA_DIR=C:\Dhiren\internship\islrtc
C:\isl\venv-pose\Scripts\python run_islrtc_pose.py --model C:\isl\models\holistic_landmarker.task --out outputs\pose_poc --labels today world professor    # or --assets <ids>
C:\isl\venv-pose\Scripts\python run_islrtc_pose.py --reassess outputs\pose_poc        # re-score stored landmarks with the current quality rules
C:\isl\venv-pose\Scripts\python run_islrtc_retarget.py --glb <yap-render>\public\models\ybot.glb --out <yap-render>\public\lab --labels eat professor   # EXPERIMENT
python -m unittest tests.test_islrtc_pose                                              # quality metrics, selection and format contract (no MediaPipe needed)
```

Outputs per clip in `outputs/pose_poc/` (git-ignored; derived from the dictionary videos, so not for redistribution): `<asset id>.landmarks.json`, `<asset id>.overlay.mp4`
(left: source frame with landmarks, right: landmarks alone), and `summary.json`. A label is only processed when it is unambiguous and clean in the manifest; others are skipped with a warning.
Speed on this PC (CPU): about 2 to 4 s of processing for a 3 s clip.

## Landmark JSON format (`islrtc-landmarks` v1.0)

Top level: `source` (asset id, label, relative path, manifest status), `video` (codec, size, analysed size, fps, probed and analysed frame counts), `extractor` (name, MediaPipe version,
model file name + sha256 + size, confidence, seconds), `coordinate_notes`, `quality`, and `frames[]`. Each frame: `i`, `t` (seconds, from the video's own timestamps),
`pose` and `pose_world` (33 points; `[x,y,z,visibility]`), `left_hand`, `right_hand`, `left_hand_world`, `right_hand_world` (21 points `[x,y,z]`), `face` (478 points), and `face_blendshapes`
(name to score, scores below 0.01 omitted). `null` means not detected in that frame. Image coordinates are normalised to the analysed frame (x right, y down; `z` is relative depth,
not metric); `*_world` is a metric-scale estimate from a single 2D view. Left/right are MediaPipe's labels for the signer; the overlays confirm signer-side labelling on the clips I inspected.

## Results on 20 clips

Selected by label from the Phase 1 sample sentences plus numbers and common words; the one-handed/two-handed split was measured afterwards (10 and 10).

| Clip | Frames | Moving hand(s) measured | Hand coverage in its active window (L / R) | Longest dropout (frames) | Wrist low-visibility frames | Usable for retargeting |
|---|---|---|---|---|---|---|
| Asia | 115 | both | 0.98 / 0.93 | 2 | 45 | yes |
| Audience | 130 | both | 0.86 / 0.80 | 3 | 63 | no |
| Eat | 88 | right | 0.60 / 0.95 | 2 | 45 | yes |
| Evening | 64 | right | 0.00 / 1.00 | 0 | 13 | yes |
| First | 78 | right | 0.00 / 0.98 | 1 | 33 | yes |
| Five | 100 | right | 0.00 / 0.98 | 1 | 48 | yes |
| Guest | 111 | both | 1.00 / 0.84 | 5 | 52 | no |
| Have | 77 | both | 1.00 / 1.00 | 0 | 21 | yes |
| Hello | 74 | right | 0.00 / 0.92 | 2 | 34 | yes |
| Mother | 71 | right | 0.00 / 0.98 | 1 | 0 | yes |
| One | 79 | right | 0.00 / 1.00 | 0 | 48 | yes |
| Philosophy | 293 | both | 1.00 / 0.98 | 1 | 6 | yes |
| Professor | 108 | both | 0.98 / 1.00 | 1 | 41 | yes |
| Speaker | 492 | both | 0.99 / 0.98 | 2 | 31 | yes |
| Teach | 73 | both | 1.00 / 0.98 | 1 | 17 | yes |
| Ten | 75 | right | 0.00 / 0.98 | 1 | 26 | yes |
| Today | 62 | right | 0.00 / 0.86 | 4 | 9 | no |
| Water | 81 | right | 0.00 / 0.97 | 1 | 41 | yes |
| We | 75 | both | 1.00 / 0.86 | 2 | 37 | no |
| World | 862 | both | 0.99 / 0.97 | 4 | 0 | no |

"Usable" is deliberately strict: pose in at least 90% of frames, and every hand that moves tracked in at least 90% of its own *active window* (first to last detected frame; a hand at rest
out of frame before and after the sign does not count against the clip), with no gap longer than 3 frames (too long to interpolate honestly). Quality issues seen:

* **Short mid-sign dropouts** (1 to 5 frames) in most clips, worst in `Guest`, `World` and `Today`; two-handed signs where the hands overlap (`Audience`, `We`) lose a hand more often.
* **Frequent low wrist visibility** (a third to a half of frames in many one-handed clips) because the resting arm is out of frame; the pose model still emits a position, so arm estimates for that side are unreliable.
* **Jittery wrist tracking** on fast signs (`Audience`, `We`, `Guest`, `Water`).
* Clip crops end at the waist, so hands at rest are simply not visible. Faces are detected and 52 blendshape scores are stored, but nothing can use them (see below).

Landmarks are observations of a 2D video: they are not a transcription of the sign and not bone rotations.

## Avatar rig audit (Yap & Render `public/models/ybot.glb`, read from the file)

| Need | Rig | Verdict |
|---|---|---|
| Shoulder / upper arm / forearm / wrist rotation | `Shoulder`, `Arm`, `ForeArm`, `Hand` per side; rest = T-pose with identity local rotations | available |
| Individual finger articulation | 20 joints per hand (4 per finger including the thumb); Yap's own signs drive 3 per finger plus the thumb | available (flexion used; finger spread untested) |
| Torso, head and neck posture | `Hips`, `Spine`, `Spine1`, `Spine2`, `Neck`, `Head` | available, not driven by the experiment |
| Facial / non-manual expression | **only `LeftEye` / `RightEye` joints; no jaw, brow or mouth bones; 0 morph targets.** The "Human" face is a static procedural overlay | **not available** |
| Animation format | Yap signs are per-bone Euler *deltas* applied step by step, not clips | the experiment bypasses it and assigns local quaternions directly |

## Retargeting experiment (`src/islrtc_pose/retarget.py`, `run_islrtc_retarget.py`, lab page `yap-render/app/lab/retarget`)

Method: direction-based. Upper arm and forearm are aimed along the landmark directions (pose world landmarks, axes mapped to the avatar), the hand is rotated to the palm frame
(wrist, index and pinky base, middle base) when the hand model saw the hand, and finger joints get a flexion angle from the landmark angles about the bone's local Z (right hand +, left hand -,
the convention Yap's own signs use), with a 3-frame moving average. Everything else stays at rest, and every driven bone is written on every frame. Self-check with forward kinematics:
elbow and wrist positions are within about 4 to 9% (mean) of arm length of the shoulder-width-scaled source across the six clips processed (max about 20%). That only tests geometry.

Rendered and inspected (headless Chrome, avatar vs source frame; see `docs/islrtc_samples/retarget_*_source_vs_avatar.png`): `Eat` (one hand) and `Professor` (two hands).

* **Works:** arm raise and lowering, forearm bend, correct left/right, and timing follows the source (frame times are copied). In `Professor` the two-arm configuration (one forearm up, the other across the body) is recognisable.
* **Does not work:**
  1. *Contact and placement.* In `Eat` the signer's hand is at the mouth; the avatar's hand ends at neck/shoulder height beside the face. The Y Bot has a different head-to-shoulder ratio and the solver only matches directions, so face contact is lost. Needs IK to head-relative targets.
  2. *Handshape.* The signer's bunched fingers become a loose fist; thumbs are not driven (no reliable axis convention yet) and finger spread is ignored.
  3. *Palm orientation / forearm roll.* Shortest-arc aiming leaves forearm twist arbitrary and the hand absorbs it, so palms look plausible in some frames and visibly off in others (the flat blade hand in `Professor` renders as an open hand).
  4. *Out-of-frame arm.* The resting arm drifts outward in `Eat` because its pose estimate is unreliable and the solver trusts it.
  5. *Not retargeted at all:* torso and head, shoulders, facial and other non-manual features, mouth patterns.
* **Not validated:** no signer has judged any render; only 2 of the 6 processed clips were inspected visually; I did not examine jitter at playback speed.

## What is missing for true 3D sign playback

1. An IK / constraint layer for contact and location (face, body, other hand) using head- and torso-relative targets scaled to the avatar's proportions.
2. Reliable hand orientation and finger *and thumb* retargeting (a calibrated mapping of MediaPipe hand landmarks to this rig's finger axes, including abduction), validated per handshape.
3. Handling of occluded or out-of-frame limbs (hold a rest pose when visibility is low) and gap interpolation with confidence.
4. A rig that can show non-manual signals (face bones or blendshapes), since ISL grammar uses them; the current avatar cannot.
5. Torso and shoulder motion, and a body-frame (not camera-frame) reference.
6. Evaluation by ISL signers; landmark agreement is not evidence of a correct sign.
7. Legal clearance of dictionary-derived landmark data for the intended use.

Until then the working fallback is the recorded dictionary video (see `yap-render/docs/islrtc_setup.md`); it is reported separately from avatar animation everywhere.
