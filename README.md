<p align="center"><img src="assets/banner.jpg" alt="Sample-and-verify planning" width="100%"></p>

# Greedy Decoding Hides Good Plans: Sample-and-Verify Planning for Vision-Language Driving Models

Code for **sample-and-verify (SaV)**. A fine-tuned vision-language planner writes its plan as text and is almost
always decoded greedily. SaV keeps the planner frozen, adds K temperature samples to its greedy plan, and lets a small
MLP verifier pick the plan to execute. The top-scored candidate replaces the greedy plan only if its score is higher
by more than a margin m. The verifier is trained from logged futures alone, without rater labels: its target for every
candidate is `TR - lambda * ADE`, where TR is the official WOD-E2E rater feedback score computed with the logged future
as the only rater trajectory.

The repository covers
* **WOD-E2E**: candidate sampling, verifier features and targets, verifier training, selection with the margin,
  evaluation (official RFS, ADE/FDE, failed and false starts, selection baselines) and test-set predictions;
* **the start-time analysis** behind the failed starts of greedy decoding (Fig. 2, left);
* **NAVSIM**: the same verifier for public planners (Curious-VLA, ReCogDrive-VLM) with their official prompts,
  decoding, parsers and the official PDM scorer.

Fine-tuning the planners themselves is not part of this repository; SaV works with candidate files from any planner
(format below).

## Qualitative results

Open-loop videos on consecutive WOD-E2E validation frames: the planner (fine-tuned Alpamayo-10B, 16 samples) replans
at every frame (10 Hz) while the camera follows the human driver; played at 0.5x speed. Red: greedy decoding; blue:
sample-and-verify; green dashed: logged future; gray: the 16 samples.

* [Green light, route straight](assets/videos/1_green_light_straight.mp4): greedy decoding waits and then turns right;
  SaV goes straight.
* [Intersection](assets/videos/2_intersection_left_turn.mp4): greedy decoding keeps waiting while the human driver
  turns left; SaV turns left.
* [Night, red turns green](assets/videos/3_night_light_turns_green.mp4): SaV starts the left turn as soon as the light
  turns green; greedy decoding waits 1.6 s longer.

## Installation

```bash
pip install -r requirements.txt          # WOD-E2E: numpy, scipy, torch, transformers, pillow
export PYTHONPATH=$PWD                   # run everything from the repository root
```

The NAVSIM part additionally needs the NAVSIM v1 devkit and data (`NAVSIM_DEVKIT_ROOT`, `OPENSCENE_DATA_ROOT`),
`vllm` for the planners, and the public AutoVLA code for the scene preprocessing (`AUTOVLA_ROOT`).

## Repository layout

```
sav/verifier.py              verifier: candidate-set features, training (centred MSE), margin, selection
sav/wod/frames.py            WOD-E2E frame format and helpers
sav/wod/traj.py              text trajectories, PCHIP upsampling to the 4 Hz grid, ADE/FDE
sav/wod/rfs.py               official rater feedback score (RFS) and ADE/FDE of a prediction file
sav/wod/prompt.py            prompt of the direct (trajectory-only) planner
sav/wod/sample.py            greedy plan + K samples with log-likelihoods (Hugging Face models)
sav/wod/build.py             verifier features (37) and targets of WOD-E2E candidate sets
sav/wod/evaluate.py          selection rules, RFS / ADE / failed starts, paired bootstrap intervals
sav/wod/predict.py           apply a verifier to candidate sets (e.g. the test split)
sav/wod/start_time.py        start-time distribution of the logged plans in stationary frames
sav/third_party/             Waymo's rater feedback score implementation (Apache 2.0, unchanged)
navsim/                      NAVSIM: subset, preprocessing, prompts, candidates (vLLM), PDMS, verifier, cross-fit
scripts/run_wod.sh           WOD-E2E pipeline
scripts/run_navsim.sh        NAVSIM pipeline
```

## WOD-E2E

### Frames

Every frame is a directory `<frames_root>/<scene_id>-<frame_id>/` with `frame.json` (ego states and labels in the
field names of the WOD-E2E protos, positions in the ego frame at t = 0) and `panorama.png` (the three front cameras
FRONT_LEFT | FRONT | FRONT_RIGHT stitched side by side). See `sav/wod/frames.py`; `python -m sav.wod.frames --check DIR`
validates a frames root. We used one frames root per split: training frames, dev frames (held-out training scenes,
early stopping and the margin), the 456 rated validation frames (RFS), the remaining validation frames (ADE/FDE)
and the test frames.

### Pipeline

```bash
# 1. candidates: greedy plan + 16 samples at T = 1 per frame
python -m sav.wod.sample --model PLANNER --frames-root TRAIN --out cands/train.jsonl --k 16
python -m sav.wod.sample --model PLANNER --frames-root DEV   --out cands/dev.jsonl   --k 16
python -m sav.wod.sample --model PLANNER --frames-root VAL   --out cands/val.jsonl   --k 16

# 2. features and targets against the logged future
python -m sav.wod.build --cands cands/train.jsonl --out feats/train.npz
python -m sav.wod.build --cands cands/dev.jsonl   --out feats/dev.npz
python -m sav.wod.build --cands cands/val.jsonl   --out feats/val.npz

# 3. verifier (lambda = 1: TR - ADE; lambda = 0: TR), early stopping and margin on dev
python -m sav.verifier train --train feats/train.npz --val feats/dev.npz --lam 1 --out verifier_lam1.pt

# 4. evaluation: RFS, ADE/FDE, failed / false starts, baselines, bootstrap intervals vs greedy decoding
python -m sav.wod.evaluate --data feats/val.npz --frames-root VAL --verifier verifier_lam1.pt \
    --clusters val_sequence_name_to_scenario_cluster.json

# 5. test split: features without targets, one plan per frame
python -m sav.wod.sample --model PLANNER --frames-root TEST --out cands/test.jsonl --k 16
python -m sav.wod.build --cands cands/test.jsonl --no-labels --out feats/test.npz
python -m sav.wod.predict --data feats/test.npz --verifier verifier_lam1.pt --out preds_test.json
```

`sav.wod.sample` shards with `--shard i --nshards n` and resumes an interrupted output file. Use `--no-intent` for a
planner trained without the route command in its prompt, and edit `sav/wod/prompt.py` if your planner was trained with
another prompt. Ablations: `sav.wod.build --kmax k` (fewer samples), `sav.verifier train --drop traj|set|ego|lik|cons`
(feature groups) and `--loss listnet` (the listwise objective).

### Candidate files

One JSON object per line, from `sav.wod.sample` or from any other planner:

```json
{"scene_id": "...", "frame_id": "...", "fdir": "<frame directory>",
 "greedy_xy": [[x, y], ...20 points at t = 0.25..5 s],
 "samples_xy": [[[x, y], ...], ..., null],
 "samples_lp": [[sum_log_prob, n_tokens], ...]}
```

### Start-time analysis

```bash
python -m sav.wod.start_time --frames-root TRAIN [--contexts contexts.json]
```

For every stationary frame (ego speed < 0.5 m/s) the start time is the first second at which the path along the five
1 Hz waypoints of the logged future reaches 3 m, the same definition as "starts within 5 s", so the bins add up to
that fraction. The script also prints `P(start at step | not started)`: below 0.5 at every step, waiting outweighs
starting, and greedy decoding never starts. `--contexts` maps clip ids (`<scene_id>_<frame_id>`) to a context label
(e.g. `stop_sign`) for per-context statistics.

## NAVSIM

`scripts/run_navsim.sh` lists the steps:

1. `navsim/make_subset.py`: navtrain scenes for the verifier, with a 10% validation split by log;
2. `navsim/run_preprocessing.py`: scene JSONs (ego state, command, logged future) with the public AutoVLA
   preprocessing; the devkit's `run_metric_caching.py` builds the metric caches;
3. `navsim/prep_prompts.py`: the planners' official prompts and front images for every token;
4. `navsim/gen_candidates.py`: greedy plan + K samples with vLLM, using the official chat templates, image
   preprocessing, parsers and fallbacks of Curious-VLA (`--kind curious`) and ReCogDrive-VLM (`--kind recog`);
5. `navsim/score_candidates.py`: official PDM score of every candidate;
6. `navsim/sav_navsim.py build|eval`: verifier features on the 4 s horizon, evaluation with paired bootstrap intervals;
   the verifier is trained with `sav.verifier train` (`--lam 1` on a file built without PDMS: log target;
   `--lam 0` on a file built with PDMS: PDMS target);
7. `navsim/crossfit.py` (diagnostic) and `navsim/start_analysis.py` (starts of stationary egos).

## Third-party code

`sav/third_party/rater_feedback_utils.py` is the rater feedback score of the Waymo Open Dataset, licensed under the
Apache License 2.0 (see its header). The NAVSIM scripts reproduce prompt strings and parsers of the official Curious-VLA
and ReCogDrive agents.
