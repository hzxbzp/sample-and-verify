#!/usr/bin/env python3
"""Why greedy plans do not start: the start-time distribution of the logged plans in stationary training frames.

For every frame where the ego is at rest (< 0.5 m/s) the start time is the first second at which the path along
the five 1 Hz waypoints of the logged future (rounded to 0.1 m, as in the planner's targets) reaches 3 m, the same
definition as "starts within 5 s", so the five bins add up to that fraction; "stay" means shorter than 3 m after
5 s. ``P(start at step | not started)`` below 0.5 at every step means that waiting (all later start times plus
stay) outweighs starting, so greedy decoding of this distribution never starts.

    python -m sav.wod.start_time --frames-root TRAIN_FRAMES [--contexts ctx.json] [--json out.json]

``--contexts`` (JSON clip id -> context, e.g. stop_sign) adds the statistics per context.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from sav.wod import frames as F
from sav.wod.traj import points_1hz


def _one(item):
    scene, fid, fdir = item
    try:
        fr = F.load(fdir)
        gt = F.future_xy(fr)
        if gt is None or F.ego_speed(fr) >= F.STAT_V:
            return None
        return f"{scene}_{fid}", [[round(float(x), 1), round(float(y), 1)] for x, y in points_1hz(gt)]
    except Exception:
        return None


def stats(pts: np.ndarray) -> dict:
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(pts), 1, 2)), pts], 1), axis=1), axis=2)
    reached = np.cumsum(seg, axis=1) >= F.GO
    t = np.where(reached.any(axis=1), reached.argmax(axis=1), 5)          # 0..4: start at 1..5 s, 5: stay
    p = np.bincount(t, minlength=6) / len(t)
    left = 1.0 - np.concatenate([[0.0], np.cumsum(p[:5])[:-1]])
    return {"n": int(len(t)), "start_within_5s": float(p[:5].sum()),
            "p_start_at_s": [float(v) for v in p[:5]], "p_stay": float(p[5]),
            "p_start_step_given_not_started": [float(p[k] / left[k]) if left[k] > 0 else None for k in range(5)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--contexts", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        recs = [r for r in pool.imap_unordered(_one, F.list_frames(args.frames_root), chunksize=256) if r]
    ctx = F.load_json_map(args.contexts)
    by = defaultdict(list)
    for clip, pts in recs:
        by["all"].append(pts)
        if clip in ctx:
            by[ctx[clip]].append(pts)
    out = {k: stats(np.asarray(v, float)) for k, v in by.items()}
    for k, s in out.items():
        print(f"{k:14s} n={s['n']:6d} start<=5s {100 * s['start_within_5s']:5.1f}%  bins 1-5 s "
              f"{[round(100 * v, 1) for v in s['p_start_at_s']]}  stay {100 * s['p_stay']:.1f}%  "
              f"P(start|not yet) {[None if v is None else round(100 * v, 1) for v in s['p_start_step_given_not_started']]}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
