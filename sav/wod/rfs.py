#!/usr/bin/env python3
"""Official WOD-E2E rater feedback score (RFS) and ADE/FDE of a set of plans.

The score itself is the unchanged Waymo implementation (``sav/third_party/rater_feedback_utils.py``). Predictions
are a JSON list ``[{"scene_id", "frame_id", "pred_xy": [[x, y]] * 20}]`` on the 4 Hz grid (t = 0.25 .. 5 s);
a missing plan counts as staying at the origin, a shorter one is resampled like the official preparation.

    python -m sav.wod.rfs --preds preds.json --frames-root FRAMES [--clusters clusters.json] [--out-dir DIR]

``--clusters`` maps a scene id to its scenario cluster (``val_sequence_name_to_scenario_cluster.json`` of the
dataset); with it the leaderboard score (mean over clusters) is reported as well.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from sav.third_party.rater_feedback_utils import get_rater_feedback_score
from sav.wod import frames as F
from sav.wod.traj import ade_fde, resample20


def rfs_scores(preds20: Sequence, rater_trajs: Sequence[Sequence], rater_scores: Sequence[Sequence[float]],
               init_speed: Sequence[float]) -> np.ndarray:
    """Per-frame RFS of one 20-point plan per frame."""
    inference = np.stack([np.asarray(p, dtype=np.float64)[None] for p in preds20], axis=0)
    res = get_rater_feedback_score(
        inference_trajectories=inference, inference_probs=np.ones((len(preds20), 1)),
        rater_specified_trajectories=[[np.asarray(t, dtype=np.float64) for t in ts] for ts in rater_trajs],
        rater_feedback_labels=[np.asarray(s, dtype=np.float64) for s in rater_scores],
        init_speed=np.asarray(init_speed, dtype=np.float64), frequency=4, length_seconds=5,
        output_trust_region_visualization=False)
    return np.asarray(res["rater_feedback_score"], dtype=np.float64).reshape(-1)


def trust_region_scores(cands: np.ndarray, logged: np.ndarray, v0: float) -> np.ndarray:
    """The verifier's training target TR of every candidate: the official RFS with the logged future as the only
    rater trajectory, scored 10 (a score in [4, 10] that tolerates speed-dependent deviations from the log)."""
    n = len(cands)
    return rfs_scores([c[:F.FUT_STEPS] for c in cands], [[logged[:F.FUT_STEPS]]] * n, [[10.0]] * n, [v0] * n)


def run(preds: List[dict], frames_root: str, clusters: Optional[Dict[str, str]] = None,
        out_dir: Optional[str] = None) -> dict:
    """RFS (frame mean and, with ``clusters``, the leaderboard mean over clusters) and ADE/FDE of ``preds``.

    Returns ``dict(frame_mean, leaderboard, per_frame={clip_id: {rfs, ade1, ade3, ade5, fde, v0, pred_len,
    ref_len}}, ade1, ade3, ade5, fde, n)``; ``ref_len`` is the 5 s path length of the best-rated trajectory."""
    clusters = clusters or {}
    rows, P, T, S, V = [], [], [], [], []
    per = {}
    for r in preds:
        sid, fid = str(r["scene_id"]), str(r["frame_id"])
        fr = F.load(F.frame_dir(frames_root, sid, fid))
        p20 = resample20(r.get("pred_xy"))
        clip = f"{sid}_{fid}"
        rec = {"v0": F.ego_speed(fr), "pred_len": F.path_len(p20)}
        gt = F.future_xy(fr)
        if gt is not None:
            rec["ade1"], rec["ade3"], rec["ade5"], rec["fde"] = ade_fde(p20, gt)
        trajs, scores = F.rated_trajectories(fr)
        if trajs:
            rec["ref_len"] = F.path_len(trajs[int(np.argmax(scores))])
            rows.append(clip)
            P.append(p20); T.append(trajs); S.append(scores); V.append(rec["v0"])
        per[clip] = rec
    out = {"n": len(per)}
    for k in ("ade1", "ade3", "ade5", "fde"):
        v = [p[k] for p in per.values() if k in p]
        out[k] = float(np.mean(v)) if v else None
    if rows:
        sc = rfs_scores(P, T, S, V)
        by_cluster = defaultdict(list)
        for clip, s in zip(rows, sc):
            per[clip]["rfs"] = float(s)
            c = clusters.get(clip.rsplit("_", 1)[0])
            if c is not None:
                by_cluster[c].append(float(s))
        out["frame_mean"] = float(sc.mean())
        out["leaderboard"] = float(np.mean([np.mean(v) for v in by_cluster.values()])) if by_cluster else None
    out["per_frame"] = per
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(out, open(os.path.join(out_dir, "rfs.json"), "w"), indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    res = run(json.load(open(args.preds)), args.frames_root, F.load_json_map(args.clusters), args.out_dir)
    f = lambda v: "-" if v is None else f"{v:.4f}"       # noqa: E731
    print(f"n={res['n']}  RFS {f(res.get('frame_mean'))}  leaderboard {f(res.get('leaderboard'))}  "
          f"ADE@3s {f(res['ade3'])}  ADE@5s {f(res['ade5'])}  FDE {f(res['fde'])}")


if __name__ == "__main__":
    main()
