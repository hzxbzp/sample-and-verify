#!/usr/bin/env python3
"""NAVSIM PDM score (official PDMSimulator/PDMScorer and metric cache, default scoring parameters) of every
candidate written by gen_candidates.py: the official decoding, the greedy plan and the K samples. Each plan is
scored on its first 8 poses (4 s at 2 Hz), as the official agents submit them. The metric cache is built with the
devkit's run_metric_caching.py.

    python navsim/score_candidates.py --cands "runs/test/cands_r*.jsonl" --metric-cache METRIC_CACHE \
        --out runs/test/pdms.jsonl --workers 32
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEVKIT = os.environ["NAVSIM_DEVKIT_ROOT"]
sys.path.insert(0, DEVKIT)
FIELDS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
          "comfort", "driving_direction_compliance", "score")
_SIM = _SCORER = _CACHE = None


def _init(cache_root: str) -> None:
    global _SIM, _SCORER, _CACHE
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import MetricCacheLoader
    cfg = OmegaConf.load(os.path.join(DEVKIT, "navsim/planning/script/config/pdm_scoring/"
                                      "default_scoring_parameters.yaml"))
    _SIM, _SCORER = instantiate(cfg.simulator), instantiate(cfg.scorer)
    assert _SIM.proposal_sampling == _SCORER.proposal_sampling
    _CACHE = MetricCacheLoader(Path(cache_root)).metric_cache_paths


def _score(rec: dict):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    tok = rec["token"]
    if tok not in _CACHE:
        return {"token": tok, "error": "no metric cache"}
    with lzma.open(_CACHE[tok], "rb") as f:
        mc = pickle.load(f)

    def one(poses):
        if poses is None or len(poses) < 8:
            return None
        traj = Trajectory(np.asarray(poses[:8], np.float32), TrajectorySampling(num_poses=8, interval_length=0.5))
        try:
            r = pdm_score(metric_cache=mc, model_trajectory=traj, future_sampling=_SIM.proposal_sampling,
                          simulator=_SIM, scorer=_SCORER)
            return [float(getattr(r, k)) for k in FIELDS]
        except Exception as e:                                                   # report, do not drop the token
            return {"error": f"{type(e).__name__}: {e}"}

    out = {"token": tok, "official": one(rec.get("official")), "greedy": one(rec.get("greedy")),
           "samples": [one(p) for p in rec.get("samples") or []]}
    if rec.get("gt_trajectory"):
        out["human"] = one([p[:3] for p in rec["gt_trajectory"]])                # logged future, for reference
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--metric-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    recs, seen = [], set()
    for pat in args.cands:
        for path in sorted(glob.glob(pat)):
            for line in open(path):
                r = json.loads(line)
                if r["token"] not in seen:
                    seen.add(r["token"])
                    recs.append(r)
    done = set()
    if os.path.isfile(args.out):
        done = {json.loads(l)["token"] for l in open(args.out)}
    todo = [r for r in recs if r["token"] not in done]
    print(f"[score] {len(recs)} tokens, {len(done)} scored, {len(todo)} to score", flush=True)
    with Pool(args.workers, initializer=_init, initargs=(args.metric_cache,)) as pool, open(args.out, "a") as fout:
        for i, row in enumerate(pool.imap_unordered(_score, todo, chunksize=8)):
            fout.write(json.dumps({"fields": FIELDS, **row}) + "\n")
            if (i + 1) % 500 == 0:
                fout.flush()
                print(f"[score] {i + 1}/{len(todo)}", flush=True)
    print("[score] done", flush=True)


if __name__ == "__main__":
    main()
