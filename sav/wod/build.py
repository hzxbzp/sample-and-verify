#!/usr/bin/env python3
"""Verifier features and targets for WOD-E2E candidate sets (the input of ``sav.verifier``).

Per candidate (37 features):
* candidate (20)     positions and path length at 1, 3 and 5 s, speeds, the extreme accelerations, mean jerk,
                     final heading, start indicators and the deviation from constant-velocity travel;
* candidate set (9)  MBR cost (mean ADE to the K samples) and its rank, ADE to the greedy plan, path length
                     relative to the sample median, fractions of moving and starting samples, log-likelihood
                     (sum and per token) and a greedy indicator;
* ego (8)            speed, acceleration, time spent at rest, length and heading change of the 4 s history, and
                     the route command.
Targets against the logged future: q = TR (official RFS with the log as the only rater trajectory, scored 10),
ADE@5s and FDE@5s; y = q - ADE. Test frames (no logged future) get zero targets.

    python -m sav.wod.build --cands cands.jsonl [more.jsonl ...] --out train.npz [--kmax 16] [--no-labels]
"""
from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np

from sav import verifier as V
from sav.wod import frames as F
from sav.wod.rfs import trust_region_scores

TRAJ_NAMES = ["x1", "y1", "x3", "y3", "x5", "y5", "L1", "L3", "L5", "v1", "v3", "v5", "vmax", "acc_max",
              "acc_min", "jerk", "head5", "moves", "goes", "L5_minus_cv"]
EGO_NAMES = ["v0", "a0", "stat_dur", "hist_len", "hist_yaw"] + [f"intent_{k}" for k in F.INTENTS]
NAMES = [f"traj:{n}" for n in TRAJ_NAMES] + [f"set:{n}" for n in V.SET_NAMES] + [f"ego:{n}" for n in EGO_NAMES]


def traj_feats(c: np.ndarray, v0: float) -> List[float]:
    p = V.poly(c, F.FUT_STEPS)
    seg = np.diff(p, axis=0)
    step = np.linalg.norm(seg, axis=1)
    L = np.cumsum(step)
    sp = step * 4.0
    acc = np.diff(np.concatenate([[v0], sp])) * 4.0
    jerk = np.abs(np.diff(acc)).mean() * 4.0
    head = float(np.arctan2(seg[-1, 1], seg[-1, 0])) if step[-1] > 1e-3 else 0.0
    return [c[3, 0], c[3, 1], c[11, 0], c[11, 1], c[19, 0], c[19, 1], L[3], L[11], L[19],
            sp[3], sp[11], sp[19], sp.max(), acc.max(), acc.min(), jerk, head,
            float(L[19] >= 1.0), float(L[19] >= 3.0), L[19] - 5.0 * v0]


def ego_feats(frame: dict) -> Tuple[List[float], float]:
    ps = frame["past_states"]
    v = np.hypot(np.asarray(ps["vel_x"], float), np.asarray(ps["vel_y"], float))
    v0 = float(v[-1])
    a0 = float(v[-1] - v[-5])
    still = 0
    for s in v[::-1]:
        if s >= 0.3:
            break
        still += 1
    px, py = np.asarray(ps["pos_x"], float), np.asarray(ps["pos_y"], float)
    hist_len = float(np.linalg.norm(np.diff(np.stack([px, py], 1), axis=0), axis=1).sum())
    dyaw = float(np.arctan2(py[-1] - py[-5], px[-1] - px[-5] + 1e-6)) if hist_len > 1 else 0.0
    it = str(frame.get("intent") or "")
    return [v0, a0, still * 0.25, hist_len, dyaw] + [float(it == k) for k in F.INTENTS], v0


def read_cands(paths: List[str]) -> List[dict]:
    rows, seen = [], set()
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            key = (r["scene_id"], r["frame_id"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", nargs="+", required=True, help="candidate files of sav.wod.sample")
    ap.add_argument("--kmax", type=int, default=0, help="use only the first k samples per frame (0 = all)")
    ap.add_argument("--no-labels", action="store_true", help="test frames: features only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X, Q, ADE, FDE, FR, CAND, meta = [], [], [], [], [], [], []
    n_skip = 0
    rows = read_cands(args.cands)
    for fi, r in enumerate(rows):
        smp = r["samples_xy"][: args.kmax] if args.kmax else r["samples_xy"]
        lp_all = (r.get("samples_lp") or [[0.0, 1]] * len(r["samples_xy"]))[: len(smp)]
        ok = [i for i, s in enumerate(smp) if s is not None]
        if r.get("greedy_xy") is None or len(ok) < min(4, len(smp)):
            n_skip += 1
            continue
        fr = F.load(r["fdir"])
        ego, v0 = ego_feats(fr)
        C = np.asarray([r["greedy_xy"]] + [smp[i] for i in ok], float)[:, :F.FUT_STEPS]
        lps = [lp_all[i] for i in ok]
        # the greedy plan's likelihood is set to the best sample's (as in the paper's experiments)
        lp = np.asarray([[max(l[0] for l in lps), int(np.median([l[1] for l in lps]))]] + lps, float)
        sf = V.set_feats(C, lp, F.FUT_STEPS)
        for ci, c in enumerate(C):
            X.append(traj_feats(c, v0) + list(sf[ci]) + ego)
        gt = None if args.no_labels else F.future_xy(fr)
        if gt is not None:
            q = trust_region_scores(C, gt, v0)
            d = np.linalg.norm(C - gt[None], axis=-1)
            Q += list(q); ADE += list(d.mean(1)); FDE += list(d[:, -1])
        else:
            Q += [0.0] * len(C); ADE += [0.0] * len(C); FDE += [0.0] * len(C)
        FR += [len(meta)] * len(C)
        CAND += [0] + [i + 1 for i in ok]
        meta.append({"scene_id": r["scene_id"], "frame_id": r["frame_id"], "fdir": r["fdir"], "v0": v0,
                     "cands": C.tolist()})
        if (fi + 1) % 1000 == 0:
            print(f"[build] {fi + 1}/{len(rows)} frames", flush=True)
    Q, ADE = np.asarray(Q, np.float32), np.asarray(ADE, np.float32)
    V.save(args.out, X=np.asarray(X, np.float32), y=Q - ADE, q=Q, ade=ADE, fde=np.asarray(FDE, np.float32),
           frame=np.asarray(FR, np.int64), cand=np.asarray(CAND, np.int64), names=NAMES, meta=meta)
    print(f"[build] {len(meta)} frames, {len(Q)} candidates, skipped {n_skip} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
