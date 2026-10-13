#!/usr/bin/env python3
"""Selection rules on WOD-E2E candidate sets: RFS, ADE/FDE, failed and false starts, paired bootstrap intervals.

Rules: greedy decoding, a random sample, the most likely sample, the sample mean, the MBR sample (lowest mean ADE
to the other samples), the greedy plan stretched along its path by ``--stretch``, and sample-and-verify with each
given verifier at its own margin and at the margins of ``--margins``.

* RFS (official) on rated frames, with the best-rated trajectory as the start reference;
* ADE@5s / FDE@5s against the logged future on every frame that has one;
* failed start: the ego is at rest (< 0.5 m/s), the reference moves on (5 s path >= 3 m), the plan stays put
  (< 1 m); false start: the reference stays (< 1 m) and the plan moves more than 3 m.
  ``--contexts`` (JSON clip id -> context, e.g. stop_sign) splits the failed starts by context.

    python -m sav.wod.evaluate --data val.npz --frames-root FRAMES --verifier v.pt [--clusters c.json] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np

from sav import verifier as V
from sav.wod import frames as F
from sav.wod import rfs as R


def stretch(traj: np.ndarray, alpha: float) -> np.ndarray:
    """Travel ``alpha`` times as far along the same path (extended along the final heading)."""
    p = np.vstack([[0.0, 0.0], traj[:20]])
    seg = np.diff(p, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    if cum[-1] < 1e-3:
        return traj[:20].copy()
    last = seg[np.flatnonzero(L > 1e-6)[-1]]
    d = last / np.linalg.norm(last)
    ext = p[-1] + d * (cum[-1] * max(alpha, 1.0) + 1.0)
    pe = np.vstack([p, ext])
    ce = np.concatenate([cum, [cum[-1] * max(alpha, 1.0) + 1.0 + cum[-1]]])
    ce = ce + np.arange(len(ce)) * 1e-9                        # strictly increasing for interp
    t = alpha * cum[1:]
    return np.stack([np.interp(t, ce, pe[:, 0]), np.interp(t, ce, pe[:, 1])], 1)


def boot_ci(diff: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def starts(per: Dict[str, dict], logged_len: Dict[str, float], contexts: Dict[str, str]) -> dict:
    """Failed / false starts; the reference is the best-rated trajectory if the frame is rated, else the log."""
    fail, false, by_ctx = [], [], defaultdict(list)
    for clip, r in per.items():
        ref = r.get("ref_len", logged_len.get(clip))
        if ref is None or r["v0"] >= F.STAT_V:
            continue
        if ref >= F.GO:
            fail.append(r["pred_len"] < F.STUCK)
            by_ctx[contexts.get(clip, "all")].append(fail[-1])
        elif ref < F.STUCK:
            false.append(r["pred_len"] > F.GO)
    return {"n_stat_go": len(fail), "failed_start": float(np.mean(fail)) if fail else None,
            "n_stat_stay": len(false), "false_start": float(np.mean(false)) if false else None,
            "failed_start_by_context": {k: (len(v), float(np.mean(v))) for k, v in sorted(by_ctx.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="feature file of sav.wod.build")
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--verifier", nargs="*", default=[])
    ap.add_argument("--margins", default="", help="extra switch margins, e.g. 0,0.5,1,2")
    ap.add_argument("--stretch", type=float, default=None, help="stretch factor of the stretched-greedy baseline")
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--contexts", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    d = V.load(args.data)
    groups = V.groups_of(d["frame"])
    meta = d["meta"]
    cands = [np.asarray(m["cands"], float) for m in meta]
    lp = d["X"][:, d["names"].index("set:lp_sum")]
    mbr = d["X"][:, d["names"].index("set:mbr_cost")]
    rng = np.random.default_rng(args.seed)
    rules: Dict[str, List[np.ndarray]] = {
        "greedy": [C[0] for C in cands],
        "random": [C[1 + rng.integers(len(C) - 1)] for C in cands],
        "most likely": [C[1 + int(np.argmax(lp[g][1:]))] for C, g in zip(cands, groups)],
        "sample mean": [C[1:].mean(0) for C in cands],
        "MBR": [C[int(np.argmin(mbr[g]))] for C, g in zip(cands, groups)],
    }
    if args.stretch:
        rules[f"stretch x{args.stretch:g}"] = [stretch(C[0], args.stretch) for C in cands]
    for vp in args.verifier:
        sc, m_val = V.scores(vp, d)
        ms = [m_val] + [float(x) for x in args.margins.split(",") if x and float(x) != m_val]
        for m in ms:
            idx = V.select(sc, groups, m)
            fr = [int(d["frame"][i]) for i in idx]
            rules[f"SaV {vp.rsplit('/', 1)[-1]} m={m:g}"] = [cands[f][i - groups[f][0]] for f, i in zip(fr, idx)]
    clusters, contexts = F.load_json_map(args.clusters), F.load_json_map(args.contexts)
    logged_len = {}
    for m in meta:
        gt = F.future_xy(F.load(m["fdir"]))
        if gt is not None:
            logged_len[f"{m['scene_id']}_{m['frame_id']}"] = F.path_len(gt)
    res, base = {}, None
    print(f"{'rule':34s} {'RFS':>6s} {'dRFS [95% CI]':>22s} {'ADE5':>6s} {'dADE5 [95% CI]':>22s} {'FDE':>6s} "
          f"{'failSt':>6s} {'falseSt':>7s} {'chg':>5s}")
    for name, plans in rules.items():
        preds = [{"scene_id": m["scene_id"], "frame_id": m["frame_id"], "pred_xy": p.tolist()} for m, p in zip(meta, plans)]
        out = R.run(preds, args.frames_root, clusters)
        per = out["per_frame"]
        keys = list(per)
        rfs = np.array([per[k].get("rfs", np.nan) for k in keys])
        ade5 = np.array([per[k].get("ade5", np.nan) for k in keys])
        if base is None:
            base = (rfs, ade5)
        r = {"rfs": out.get("frame_mean"), "leaderboard": out.get("leaderboard"), "ade5": out["ade5"], "fde": out["fde"],
             **starts(per, logged_len, contexts),
             "changed": float(np.mean([np.abs(p - C[0]).max() > 1e-6 for p, C in zip(plans, cands)]))}
        okr, oka = np.isfinite(rfs), np.isfinite(ade5)
        if name != "greedy":
            if okr.any():
                r["drfs_ci"] = boot_ci(rfs[okr] - base[0][okr])
            if oka.any():
                r["dade5_ci"] = boot_ci(ade5[oka] - base[1][oka])
        res[name] = r
        f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"                  # noqa: E731
        ci = lambda c: "" if c is None else f"{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]"   # noqa: E731
        print(f"{name:34s} {f(r['rfs']):>6s} {ci(r.get('drfs_ci')):>22s} {f(r['ade5']):>6s} {ci(r.get('dade5_ci')):>22s} "
              f"{f(r['fde']):>6s} {f(r['failed_start'], 2):>6s} {f(r['false_start'], 2):>7s} {r['changed']:5.2f}", flush=True)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
