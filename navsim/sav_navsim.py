#!/usr/bin/env python3
"""Sample-and-verify on NAVSIM for off-the-shelf text-waypoint planners (Curious-VLA, ReCogDrive-VLM): the WOD-E2E
verifier with NAVSIM's 4 s horizon. Plans (2 Hz poses) and the logged future are resampled to 16 points at 4 Hz.

  build  candidates (gen_candidates.py) -> feature file of ``sav.verifier``. Row 0 of a frame is the plan the official
         agent executes (greedy, or the agent's fallback when the greedy output cannot be parsed). Targets: ade / fde
         at 4 s against the logged future; q = PDM score of the candidate if ``--pdms`` is given, else 0. Hence
         ``sav.verifier train --lam 1`` on a file built without ``--pdms`` uses the log target (-ADE) and
         ``--lam 0`` on a file built with ``--pdms`` the PDMS target.
  eval   PDMS (official scorer) and ADE of greedy, the official decoding, simple rules, SaV and oracles, with paired
         bootstrap intervals; start rates on stationary frames whose logged future moves >= 3 m in 4 s.

    python navsim/sav_navsim.py build --cands runs/train/cands_r*.jsonl --logs splits/navtrain_sub.tsv --out train.npz
    python navsim/sav_navsim.py build --cands runs/test/cands_r*.jsonl --pdms runs/test/pdms.jsonl --out test.npz
    python -m sav.verifier train --train train.npz --lam 1 --out v_log.pt
    python navsim/sav_navsim.py eval --test test.npz --pdms runs/test/pdms.jsonl --verifier v_log.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sav import verifier as V                                         # noqa: E402

N_WP = 16                                                             # 4 s at 4 Hz
COMMANDS = ("turn left", "keep forward", "turn right", "unknown")
TRAJ_NAMES = ["x1", "y1", "x3", "y3", "x4", "y4", "L1", "L3", "L4", "v1", "v3", "v4", "vmax", "acc_max",
              "acc_min", "jerk", "head4", "moves", "goes", "L4_minus_cv"]
EGO_NAMES = ["v0", "a0", "stat_dur", "hist_len", "hist_yaw"] + [f"cmd_{c.replace(' ', '_')}" for c in COMMANDS]
NAMES = [f"traj:{n}" for n in TRAJ_NAMES] + [f"set:{n}" for n in V.SET_NAMES] + [f"ego:{n}" for n in EGO_NAMES]
STAT_V, GO = 0.5, 3.0


def to_4hz(poses, n_poses: int = 8) -> np.ndarray:
    """First n_poses poses at 2 Hz (ego frame) -> 2 * n_poses positions at 4 Hz, linear in time from the origin."""
    p = np.asarray(poses, float)[:n_poses, :2]
    t = 0.5 * np.arange(len(p) + 1)
    p = np.vstack([[0.0, 0.0], p])
    tq = np.minimum(0.25 * np.arange(1, 2 * n_poses + 1), t[-1])
    return np.stack([np.interp(tq, t, p[:, 0]), np.interp(tq, t, p[:, 1])], 1)


def traj_feats(c: np.ndarray, v0: float) -> List[float]:
    """The WOD-E2E candidate features at 1, 3 and 4 s (the NAVSIM horizon) instead of 1, 3 and 5 s."""
    p = V.poly(c, N_WP)
    seg = np.diff(p, axis=0)
    step = np.linalg.norm(seg, axis=1)
    L = np.cumsum(step)
    sp = step * 4.0
    acc = np.diff(np.concatenate([[v0], sp])) * 4.0
    jerk = np.abs(np.diff(acc)).mean() * 4.0
    head = float(np.arctan2(seg[-1, 1], seg[-1, 0])) if step[-1] > 1e-3 else 0.0
    return [c[3, 0], c[3, 1], c[11, 0], c[11, 1], c[15, 0], c[15, 1], L[3], L[11], L[15],
            sp[3], sp[11], sp[15], sp.max(), acc.max(), acc.min(), jerk, head,
            float(L[15] >= 1.0), float(L[15] >= 3.0), L[15] - 4.0 * v0]


def ego_feats(rec: dict):
    v = np.asarray(rec["velocity"], float)
    a = np.asarray(rec["acceleration"], float)
    v0, a0 = float(np.hypot(*v[:2])), float(np.hypot(*a[:2]))
    h = np.asarray(rec.get("his_trajectory") or [[0.0, 0.0, 0.0]], float)[:, :2]     # 4 poses at 2 Hz, last = now
    step = np.linalg.norm(np.diff(h, axis=0), axis=1) if len(h) > 1 else np.zeros(1)
    still = 0
    for s in step[::-1] / 0.5:
        if s >= 0.3:
            break
        still += 1
    stat_dur = still * 0.5 if v0 < 0.3 else 0.0
    hist_len = float(step.sum())
    d = h[-1] - h[0]
    hist_yaw = float(np.arctan2(d[1], d[0] + 1e-6)) if hist_len > 1 else 0.0
    cmd = str(rec.get("instruction") or "").lower()
    return [v0, a0, stat_dur, hist_len, hist_yaw] + [float(cmd == c) for c in COMMANDS], v0


def path_len(c: np.ndarray, n: int = N_WP) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], c[:n]]), axis=0), axis=1).sum())


def read_jsonl(patterns: List[str]) -> List[dict]:
    out, seen = [], set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for line in open(path):
                r = json.loads(line)
                if r["token"] not in seen:
                    seen.add(r["token"])
                    out.append(r)
    return out


def _score(entry, si: int) -> Optional[float]:
    return float(entry[si]) if isinstance(entry, list) else None


def cmd_build(args) -> None:
    logs = {}
    if args.logs:
        for line in open(args.logs):
            tok, log = line.split()[:2]
            logs[tok] = log
    pd = {r["token"]: r for r in read_jsonl([args.pdms])} if args.pdms else {}
    X, Q, ADE, FDE, FR, CAND, meta = [], [], [], [], [], [], []
    n_skip = 0
    for r in read_jsonl(args.cands):
        smp = [(i, s) for i, s in enumerate(r.get("samples") or []) if s is not None and len(s) >= 8]
        row0 = r.get("greedy") if r.get("greedy") is not None else r.get("official")
        gt = r.get("gt_trajectory")
        if row0 is None or len(smp) < 4 or not gt or len(gt) < 8:
            n_skip += 1
            continue
        ego, v0 = ego_feats(r)
        C = np.stack([to_4hz(row0)] + [to_4hz(s) for _, s in smp])
        glp = r.get("greedy_lp") or [max(r["samples_lp"][i][0] for i, _ in smp),
                                     int(np.median([r["samples_lp"][i][1] for i, _ in smp]))]
        lp = np.asarray([glp] + [r["samples_lp"][i] for i, _ in smp], float)
        g = to_4hz(gt)
        d = np.linalg.norm(C - g[None], axis=-1)
        sf = V.set_feats(C, lp, N_WP)
        for ci, c in enumerate(C):
            X.append(traj_feats(c, v0) + list(sf[ci]) + ego)
        if pd:
            p = pd.get(r["token"], {})
            si = p.get("fields", ["score"] * 7).index("score") if p else 6
            q0 = _score(p.get("greedy"), si) if r.get("greedy") is not None else _score(p.get("official"), si)
            q = [q0] + [_score((p.get("samples") or [None] * (i + 1))[i], si) for i, _ in smp]
            q = [np.nan if v is None else v for v in q]
        else:
            q = [0.0] * len(C)
        Q += q
        ADE += list(d.mean(1))
        FDE += list(d[:, -1])
        FR += [len(meta)] * len(C)
        CAND += [-1] + [i for i, _ in smp]                    # -1 = row 0 (greedy / official), else sample index
        meta.append({"scene_id": logs.get(r["token"], r["token"]), "frame_id": r["token"], "token": r["token"],
                     "v0": v0, "gt_len": path_len(g), "row0_is_greedy": r.get("greedy") is not None})
    Q = np.asarray(Q, np.float32)
    if pd and np.isnan(Q).any():                               # unscorable candidates: worst score of the frame
        FRa = np.asarray(FR)
        for f in np.unique(FRa[np.isnan(Q)]):
            m = FRa == f
            Q[m & np.isnan(Q)] = np.nanmin(Q[m]) if np.isfinite(Q[m]).any() else 0.0
    ADE, FDE = np.asarray(ADE, np.float32), np.asarray(FDE, np.float32)
    V.save(args.out, X=np.asarray(X, np.float32), y=(Q - ADE).astype(np.float32), q=Q, ade=ADE, fde=FDE,
           frame=np.asarray(FR, np.int64), cand=np.asarray(CAND, np.int64), names=NAMES, meta=meta)
    print(f"[build] {len(meta)} tokens, {len(Q)} candidates, skipped {n_skip} -> {args.out}", flush=True)


def boot(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    rs = np.random.default_rng(seed)
    d = a - b
    m = [d[rs.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cmd_eval(args) -> None:
    d = V.load(args.test)
    groups = V.groups_of(d["frame"])
    pd = {r["token"]: r for r in read_jsonl([args.pdms])}
    fields = next(iter(pd.values()))["fields"]
    si = fields.index("score")
    toks = [d["meta"][int(d["frame"][g[0]])]["token"] for g in groups]
    keep = [i for i, t in enumerate(toks) if t in pd]
    print(f"[eval] {len(groups)} tokens with features, {len(keep)} with PDMS", flush=True)

    def entry(row: int, tok: str):
        c = int(d["cand"][row])
        p = pd[tok]
        e = (p.get("greedy") if isinstance(p.get("greedy"), list) else p.get("official")) if c < 0 \
            else (p.get("samples") or [None] * (c + 1))[c]
        return e if isinstance(e, list) else None

    def summarize(pick: Dict[int, int], name: str) -> dict:
        S, sub, A, Fd, chg = [], [], [], [], []
        for i in keep:
            g, tok = groups[i], toks[i]
            row = pick[i]
            e = entry(row, tok) or entry(g[0], tok)
            if e is None:
                continue
            S.append(e[si]); sub.append(e); A.append(d["ade"][row]); Fd.append(d["fde"][row]); chg.append(row != g[0])
        sub = np.asarray(sub)
        out = {"name": name, "n": len(S), "pdms": float(np.mean(S)), "ade4": float(np.mean(A)), "fde4": float(np.mean(Fd)),
               "changed": float(np.mean(chg)), **{f: float(sub[:, j].mean()) for j, f in enumerate(fields) if f != "score"}}
        out["_scores"], out["_pick"] = np.asarray(S), pick
        return out

    nm = d["names"]
    res = [summarize({i: groups[i][0] for i in keep}, "row 0 (greedy / official fallback)")]
    off = [pd[toks[i]]["official"][si] for i in keep if isinstance(pd[toks[i]].get("official"), list)]
    res.append({"name": "official decoding", "pdms": float(np.mean(off)), "n": len(off)})
    rs = np.random.default_rng(0)
    res.append(summarize({i: int(rs.choice(groups[i][1:])) for i in keep}, "random sample"))
    res.append(summarize({i: groups[i][1 + int(np.argmax(d["X"][groups[i][1:], nm.index("set:lp_sum")]))] for i in keep},
                         "most likely sample"))
    res.append(summarize({i: groups[i][1 + int(np.argmin(d["X"][groups[i][1:], nm.index("set:mbr_cost")]))] for i in keep},
                         "MBR sample"))
    for vpath in args.verifier or []:
        sc, margin = V.scores(vpath, d)
        for m, tag in ((None, "no margin"), (margin, f"margin {margin:g}")):
            sel = V.select(sc, [groups[i] for i in keep], m)
            res.append(summarize({i: int(s) for i, s in zip(keep, sel)}, f"SaV {os.path.basename(vpath)} ({tag})"))

    def best(i, key):
        vals = [(key(row), row) for row in groups[i]]
        return max((v, row) for v, row in vals if v is not None)[1]
    res.append(summarize({i: best(i, lambda row: (entry(row, toks[i]) or [None] * len(fields))[si]) for i in keep},
                         "oracle: best PDMS"))
    res.append(summarize({i: best(i, lambda row: -float(d["ade"][row])) for i in keep}, "oracle: closest to log"))
    hum = [pd[toks[i]]["human"][si] for i in keep if isinstance(pd[toks[i]].get("human"), list)]
    res.append({"name": "logged future (human)", "pdms": float(np.mean(hum)) if hum else None, "n": len(hum)})
    base = res[0]["_scores"]
    for r in res:
        if "_scores" in r and len(r["_scores"]) == len(base):
            r["delta_vs_row0"] = boot(r["_scores"], base)
    L4 = nm.index("traj:L4")
    st = [i for i in keep if d["meta"][int(d["frame"][groups[i][0]])]["v0"] < STAT_V
          and d["meta"][int(d["frame"][groups[i][0]])]["gt_len"] >= GO]
    starts = {"n_frames": len(st)}
    if st:
        starts["row0"] = float(np.mean([d["X"][groups[i][0], L4] >= GO for i in st]))
        starts["samples"] = float(np.mean([np.mean(d["X"][groups[i][1:], L4] >= GO) for i in st]))
        for r in res:
            if "_pick" in r and r["name"].startswith("SaV"):
                starts[r["name"]] = float(np.mean([d["X"][r["_pick"][i], L4] >= GO for i in st]))
    print(f"{'rule':52s} {'PDMS':>7s} {'dPDMS [95% CI]':>24s} {'ADE4':>6s} {'chg':>5s}")
    for r in res:
        dl = r.get("delta_vs_row0")
        ci = f"{100 * dl[0]:+.2f} [{100 * dl[1]:+.2f},{100 * dl[2]:+.2f}]" if dl else ""
        pdms = 100 * r["pdms"] if r.get("pdms") is not None else float("nan")
        print(f"{r['name']:52s} {pdms:7.2f} {ci:>24s} {r.get('ade4', float('nan')):6.2f} {r.get('changed', float('nan')):5.2f}")
    print("start rates (stationary, log moves >= 3 m in 4 s):", json.dumps(starts))
    if args.json:
        json.dump({"rules": [{k: v for k, v in r.items() if not k.startswith("_")} for r in res], "starts": starts},
                  open(args.json, "w"), indent=1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("build")
    a.add_argument("--cands", nargs="+", required=True)
    a.add_argument("--logs", default=None, help="token<TAB>log file (scene grouping for the held-out split)")
    a.add_argument("--pdms", default=None, help="PDMS of the candidates (score_candidates.py) -> q")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_build)
    a = sp.add_parser("eval")
    a.add_argument("--test", required=True)
    a.add_argument("--pdms", required=True)
    a.add_argument("--verifier", nargs="*", default=None)
    a.add_argument("--json", default=None)
    a.set_defaults(fn=cmd_eval)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
