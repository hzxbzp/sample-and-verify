#!/usr/bin/env python3
"""Diagnostic (not a benchmark protocol): a verifier trained on candidates of scenes the planner never saw.

navtest is split into K folds by log; for every fold the verifier (``sav.verifier train``, unchanged) is trained on
the other folds (10% of their logs for early stopping and the margin) and applied to the held-out fold. The picks of
all folds give one navtest PDMS, compared with greedy decoding by a paired bootstrap.

    python navsim/crossfit.py --test test.npz --pdms runs/test/pdms.jsonl --prompts prompts/test.jsonl --lam 0 --out cf.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import sav_navsim as SN                                               # noqa: E402
from sav import verifier as V                                         # noqa: E402


def subset(d: dict, frames: np.ndarray, path: str) -> None:
    """Write the rows of the given frames (renumbered) as a feature file."""
    keep = np.isin(d["frame"], frames)
    remap = {int(f): i for i, f in enumerate(frames)}
    V.save(path, X=d["X"][keep], y=d["y"][keep], q=d["q"][keep], ade=d["ade"][keep], fde=d["fde"][keep],
           frame=np.asarray([remap[int(f)] for f in d["frame"][keep]], np.int64), cand=d["cand"][keep],
           names=d["names"], meta=[d["meta"][int(f)] for f in frames])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--pdms", required=True)
    ap.add_argument("--prompts", required=True, help="prompt file of prep_prompts.py (token -> log)")
    ap.add_argument("--lam", type=float, required=True, help="0: PDMS target, 1: log target")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = V.load(args.test)
    if args.lam > 0:                                                   # log target: drop the PDMS from q
        d["q"] = np.zeros_like(d["q"])
    tok2log = {}
    for line in open(args.prompts):
        r = json.loads(line)
        tok2log[r["token"]] = r.get("log")
    logs = np.array([tok2log.get(m["token"], m["token"]) for m in d["meta"]])
    for i, m in enumerate(d["meta"]):
        m["scene_id"] = logs[i]
    ulogs = np.unique(logs)
    rs = np.random.default_rng(args.seed)
    fold_of_log = dict(zip(rs.permutation(ulogs), np.arange(len(ulogs)) % args.folds))
    fold = np.array([fold_of_log[lg] for lg in logs])
    groups = V.groups_of(d["frame"])
    pick = {}
    tmp = tempfile.mkdtemp(prefix="crossfit_")
    for k in range(args.folds):
        tr_frames, te_frames = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        subset(d, tr_frames, f"{tmp}/tr.npz")
        subset(d, te_frames, f"{tmp}/te.npz")
        vpath = f"{tmp}/v{k}.pt"
        subprocess.run([sys.executable, "-m", "sav.verifier", "train", "--train", f"{tmp}/tr.npz", "--lam", str(args.lam),
                        "--seed", str(args.seed), "--out", vpath], check=True, cwd=os.path.join(HERE, ".."),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        te = V.load(f"{tmp}/te.npz")
        sc, margin = V.scores(vpath, te)
        te_groups = V.groups_of(te["frame"])
        for j, s in enumerate(V.select(sc, te_groups, margin)):       # map back to rows of the full file
            f = int(te_frames[j])
            pick[f] = groups[f][0] + int(s - te_groups[j][0])
        print(f"[crossfit] fold {k}: {len(te_frames)} frames, margin {margin:g}", flush=True)
    pd = {r["token"]: r for r in SN.read_jsonl([args.pdms])}
    si = next(iter(pd.values()))["fields"].index("score")

    def score(row: int, tok: str):
        c = int(d["cand"][row])
        p = pd[tok]
        e = (p.get("greedy") if isinstance(p.get("greedy"), list) else p.get("official")) if c < 0 else p["samples"][c]
        return e[si] if isinstance(e, list) else None

    g, s, ag, as_ = [], [], [], []
    for f in range(len(d["meta"])):
        tok = d["meta"][f]["token"]
        if tok not in pd:
            continue
        r0, rp = groups[f][0], pick[f]
        a, b = score(r0, tok), score(rp, tok)
        if a is None:
            continue
        g.append(a); s.append(b if b is not None else a); ag.append(d["ade"][r0]); as_.append(d["ade"][rp])
    g, s = np.asarray(g), np.asarray(s)
    res = {"lam": args.lam, "folds": args.folds, "n": len(g), "pdms_greedy": float(g.mean()), "pdms_sav": float(s.mean()),
           "delta": SN.boot(s, g), "ade4_greedy": float(np.mean(ag)), "ade4_sav": float(np.mean(as_)),
           "changed": float(np.mean([pick[f] != groups[f][0] for f in range(len(d["meta"]))]))}
    print(json.dumps(res), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
