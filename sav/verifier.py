#!/usr/bin/env python3
"""The verifier of sample-and-verify: a small MLP that scores every candidate plan of a frame.

Input is a feature file written by ``sav.wod.build`` (WOD-E2E) or ``navsim/sav_navsim.py build`` (NAVSIM):

    X      [N, D] float32   one row per candidate (row 0 of every frame = the greedy plan)
    q, ade, fde [N]         trust-region / planner score, ADE and FDE of the candidate against the logged future
    y      [N]              training target, q - lambda * ade (lambda = 1 when the file is written)
    frame  [N] int64        frame index of the row (rows of a frame are contiguous)
    cand   [N] int64        candidate index inside the frame (0 / -1 = greedy, else sample index)
    names  json list        feature names, prefixed with their group (traj:, set:, ego:)
    meta   json list        per frame: scene_id, frame_id, v0, cands (WOD) or token (NAVSIM)

``train`` regresses the score of every candidate, centred within its frame, to its centred target
(the expected relative quality), with early stopping on held-out scenes and a switch margin m chosen on
the same held-out frames. ``eval`` prints the mean target / ADE / FDE of greedy decoding, the MBR sample,
the oracle and the verifier.

    python -m sav.verifier train --train F.npz [--val V.npz] --lam 1 --out verifier.pt
    python -m sav.verifier eval --data V.npz --verifier verifier.pt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: candidate-set features; feature files name them "set:<name>" (candidate features "traj:", ego features "ego:")
SET_NAMES = ["mbr_cost", "mbr_rank", "ade_to_greedy", "L_minus_median", "p_move", "p_go", "lp_sum", "lp_mean",
             "is_greedy"]
#: feature groups that ``train --drop`` can remove besides the prefixes traj / set / ego
GROUPS = {"lik": ["set:lp_sum", "set:lp_mean"],
          "cons": ["set:mbr_cost", "set:mbr_rank", "set:ade_to_greedy", "set:L_minus_median", "set:p_move", "set:p_go"],
          "greedyflag": ["set:is_greedy"]}
MARGINS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 1e9)


def poly(c: np.ndarray, n: int) -> np.ndarray:
    """The first n waypoints of a plan with the ego origin prepended."""
    return np.vstack([[0.0, 0.0], c[:n]])


def set_feats(C: np.ndarray, lp: np.ndarray, n: int = 20) -> np.ndarray:
    """Features of every candidate relative to the candidate set (row 0 = greedy, rows 1.. = samples).

    ``lp`` holds [summed log-likelihood, number of tokens] per candidate. Returns [len(C), len(SET_NAMES)]."""
    k = len(C)
    D = np.linalg.norm(C[:, None, :n] - C[None, :, :n], axis=-1).mean(-1)            # pairwise ADE
    smp = np.arange(1, k)
    mbr = np.array([D[i, smp[smp != i]].mean() for i in range(k)])                   # mean ADE to the other samples
    rank = np.argsort(np.argsort(mbr)).astype(float)
    L = np.array([np.linalg.norm(np.diff(poly(c, n), axis=0), axis=1).sum() for c in C])
    med = np.median(L[1:])
    p_move = float((L[1:] >= 1.0).mean())
    p_go = float((L[1:] >= 3.0).mean())
    lp_sum = lp[:, 0].copy()
    lp_mean = lp[:, 0] / np.maximum(lp[:, 1], 1)
    return np.stack([mbr, rank, D[:, 0], L - med, np.full(k, p_move), np.full(k, p_go),
                     lp_sum, lp_mean, (np.arange(k) == 0).astype(float)], 1)


# ======================================================================================
# Feature files, frame groups, selection
# ======================================================================================
def load(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["names"] = json.loads(str(d["names"]))
    d["meta"] = json.loads(str(d["meta"]))
    return d


def save(path: str, **arrays) -> None:
    arrays["names"] = json.dumps(arrays["names"])
    arrays["meta"] = json.dumps(arrays["meta"])
    np.savez_compressed(path, **arrays)


def groups_of(frame: np.ndarray) -> List[np.ndarray]:
    starts = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]])
    ends = np.r_[starts[1:], len(frame)]
    return [np.arange(s, e) for s, e in zip(starts, ends)]


def select(scores: np.ndarray, groups: List[np.ndarray], margin: Optional[float] = None) -> np.ndarray:
    """Top-scored candidate per frame; with ``margin``, keep the greedy plan (row 0 of every group) unless the
    best candidate beats it by more than ``margin``."""
    out = []
    for g in groups:
        b = g[int(np.argmax(scores[g]))]
        if margin is not None and scores[b] - scores[g[0]] <= margin:
            b = g[0]
        out.append(b)
    return np.array(out)


def summary(d: Dict[str, Any], idx: np.ndarray) -> Dict[str, float]:
    return {"y": float(d["y"][idx].mean()), "q": float(d["q"][idx].mean()), "ade": float(d["ade"][idx].mean()),
            "fde": float(d["fde"][idx].mean())}


# ======================================================================================
# Network
# ======================================================================================
def mlp(dim: int):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(dim, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))


def scores(path: str, d: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """Verifier scores of all rows of a feature file and the verifier's switch margin."""
    import torch
    ck = torch.load(path, weights_only=False)
    net = mlp(ck["dim"])
    net.load_state_dict(ck["state"])
    net.eval()
    X = torch.tensor((d["X"][:, ck["keep"]] - ck["mu"]) / ck["sd"])
    with torch.no_grad():
        return net(X).squeeze(-1).numpy(), float(ck["margin"])


def cmd_train(args) -> None:
    import torch
    torch.manual_seed(args.seed)
    tr = load(args.train)
    names = tr["names"]
    drop = set()
    for gname in (args.drop.split(",") if args.drop else []):
        drop |= set(GROUPS[gname] if gname in GROUPS else [n for n in names if n.startswith(gname + ":")])
    keep = np.array([i for i, n in enumerate(names) if n not in drop], dtype=np.int64)
    X = tr["X"][:, keep]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    groups = groups_of(tr["frame"])
    # held-out training scenes for early stopping and the margin, unless a separate validation file is given
    if args.val:
        va = load(args.val)
        tr_groups, (vd, vg) = groups, (va, groups_of(va["frame"]))
    else:
        scenes = np.array([tr["meta"][int(tr["frame"][g[0]])]["scene_id"] for g in groups])
        rs = np.random.default_rng(args.seed)
        hold = set(rs.choice(np.unique(scenes), max(1, len(np.unique(scenes)) // 10), replace=False))
        tr_groups = [g for g, s in zip(groups, scenes) if s not in hold]
        vd, vg = tr, [g for g, s in zip(groups, scenes) if s in hold]
    # target q - lam * ADE for training AND for the validation selection / margin choice
    tr["y"] = (tr["q"] - args.lam * tr["ade"]).astype(np.float32)
    if vd is not tr:
        vd["y"] = (vd["q"] - args.lam * vd["ade"]).astype(np.float32)
    Xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor(tr["y"])
    net = mlp(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    Xv = torch.tensor((vd["X"][:, keep] - mu) / sd)
    best, best_state, bad = -1e9, None, 0
    for ep in range(args.epochs):
        net.train()
        order = np.random.default_rng(args.seed + ep).permutation(len(tr_groups))
        for b in range(0, len(order), args.batch):
            gs = [tr_groups[i] for i in order[b:b + args.batch]]
            loss = 0.0
            s_all = net(Xt[np.concatenate(gs)]).squeeze(-1)
            off = 0
            for g in gs:
                s = s_all[off:off + len(g)]
                off += len(g)
                if args.loss == "listnet":   # probability of being the best candidate: rewards lucky outliers
                    tgt = torch.softmax(yt[g] / args.tau, 0)
                    loss = loss - (tgt * torch.log_softmax(s, 0)).sum()
                else:                        # expected quality relative to the frame mean
                    yr = yt[g] - yt[g].mean()
                    loss = loss + ((s - s.mean()) - yr).pow(2).mean()
            loss = loss / len(gs)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            sv = net(Xv).squeeze(-1).numpy()
        m = summary(vd, select(sv, vg))
        if m["y"] > best + 1e-4:
            best, bad = m["y"], 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"[train] ep {ep:2d} loss {float(loss):.4f} val y {m['y']:.3f} q {m['q']:.3f} "
              f"ade {m['ade']:.3f} fde {m['fde']:.3f}", flush=True)
        if bad >= args.patience:
            break
    greedy = summary(vd, np.array([g[0] for g in vg]))
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        sv = net(Xv).squeeze(-1).numpy()
    by_margin = {m: summary(vd, select(sv, vg, m))["y"] for m in MARGINS}
    margin = max(by_margin, key=by_margin.get)
    print(f"[train] best val y {best:.3f} (greedy y {greedy['y']:.3f} ade {greedy['ade']:.3f}); "
          f"switch margin {margin} -> val y {by_margin[margin]:.3f}  " +
          " ".join(f"{m:g}:{v:.3f}" for m, v in by_margin.items()))
    torch.save({"state": best_state, "mu": mu, "sd": sd, "dim": X.shape[1], "keep": keep,
                "names": [names[i] for i in keep], "margin": float(margin), "loss": args.loss, "lam": args.lam,
                "drop": args.drop}, args.out)


def cmd_eval(args) -> None:
    d = load(args.data)
    groups = groups_of(d["frame"])
    picks = {"greedy": np.array([g[0] for g in groups]),
             "mbr": select(-d["X"][:, d["names"].index("set:mbr_cost")], groups),
             "oracle": select(d["y"], groups)}
    for vp in args.verifier:
        name = os.path.splitext(os.path.basename(vp))[0]
        sc, margin = scores(vp, d)
        picks[name] = select(sc, groups, margin)
        picks[name + "@m0"] = select(sc, groups, 0.0)
    res = {}
    for k, idx in picks.items():
        res[k] = summary(d, idx)
        res[k]["changed"] = float(np.mean(idx != np.array([g[0] for g in groups])))
        print(f"[eval] {k:22s} " + " ".join(f"{m} {v:.3f}" for m, v in res[k].items()), flush=True)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("train")
    a.add_argument("--train", required=True)
    a.add_argument("--val", default=None, help="separate validation file (default: 10%% of the training scenes)")
    a.add_argument("--lam", type=float, default=1.0, help="target = q - lam * ADE")
    a.add_argument("--loss", choices=("mse_rel", "listnet"), default="mse_rel")
    a.add_argument("--tau", type=float, default=0.5, help="temperature of the listnet target")
    a.add_argument("--drop", default="", help="comma-separated feature groups to remove: traj,set,ego,lik,cons,greedyflag")
    a.add_argument("--epochs", type=int, default=40)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--patience", type=int, default=6)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_train)
    a = sp.add_parser("eval")
    a.add_argument("--data", required=True)
    a.add_argument("--verifier", nargs="*", default=[])
    a.add_argument("--json", default=None)
    a.set_defaults(fn=cmd_eval)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
