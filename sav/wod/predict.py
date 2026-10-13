#!/usr/bin/env python3
"""Apply a trained verifier to candidate sets (e.g. the test split) and write one plan per frame.

The top-scored candidate replaces the greedy plan only if its score is higher by more than the verifier's margin.
Output: ``[{"scene_id", "frame_id", "pred_xy": [[x, y]] * 20, "switched": bool}]``, the input format of
``sav.wod.rfs`` and of a WOD-E2E submission (which needs the 20-point plans per frame).

    python -m sav.wod.build --cands test_cands.jsonl --no-labels --out test.npz
    python -m sav.wod.predict --data test.npz --verifier v.pt --out preds.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sav import verifier as V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--margin", type=float, default=None, help="override the verifier's margin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = V.load(args.data)
    groups = V.groups_of(d["frame"])
    sc, m = V.scores(args.verifier, d)
    idx = V.select(sc, groups, m if args.margin is None else args.margin)
    out = []
    for f, (g, i) in enumerate(zip(groups, idx)):
        meta = d["meta"][f]
        out.append({"scene_id": meta["scene_id"], "frame_id": meta["frame_id"],
                    "pred_xy": np.asarray(meta["cands"][int(i - g[0])]).round(4).tolist(), "switched": bool(i != g[0])})
    json.dump(out, open(args.out, "w"))
    print(f"[predict] {len(out)} frames, switched {np.mean([o['switched'] for o in out]):.1%} -> {args.out}")


if __name__ == "__main__":
    main()
