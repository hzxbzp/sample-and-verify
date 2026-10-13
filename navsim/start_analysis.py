#!/usr/bin/env python3
"""Starts on NAVSIM: frames where the ego has not moved during the whole 1.5 s history (stat_dur >= 1.5) and the
logged future moves >= 3 m in 4 s (go) or < 1 m (wait); start rate of row 0 (greedy) and of the samples.

    python navsim/start_analysis.py test_curious_sft.npz test_recogdrive.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sav import verifier as V                                         # noqa: E402

for path in sys.argv[1:]:
    d = V.load(path)
    nm = d["names"]
    sd, L4 = nm.index("ego:stat_dur"), nm.index("traj:L4")
    groups = V.groups_of(d["frame"])
    out = {}
    for tag, cond in (("go", lambda m: m["gt_len"] >= 3.0), ("wait", lambda m: m["gt_len"] < 1.0)):
        fr = [g for g in groups if d["X"][g[0], sd] >= 1.5 and cond(d["meta"][int(d["frame"][g[0]])])]
        if fr:
            out[tag] = {"n": len(fr), "greedy_starts": float(np.mean([d["X"][g[0], L4] >= 3.0 for g in fr])),
                        "sample_starts": float(np.mean([np.mean(d["X"][g[1:], L4] >= 3.0) for g in fr]))}
    print(path, json.dumps(out))
