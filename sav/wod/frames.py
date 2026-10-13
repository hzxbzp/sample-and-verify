"""WOD-E2E frames as used throughout this package.

Every frame is a directory ``<frames_root>/<scene_id>-<frame_id>/`` with

* ``frame.json``          the frame's ego states and labels, in the field names of the WOD-E2E protos:
                            ``past_states``   {pos_x, pos_y, vel_x, vel_y}, 16 steps at 4 Hz, the last one = now
                            ``future_states`` {pos_x, pos_y}, 20 steps at 4 Hz (t = 0.25 .. 5 s; absent on test)
                            ``intent``        GO_STRAIGHT | GO_LEFT | GO_RIGHT (the route command)
                            ``preference_trajectories`` [{pos_x, pos_y, preference_score}] (rated frames only)
                          all positions in the ego frame at t = 0 (metres, x forward, y left); the dict fields
                          may also be stored as their Python literal strings
* ``panorama.png``        the three front cameras (FRONT_LEFT | FRONT | FRONT_RIGHT) stitched side by side,
                          the image the planner sees

Run ``python -m sav.wod.frames --check DIR`` to validate a frames root.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DT = 0.25            # s, the official 4 Hz grid
FUT_STEPS = 20       # 5 s
HIST_STEPS = 16      # 4 s
IMAGE_NAME = "panorama.png"
INTENTS = ("GO_STRAIGHT", "GO_LEFT", "GO_RIGHT")

STAT_V = 0.5         # m/s: the ego is at rest below this speed
GO = 3.0             # m: a plan starts / moves on when its 5 s path is at least this long
STUCK = 1.0          # m: a plan stays put when its 5 s path is shorter than this


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def frame_dir(frames_root: str, scene_id: str, frame_id: str) -> str:
    return os.path.join(frames_root, f"{scene_id}-{frame_id}")


def load(fdir: str) -> dict:
    fr = json.load(open(os.path.join(fdir, "frame.json")))
    for k in ("past_states", "future_states"):
        if k in fr:
            fr[k] = _lit(fr[k])
    return fr


def ego_speed(frame: dict) -> float:
    ps = frame["past_states"]
    return float(np.hypot(ps["vel_x"][-1], ps["vel_y"][-1]))


def history_xy(frame: dict) -> np.ndarray:
    ps = frame["past_states"]
    return np.stack([ps["pos_x"], ps["pos_y"]], 1).astype(float)


def future_xy(frame: dict) -> Optional[np.ndarray]:
    fs = frame.get("future_states") or {}
    if len(fs.get("pos_x") or []) < FUT_STEPS:
        return None
    return np.stack([fs["pos_x"], fs["pos_y"]], 1)[:FUT_STEPS].astype(float)


def intent_of(frame: dict) -> str:
    return str(frame.get("intent") or "GO_STRAIGHT")


def rated_trajectories(frame: dict) -> Tuple[List[np.ndarray], List[float]]:
    """Rater-specified trajectories of a rated frame and their scores (entries without a score are skipped)."""
    trajs, scores = [], []
    for p in frame.get("preference_trajectories") or []:
        if not isinstance(p, dict) or p.get("preference_score", -1) == -1 or not isinstance(p.get("pos_x"), list):
            continue
        trajs.append(np.stack([p["pos_x"], p["pos_y"]], 1).astype(float))
        scores.append(float(p["preference_score"]))
    return trajs, scores


def path_len(xy, n: int = FUT_STEPS) -> float:
    """Arc length of the first n waypoints, starting from the ego origin."""
    p = np.asarray(xy, float)[:n]
    if p.size == 0:
        return float("nan")
    p = np.vstack([[0.0, 0.0], p])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def list_frames(frames_root: str) -> List[Tuple[str, str, str]]:
    """(scene_id, frame_id, frame dir) of every frame directory below ``frames_root``."""
    out = []
    for name in sorted(os.listdir(frames_root)):
        d = os.path.join(frames_root, name)
        if os.path.isfile(os.path.join(d, "frame.json")) and "-" in name:
            scene, fid = name.rsplit("-", 1)
            out.append((scene, fid, d))
    return out


def load_json_map(path: Optional[str]) -> Dict[str, str]:
    """A ``{key: value}`` JSON file (scene -> scenario cluster, or clip id -> context), {} if not given. Values
    of the form {"scenario_cluster": c} (the dataset's cluster file) are unpacked."""
    if not path:
        return {}
    return {k: (v.get("scenario_cluster") if isinstance(v, dict) else v) for k, v in json.load(open(path)).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="frames root to validate")
    args = ap.parse_args()
    n = n_fut = n_rated = 0
    for _, _, d in list_frames(args.check):
        fr = load(d)
        assert len(fr["past_states"]["pos_x"]) >= 3, d
        assert os.path.isfile(os.path.join(d, IMAGE_NAME)), d
        n += 1
        n_fut += future_xy(fr) is not None
        n_rated += bool(rated_trajectories(fr)[0])
    print(f"{n} frames, {n_fut} with the logged future, {n_rated} rated")


if __name__ == "__main__":
    main()
