"""Text trajectories of the planner and their conversion to the official 4 Hz grid.

The planner writes 5 waypoints at t = 1..5 s, one decimal each, inside ``<traj>...</traj>``::

    <traj>[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]</traj>

They are upsampled to 20 points at t = 0.25 .. 5 s with shape-preserving PCHIP per coordinate over the knots
t = -0.5, -0.25 (the last two history points), 0 (the origin) and 1..5 s.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from sav.wod.frames import DT, FUT_STEPS

Point = Tuple[float, float]
N_WAYPOINTS = 5
TRAJ_T = np.arange(1, FUT_STEPS + 1, dtype=float) * DT
HIST_ANCHOR_T = (-2.0 * DT, -1.0 * DT)

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)"
_PAIR_RE = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]")
_NUM_RE = re.compile(_NUM)
_TRAJ_TAG_RE = re.compile(r"<traj>(.*?)(?:</traj>|$)", re.S)


def fmt1(v: float) -> str:
    """One decimal, with -0.0 written as 0.0."""
    r = round(float(v), 1)
    return f"{0.0 if r == 0.0 else r:.1f}"


def parse_traj_text(s: Optional[str], n_points: int = N_WAYPOINTS) -> Optional[List[Point]]:
    """The first ``n_points`` (x, y) pairs inside ``<traj>`` (a missing closing tag is fine), or None.

    Bracketed pairs are read first; inside the tag, bare numbers are paired up as a fallback."""
    if not s:
        return None
    m = _TRAJ_TAG_RE.search(s)
    body = (m.group(1) if m else s).strip()
    if not body:
        return None
    pairs = _PAIR_RE.findall(body)
    if len(pairs) < n_points and m is not None:
        nums = _NUM_RE.findall(body)
        pairs = list(zip(nums[0::2], nums[1::2]))
    if len(pairs) < n_points:
        return None
    try:
        out = [(float(a), float(b)) for a, b in pairs[:n_points]]
    except ValueError:
        return None
    return out if all(math.isfinite(v) for p in out for v in p) else None


def upsample_pchip(points5, hist_xy=None) -> np.ndarray:
    """5 waypoints at 1 Hz (+ the ego history) -> (20, 2) on the official grid; held at the ends."""
    pts = np.asarray(points5, dtype=float).reshape(-1, 2)
    t_nodes = [0.0] + [float(i + 1) for i in range(len(pts))]
    xy_nodes = [(0.0, 0.0)] + [tuple(p) for p in pts]
    if hist_xy is not None:
        h = np.asarray(hist_xy, dtype=float).reshape(-1, 2)
        if h.shape[0] >= 3 and np.all(np.isfinite(h[-3:-1])):
            t_nodes = list(HIST_ANCHOR_T) + t_nodes
            xy_nodes = [tuple(h[-3]), tuple(h[-2])] + xy_nodes
    t_arr, xy_arr = np.asarray(t_nodes), np.asarray(xy_nodes, dtype=float)
    t_eval = np.clip(TRAJ_T, t_arr[0], t_arr[-1])
    return np.stack([PchipInterpolator(t_arr, xy_arr[:, c], extrapolate=False)(t_eval) for c in range(2)], 1)


def traj_from_text(text: Optional[str], hist_xy=None) -> Optional[np.ndarray]:
    """Planner answer -> (20, 2) plan, or None if it carries no valid ``<traj>``."""
    pts = parse_traj_text(text) if text and "<traj>" in text else None
    if not pts:
        return None
    try:
        return upsample_pchip(pts, hist_xy)
    except Exception:
        return None


def resample20(pred, src_dt: float = DT) -> np.ndarray:
    """(N, 2) points at t = src_dt, 2 src_dt, ... -> (20, 2) at t = 0.25 .. 5 s (linear from the origin, last point
    held). Empty or invalid -> zeros (the official rule for a missing prediction)."""
    if pred is None:
        return np.zeros((FUT_STEPS, 2))
    pts = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or not np.all(np.isfinite(pts)):
        return np.zeros((FUT_STEPS, 2))
    if pts.shape[0] == FUT_STEPS and abs(src_dt - DT) < 1e-9:
        return pts.copy()
    src_t = np.concatenate([[0.0], np.arange(1, len(pts) + 1) * src_dt])
    src_xy = np.concatenate([[[0.0, 0.0]], pts], axis=0)
    return np.stack([np.interp(TRAJ_T, src_t, src_xy[:, 0]), np.interp(TRAJ_T, src_t, src_xy[:, 1])], axis=1)


def ade_fde(pred20, gt20) -> Tuple[float, float, float, float]:
    """Official ADE@1s, ADE@3s, ADE@5s and FDE@5s on the 4 Hz grid (a short prediction is padded with its last point)."""
    pred = np.asarray(pred20, dtype=float).reshape(-1, 2)
    gt = np.asarray(gt20, dtype=float).reshape(-1, 2)
    if pred.shape[0] < gt.shape[0]:
        pred = np.concatenate([pred, np.repeat(pred[-1:], gt.shape[0] - pred.shape[0], axis=0)])
    n = min(len(pred), len(gt))
    d = np.hypot(pred[:n, 0] - gt[:n, 0], pred[:n, 1] - gt[:n, 1])
    k1, k3, k5 = min(4, n), min(12, n), min(FUT_STEPS, n)
    return float(d[:k1].mean()), float(d[:k3].mean()), float(d[:k5].mean()), float(d[k5 - 1])


def ade(a: np.ndarray, b: np.ndarray) -> float:
    """Mean displacement over the common length (a few rater trajectories are shorter than 20 points)."""
    n = min(len(a), len(b), FUT_STEPS)
    return float(np.linalg.norm(np.asarray(a)[:n] - np.asarray(b)[:n], axis=-1).mean())


def points_1hz(xy: Sequence) -> np.ndarray:
    """The t = 1..5 s points (indices 3, 7, 11, 15, 19) of a 20-point trajectory."""
    return np.asarray(xy, float)[[3, 7, 11, 15, 19]]
