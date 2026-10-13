"""Prompt of the direct (trajectory-only) VLM planner: system text, then the panorama and the user turn.

The planner must be fine-tuned with exactly this prompt; change it here if your planner was trained with another.
"""
from __future__ import annotations

from typing import Any, Dict, List

from sav.wod.frames import HIST_STEPS, intent_of
from sav.wod.traj import fmt1

SYSTEM = (
    'You are an autonomous-driving trajectory model. Inputs: one panorama stitched from three '
    "forward cameras, the ego vehicle's past 4 s trajectory as 16 (x, y) waypoints in meters "
    "(ego frame, current position at the origin, x forward), and the ego's high-level intent. "
    'Output ONLY <traj>: the future 5 s trajectory as 5 waypoints [x, y] in meters at t = 1, '
    '2, 3, 4, 5 s in the ego frame.'
)
SYSTEM_NO_INTENT = (
    'You are an autonomous-driving trajectory model. Inputs: one panorama stitched from three '
    "forward cameras and the ego vehicle's past 4 s trajectory as 16 (x, y) waypoints in meters "
    '(ego frame, current position at the origin, x forward). '
    'Output ONLY <traj>: the future 5 s trajectory as 5 waypoints [x, y] in meters at t = 1, '
    '2, 3, 4, 5 s in the ego frame.'
)
HISTORY_LABEL = "History (past 4 s, 16 waypoints at 4 Hz, ego frame, meters): "
INSTRUCTION = "Output the future trajectory per the required format."


def history_text(frame: Dict[str, Any]) -> str:
    ps = frame["past_states"]
    xs, ys = list(ps["pos_x"])[-HIST_STEPS:], list(ps["pos_y"])[-HIST_STEPS:]
    return ", ".join(f"[{fmt1(x)}, {fmt1(y)}]" for x, y in zip(xs, ys))


def user_text(frame: Dict[str, Any], intent: bool = True) -> str:
    if intent:
        return f"{HISTORY_LABEL}{history_text(frame)}\nIntent: {intent_of(frame)}\n{INSTRUCTION}"
    return f"{HISTORY_LABEL}{history_text(frame)}\n{INSTRUCTION}"


def chat_messages(frame: Dict[str, Any], intent: bool = True) -> List[Dict[str, Any]]:
    """The conversation in the format of ``processor.apply_chat_template`` (system text, user [image, text])."""
    return [{"role": "system", "content": SYSTEM if intent else SYSTEM_NO_INTENT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(frame, intent)}]}]
