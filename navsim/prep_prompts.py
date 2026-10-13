#!/usr/bin/env python3
"""Prompts of the two text-waypoint NAVSIM planners for every token of a split, built from the NAVSIM scene exactly
as their official agents build them (strings copied verbatim from the official code):

  curious     Curious-VLA, navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py (cam_type='single'):
              system "You are an expert driver.", the 4-task CoT prompt with the ego history and command; the
              front image is re-encoded as JPEG by PIL (default quality) exactly as the agent does before sending it.
  recogdrive  ReCogDrive-VLM, navsim/agents/intervl_agent.py (prompt_type='base', cam_type='single'): the system
              message of the agent (lmdeploy meta instruction), the trajectory question, original front image.

Also stores the ego state needed by the verifier and the constant-velocity fallback of Curious-VLA.

    python navsim/prep_prompts.py --split test --tokens-from json/test --out prompts/test.jsonl --img-dir prompts/img_test

Needs the NAVSIM devkit (``NAVSIM_DEVKIT_ROOT``) and data (``OPENSCENE_DATA_ROOT`` with navsim_logs/ and sensor_blobs/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

DEVKIT = os.environ["NAVSIM_DEVKIT_ROOT"]
DATA = os.environ["OPENSCENE_DATA_ROOT"]
sys.path.insert(0, DEVKIT)

CURIOUS_SYSTEM = "You are an expert driver."
RECOG_SYSTEM = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


def format_number(n, decimal_places=2):                                 # identical in both official agents
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)


def curious_prompt(ego_statuses) -> str:
    status_lines = []
    size = len(ego_statuses)
    for i in range(size):
        s = ego_statuses[i]
        status_lines.append(
            f'   - t-{size - i - 1}: ({format_number(s.ego_pose[0])}, {format_number(s.ego_pose[1])}, {format_number(s.ego_pose[2])})'
        )
    status_lines.append(f'   - t-{0}: ({format_number(0)}, {format_number(0)}, {format_number(0)})')
    high_command_one_hot = ego_statuses[-1].driving_command
    nav_commands = ['turn left', 'go straight', 'turn right', 'unknown']
    command_str = [nav_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1]
    command_str = command_str[0] if command_str else "unknown"
    history_trajectory = ", ".join(status_lines)
    Q1 = f"""Suppose you are driving. Let's complete the following tasks step by step.
Input:
- 1 frame of front-view image collected from the ego-vehicle at the present timestep
Picture 1: <image> the front view of the ego-vehicle
- Current high-level intent (string): {command_str}
- 1.5-second past trajectory(3 steps at 2 Hz): {history_trajectory}
Each trajectory point format: (x:float, y:float, heading:float)""" + """
Task 1: Critical Objects and Conditions Detection
Decide whether at least one critical instance of each class could influence the ego-vehicle's future path (no omissions). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego's future path. Output "yes" or "no" for every class (no omissions).
   Object classes to audit:
     - nearby_vehicle
     - conflicting_pedestrian
     - cyclist
     - construction
     - traffic_element
     - weather_condition
     - road_hazard
     - emergency_vehicle
     - animal
     - special_vehicle
     - conflicting_vehicle
     - door_opening_vehicle
"""
    Q2 = """Task 2: Natural Language Explanation
Compose a concise natural-language description of the optimal future 5-second trajectory for the ego vehicle that the expert driver (you) plans and explain why the expert driver plans to execute this trajectory.
   - Mention only the classes you marked "yes" in the previous task.
   - Describe how each of those critical objects or conditions influences the optimal trajectory.
   - Do not invent objects or conditions not present in the input.
"""
    Q3 = """Task 3: Meta-Behaviour Selection
Assign exactly one category from each list. Choose the label that best summarises the overall behaviour of the optimal future trajectory:
   - speed ∈ { keep, accelerate, decelerate }
   - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse }
   Choose the label that best summarises the overall behaviour of the optimal future trajectory.
   - If none fits, use 'other', but do this sparingly.
"""
    Q4 = """Task 4: Future Trajectory Prediction
(answer output should be wrapped in ...)
Given the input, critical objects/conditions, natural language explanation, and meta-behaviour, predict the optimal 4-second normalized future trajectory (8 steps at 2 Hz) of the ego vehicle. Predict 8 normalized future trajectory points in [PT, ...] format. Each point is (x, y, heading).
Output format (strict JSON, no extra keys, no markdown codeblock chars(```), no commentary):
{
  "critical_objects": {
    "nearby_vehicle": "yes | no",
    "conflicting_pedestrian": "yes | no",
    "cyclist": "yes | no",
    "construction": "yes | no",
    "traffic_element": "yes | no",
    "weather_condition": "yes | no",
    "road_hazard": "yes | no",
    "emergency_vehicle": "yes | no",
    "animal": "yes | no",
    "special_vehicle": "yes | no",
    "conflicting_vehicle": "yes | no",
    "door_opening_vehicle": "yes | no"
  },
  "explanation": "100-word description that references only the classes marked 'yes'",
  "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left | lane_change_right | reverse | other"
  },
  "future_trajectory": [PT, ...]
}
"""
    return f'{Q1}\n{Q2}\n{Q3}\n{Q4}'


def recog_prompt(ego_statuses) -> str:
    history_trajectory = []
    for i in range(4):
        ego_status = ego_statuses[i]
        history_trajectory.append({
            "x": format_number(ego_status.ego_pose[0]),
            "y": format_number(ego_status.ego_pose[1]),
            "heading": format_number(ego_status.ego_pose[2])
        })
    high_command_one_hot = ego_statuses[-1].driving_command
    navigation_commands = ['turn left', 'go straight', 'turn right']
    command_str = [navigation_commands[i] for i in range(len(high_command_one_hot)) if high_command_one_hot[i] == 1]
    command_str = command_str[0] if command_str else "unknown"
    image_prompt_desc = "1. Visual perception from front camera view\n"
    common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt_desc}2. Historical motion context (last 4 timesteps):{" ".join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""
    output_requirements = ("\nOutput requirements:\n- Predict 8 future trajectory points\n"
                           "- Each point format: (x:float, y:float, heading:float)\n"
                           "- Use [PT, ...] to encapsulate the trajectory\n"
                           "- Maintain numerical precision to 2 decimal places")
    return f"{''.join(['<image>' for _ in range(1)])}\n{common_prompt}{output_requirements}"


_LOADER = None
_SENSOR_ROOT = None


def _init(split: str, tokens: list):
    global _LOADER, _SENSOR_ROOT
    _SENSOR_ROOT = os.path.join(DATA, f"sensor_blobs/{split}")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    flt = OmegaConf.load(os.path.join(DEVKIT, "navsim/planning/script/config/common/train_test_split/"
                                      f"scene_filter/{'navtest' if split == 'test' else 'navtrain'}.yaml"))
    sf = instantiate(flt)
    sf.tokens = tokens
    sf.log_names = None
    _LOADER = SceneLoader(data_path=Path(os.path.join(DATA, f"navsim_logs/{split}")),
                          sensor_blobs_path=Path(os.path.join(DATA, f"sensor_blobs/{split}")),
                          scene_filter=sf, sensor_config=SensorConfig(cam_f0=[3], cam_l0=False, cam_l1=False,
                                                                      cam_l2=False, cam_r0=False, cam_r1=False,
                                                                      cam_r2=False, cam_b0=False, lidar_pc=False))


def _one(args):
    tok, img_dir = args
    try:
        scene = _LOADER.get_scene_from_token(tok)
        ai = scene.get_agent_input()
        es = ai.ego_statuses
        cam = ai.cameras[-1].cam_f0
        out_img = os.path.join(img_dir, f"{tok}.jpg")
        if not os.path.isfile(out_img):                                  # as NavsimCoTQwenAgent: PIL re-encode
            Image.fromarray(cam.image.astype('uint8'), 'RGB').save(out_img)
        orig = getattr(cam, "camera_path", None)
        if orig is not None and not os.path.isabs(str(orig)):
            orig = os.path.join(_SENSOR_ROOT, str(orig))
        return {"token": tok, "curious_system": CURIOUS_SYSTEM, "curious_user": curious_prompt(es),
                "recog_system": RECOG_SYSTEM, "recog_user": recog_prompt(es),
                "img_reencoded": out_img, "img_orig": str(orig) if orig is not None else None,
                "ego_poses": [list(map(float, s.ego_pose)) for s in es],
                "ego_velocity": list(map(float, es[-1].ego_velocity)), "ego_acceleration": list(map(float, es[-1].ego_acceleration)),
                "driving_command": list(map(int, es[-1].driving_command)), "log": scene.scene_metadata.log_name}
    except Exception as e:
        return {"token": tok, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("test", "trainval"), required=True)
    ap.add_argument("--tokens-from", required=True, help="directory of the scene JSONs of run_preprocessing.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    tokens = sorted(f[:-5] for f in os.listdir(args.tokens_from) if f.endswith(".json"))
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_err = 0
    with Pool(args.workers, initializer=_init, initargs=(args.split, tokens)) as pool, open(args.out, "w") as f:
        for i, r in enumerate(pool.imap(_one, [(t, args.img_dir) for t in tokens], chunksize=16)):
            n_err += "error" in r
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"[prompts] {i + 1}/{len(tokens)} ({n_err} errors)", flush=True)
    print(f"[prompts] done: {len(tokens)} tokens, {n_err} errors -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
