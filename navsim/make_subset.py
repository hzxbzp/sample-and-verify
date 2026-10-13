#!/usr/bin/env python3
"""The navtrain scenes used to train the verifier: tokens of the navtrain split whose 4 history frames x 8 cameras
are all in the downloaded sensor data (the navtrain sensor archive comes in parts and a log can span parts), sampled
uniformly. Writes a token<TAB>log<TAB>train|val list (10% of the logs for verifier validation) and a scene filter for
the preprocessing and the metric cache.

    python navsim/make_subset.py --n 8000 --out-dir splits
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

DEVKIT = os.environ["NAVSIM_DEVKIT_ROOT"]
DATA = os.environ["OPENSCENE_DATA_ROOT"]
sys.path.insert(0, DEVKIT)
from hydra.utils import instantiate                                          # noqa: E402
from omegaconf import OmegaConf                                              # noqa: E402
from navsim.common.dataclasses import SensorConfig                           # noqa: E402
from navsim.common.dataloader import SceneLoader                             # noqa: E402

SENS = os.path.join(DATA, "sensor_blobs/trainval")
FILTERS = os.path.join(DEVKIT, "navsim/planning/script/config/common/train_test_split/scene_filter")


def complete(frames) -> bool:
    return all(os.path.isfile(os.path.join(SENS, cam["data_path"])) for f in frames[:4] for cam in f["cams"].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = OmegaConf.load(os.path.join(FILTERS, "navtrain.yaml"))
    have = set(os.listdir(SENS))
    cfg.log_names = [lg for lg in cfg.log_names if lg in have]
    loader = SceneLoader(data_path=Path(os.path.join(DATA, "navsim_logs/trainval")), sensor_blobs_path=None,
                         scene_filter=instantiate(cfg), sensor_config=SensorConfig.build_no_sensors())
    frames = loader.scene_frames_dicts
    pairs = [(t, lg) for lg, ts in loader.get_tokens_list_per_log().items() for t in ts]
    rs = np.random.default_rng(args.seed)
    pick = []
    for i in rs.permutation(len(pairs)):
        if len(pick) >= args.n:
            break
        if complete(frames[pairs[i][0]]):
            pick.append(pairs[i])
    pick.sort()
    ulogs = sorted({lg for _, lg in pick})
    val_logs = set(rs.choice(ulogs, max(1, len(ulogs) // 10), replace=False))
    with open(os.path.join(args.out_dir, "navtrain_sub.tsv"), "w") as f:
        for t, lg in pick:
            f.write(f"{t}\t{lg}\t{'val' if lg in val_logs else 'train'}\n")
    flt = OmegaConf.to_container(cfg, resolve=True)
    flt["log_names"], flt["tokens"] = ulogs, [t for t, _ in pick]
    yaml.safe_dump(flt, open(os.path.join(args.out_dir, "navtrain_sub.yaml"), "w"))
    print(f"picked {len(pick)} tokens from {len(ulogs)} logs ({len(val_logs)} validation logs)")


if __name__ == "__main__":
    main()
