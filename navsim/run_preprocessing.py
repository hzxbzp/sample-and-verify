#!/usr/bin/env python3
"""Scene JSONs (ego state, history, command, logged future) of a NAVSIM split, written by the public AutoVLA release's
``tools/preprocessing/nocot_sample_generation.py``, run unchanged except that
  (i)  its unused Waymo import (which pulls in TensorFlow) is replaced by an empty stub, and
  (ii) with HISTORY_SENSORS_ONLY=1 the scene loader reads the camera images of the 4 history frames only; the JSON
       records only these frames' camera paths, so its content is unchanged. This is needed for the navtrain sensor
       package, which contains no future frames.
Arguments are passed through; run from the AutoVLA root (the script reads ./config/<name>.yaml).

    cd $AUTOVLA_ROOT && python /path/to/navsim/run_preprocessing.py --config dataset/<cfg> --output_dir json/test
"""
import os
import runpy
import sys
import types

stub = types.ModuleType("dataset_utils.preprocessing.waymo_e2e_dataset")
stub.WaymoE2ECoTAnnotationDataset = None
stub.DataCollator = None
sys.modules["dataset_utils.preprocessing.waymo_e2e_dataset"] = stub
if os.environ.get("HISTORY_SENSORS_ONLY") == "1":
    from navsim.agents.vla_agent import VlaAgent
    from navsim.common.dataclasses import SensorConfig
    VlaAgent.get_sensor_config = lambda self: SensorConfig.build_all_sensors(include=[0, 1, 2, 3])
script = os.path.join(os.environ["AUTOVLA_ROOT"], "tools/preprocessing/nocot_sample_generation.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
