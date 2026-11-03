#!/usr/bin/env python3
"""
Render and save the first simulation frame (head + wrist RGB; optional head_camera intrinsics JSON).

Run from the RoboTwin repo root:
  python script/render_frame.py [--task-config demo_clean] [--seed 0] [--output first_frame.png]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import imageio
import numpy as np
import yaml

# Repo root = parent of script/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.chdir(_REPO_ROOT)

from envs import *  # noqa: F401,F403 — CONFIGS_PATH, env imports
import importlib


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def save_obs_camera_rgb(cam_data: dict, path: str, label: str) -> None:
    rgb = cam_data["rgb"]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(path, rgb)
    print(f"Saved {label} RGB: {path}  shape={rgb.shape}")


def get_embodiment_config(robot_file: str):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_args(task_config_name: str, seed: int, head_camera_type: str | None, wrist_camera_type: str | None):
    config_path = os.path.join(_REPO_ROOT, "task_config", f"{task_config_name}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = "stack_blocks_two"
    args["render_freq"] = 0
    args["collect_data"] = False
    args["save_path"] = os.path.join(_REPO_ROOT, ".cache_first_frame")
    os.makedirs(args["save_path"], exist_ok=True)

    embodiment_type = args["embodiment"]
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(et):
        robot_file = _embodiment_types[et]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment must have length 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if "data_type" not in args:
        args["data_type"] = {}
    args["data_type"]["rgb"] = True

    if head_camera_type is not None:
        args.setdefault("camera", {})["head_camera_type"] = head_camera_type

    if wrist_camera_type is not None:
        args.setdefault("camera", {})["wrist_camera_type"] = wrist_camera_type

    args["seed"] = seed
    args["now_ep_num"] = 0
    args["need_plan"] = True
    args["eval_mode"] = False

    return args


def main():
    parser = argparse.ArgumentParser(description="First head-camera frame for my_pad_pillbottle")
    parser.add_argument("--task-config", type=str, default="demo_clean", help="YAML in task_config/ (without .yml)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--head-camera-type",
        type=str,
        default=None,
        help="Override camera.head_camera_type (e.g. D435, Large_D435); default = from task YAML",
    )
    parser.add_argument(
        "--wrist-camera-type",
        type=str,
        default=None,
        help="Override camera.wrist_camera_type (e.g. D435, Large_D435); default = from task YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_REPO_ROOT, "first_frame_my_pad_pillbottle.png"),
        help="Head camera PNG path; left/right wrist images use <base>_left_camera.png and <base>_right_camera.png",
    )
    parser.add_argument(
        "--intrinsics-json",
        type=str,
        default=None,
        help="Optional path to save head_camera intrinsic_cv / extrinsic_cv as JSON (arrays as lists)",
    )
    args = parser.parse_args()

    task_args = build_args(args.task_config, args.seed, args.head_camera_type, args.wrist_camera_type)
    env = class_decorator("stack_blocks_two")

    try:
        env.setup_demo(**task_args)
        obs = env.get_obs()
        obs_cam = obs["observation"]
        base, ext = os.path.splitext(args.output)
        ext = ext or ".png"

        save_obs_camera_rgb(obs_cam["head_camera"], args.output, "head_camera")
        save_obs_camera_rgb(obs_cam["left_camera"], f"{base}_left_camera{ext}", "left_camera")
        save_obs_camera_rgb(obs_cam["right_camera"], f"{base}_right_camera{ext}", "right_camera")

        if args.intrinsics_json:
            head = obs_cam["head_camera"]
            payload = {
                "intrinsic_cv": np.asarray(head["intrinsic_cv"]).tolist(),
                "extrinsic_cv": np.asarray(head["extrinsic_cv"]).tolist(),
                "cam2world_gl": np.asarray(head["cam2world_gl"]).tolist(),
            }
            with open(args.intrinsics_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved camera matrices: {args.intrinsics_json}")
    finally:
        env.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
