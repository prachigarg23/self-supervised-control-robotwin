"""
The openpi imports are all from openpi_generalization/src/openpi, as that's the module installed 
"""
#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config # correct, from the common openpi_generalization/src/openpi/training/config.py
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
import pathlib


def _resolve_orbax_step_dir(checkpoint_id: str) -> pathlib.Path:
    """Orbax step dir is usually .../<exp>/<exp>/<step> or .../<exp>/<step>; pick the one that contains assets."""
    p = pathlib.Path(checkpoint_id).expanduser().resolve()
    candidates = [p]
    # Train often writes .../checkpoints/<name>/<name>/<step>; rsync may land as .../checkpoints/<name>/<step>
    if p.name.isdigit() and p.parent.name == p.parent.parent.name:
        candidates.append(p.parent.parent / p.name)
    for c in candidates:
        root = c / "assets"
        if root.is_dir() and any(root.rglob("norm_stats.json")):
            if c != p:
                print(f"Using checkpoint dir with assets: {c}")
            return c
    return p


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        resolved = _resolve_orbax_step_dir(checkpoint_id)
        self.checkpoint_id = str(resolved)

        # checkpoint_id is the absolute path to the orbax step directory (must contain assets/)
        assets_root = pathlib.Path(self.checkpoint_id) / "assets"
        norm_stats_files = list(assets_root.rglob("norm_stats.json"))
        if not norm_stats_files:
            raise FileNotFoundError(f"No norm_stats.json found under {assets_root}")
        assets_id = str(norm_stats_files[0].parent.relative_to(assets_root))

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            self.checkpoint_id,
            robotwin_repo_id=assets_id,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
