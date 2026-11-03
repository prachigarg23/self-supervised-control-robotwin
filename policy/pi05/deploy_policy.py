import numpy as np
import torch
import dill
import os, sys
import cv2

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *
import time

# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation, now_id, video_save_freq, exec_horizon=None, save_hdf5_file = False, current_observations = []):
    # now_id is episode index; eval() runs many times per episode — print once per milestone episode.
    if now_id % 10 == 0 and model.observation_window is None:
        print(f"now_id={now_id}", flush=True)
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    # ======== Get Action ========

    n_exec = exec_horizon if exec_horizon is not None else model.pi0_step
    if not getattr(model, "_exec_horizon_printed", False):
        print(f"exec_horizon={n_exec} (chunk size={model.pi0_step})")
        model._exec_horizon_printed = True
    actions = model.get_action()[:n_exec]

    for action in actions:
        TASK_ENV.take_action(action)
        # remove the next three lines 
        if save_hdf5_file:
            observation = TASK_ENV.get_obs()
            current_observations['joint_action'].append(observation['joint_action'])
            current_observations['endpose'].append(observation['endpose'])
            current_observations['pointcloud'].append(observation['pointcloud'])
            current_observations['observation'].append(observation['observation'])
        if now_id % video_save_freq == 0:
            observation = TASK_ENV.get_obs()
            input_rgb_arr, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)
    if save_hdf5_file:
        return current_observations


def reset_model(model):
    model.reset_obsrvationwindows()
