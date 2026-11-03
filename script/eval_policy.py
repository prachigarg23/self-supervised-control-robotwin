import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import time
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _robotwin_init_dbg(msg: str) -> None:
    """CuRobo / env init milestones. Enable with ROBOTWIN_DEBUG_INIT=1."""
    if os.environ.get("ROBOTWIN_DEBUG_INIT", "").lower() not in ("1", "true", "yes"):
        return
    print(f"[robotwin-init {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    eval_dir = usr_args.get("eval_dir") or "eval_result"
    save_dir = Path(f"{eval_dir}/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    args["task_prompt"] = usr_args.get("task_prompt")
    args["seeds_cache_dir"] = usr_args.get("seeds_cache_dir")
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = seed
    suc_nums = []
    test_num = usr_args.get("test_num", 20)
    if test_num is None:
        test_num = 20
    topk = 1

    _robotwin_init_dbg("eval_policy: loading policy model ...")
    model = get_model(usr_args)
    _robotwin_init_dbg("eval_policy: entering eval_policy (seed validation / rollout) ...")
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   video_save_freq=usr_args.get("video_save_freq", 10),
                                   exec_horizon=usr_args.get("exec_horizon"))
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                video_save_freq=10,
                exec_horizon=None):
    import json

    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    # Build (or load) the fixed list of test seeds via expert check.
    seeds_cache_dir = args.get("seeds_cache_dir")
    cache_file = os.path.join(seeds_cache_dir, f"{task_name}_seed{st_seed}_n{test_num}.json") if seeds_cache_dir else None
    ood_seed_dir = os.path.join("envs", "ood_colors", task_name)
    test_seeds = None

    # For OOD tasks, source seeds from per-seed json files directly.
    if os.path.isdir(ood_seed_dir):
        ood_seeds = []
        for fname in os.listdir(ood_seed_dir):
            if not (fname.startswith("seed_") and fname.endswith(".json")):
                continue
            try:
                ood_seeds.append(int(fname[len("seed_"):-len(".json")]))
            except ValueError:
                continue
        ood_seeds = sorted(set(ood_seeds))
        ood_seeds = [s for s in ood_seeds if s >= st_seed]
        if test_num is not None:
            ood_seeds = ood_seeds[:test_num]
        if ood_seeds:
            test_seeds = ood_seeds
            print(f"Loaded {len(test_seeds)} OOD seeds from {ood_seed_dir}")
            print(f"[seed-source] ood_seed_dir={ood_seed_dir}")
            for s in test_seeds:
                print(f"[seed-file] {os.path.join(ood_seed_dir, f'seed_{s}.json')}")

    if test_seeds is None and cache_file and os.path.exists(cache_file):
        with open(cache_file) as f:
            test_seeds = json.load(f)
        print(f"Loaded {len(test_seeds)} cached seeds from {cache_file}")
        print(f"[seed-source] cache_file={cache_file}")
    elif test_seeds is None:
        test_seeds = []
        now_seed = st_seed
        consecutive_errors = 0
        while len(test_seeds) < test_num:
            try:
                TASK_ENV.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **args)
                TASK_ENV.play_once()
                consecutive_errors = 0
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                import traceback; traceback.print_exc()
                print(f"error occurs ! ({type(e).__name__}: {e})", flush=True)
                consecutive_errors += 1
                if consecutive_errors >= 20:
                    raise RuntimeError(f"seed search failed {consecutive_errors} times in a row") from e
                continue
            if TASK_ENV.plan_success and TASK_ENV.check_success():
                test_seeds.append(now_seed)
            TASK_ENV.close_env()
            print (f"Successfully evaluated seed {now_seed}")
            print (f"Total seeds evaluated: {len(test_seeds)}")
            now_seed += 1
        if cache_file:
            os.makedirs(seeds_cache_dir, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(test_seeds, f)
            print(f"Saved {len(test_seeds)} validated seeds to {cache_file}")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    fixed_task_prompt = args.get("task_prompt")

    for now_id, now_seed in enumerate(test_seeds):
        # assumes you either get task prompt from args.task_prompt or read it from a seed-wise json file (ood), does not handle case where both passed
        episode_info_list = [{}]
        resolved_task_prompt = fixed_task_prompt
        # OOD per-seed override: if envs/ood_colors/<task>/seed_<now_seed>.json exists,
        # use its `task` field as the instruction (overrides --task_prompt for OOD only).
        per_seed_path = f"envs/ood_colors/{args['task_name']}/seed_{now_seed}.json"
        per_seed_prompt = None
        if os.path.exists(per_seed_path):
            try:
                with open(per_seed_path) as _f:
                    per_seed_prompt = json.load(_f).get("task")
            except Exception:
                per_seed_prompt = None
        if per_seed_prompt is not None:
            resolved_task_prompt = per_seed_prompt
        args["task_prompt"] = resolved_task_prompt

        if resolved_task_prompt:
            instruction = resolved_task_prompt
        else:
            print(f'RANDOM INSTRUCTION')
            print(episode_info_list, args["task_name"], test_num)
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            print(results)
            print(results[0][instruction_type])
            instruction = np.random.choice(results[0][instruction_type])
        # gets latest task prompt, if its dynamically read for ood-tasks
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None and now_id % video_save_freq == 0:
            try:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_bin = "ffmpeg"
            ffmpeg = subprocess.Popen(
                [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)
        else:
            TASK_ENV._set_eval_video_ffmpeg(None)

        succ = False
        reset_func(model)
        ep_n = len(test_seeds)
        first_rollout_step = True
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            if first_rollout_step:
                t_obs = time.monotonic()
                print(
                    f"[eval_policy] ep {now_id + 1}/{ep_n} seed={now_seed}: get_obs ... "
                    f"(cold start: first physics/render buffers here)",
                    flush=True,
                )
            observation = TASK_ENV.get_obs()
            if first_rollout_step:
                print(
                    f"[eval_policy] get_obs OK in {time.monotonic() - t_obs:.2f}s; policy.forward + take_action chunk ... "
                    "(first JAX/CUDA compile can sit here many minutes)",
                    flush=True,
                )
                t_act = time.monotonic()
            eval_func(TASK_ENV, model, observation, now_id, video_save_freq, exec_horizon=exec_horizon)
            if first_rollout_step:
                print(
                    f"[eval_policy] first chunk done in {time.monotonic() - t_act:.2f}s "
                    f"(take_action_cnt={TASK_ENV.take_action_cnt})",
                    flush=True,
                )
                first_rollout_step = False
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        TASK_ENV.close_env(clear_cache=((now_id + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
