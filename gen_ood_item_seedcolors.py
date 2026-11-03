#!/usr/bin/env python3
"""Pre-generate per-seed OOD color configs for the 4 OOD stack_blocks_two_* tasks.

For each (task, seed) it writes envs/ood_colors/<task>/seed_<N>.json with
{block1_color, sphere_color or cylinder_color, [block2_color], task}.

Used ONLY by run_ood_test.py + OOD env files. Does not affect any other task.

By default seeds are 0..49 (matches run_ood_test.py test_num=50). Override with
--num_seeds, or pass --seeds 1000000 1000001 ... to generate for explicit seeds.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLOR_MAP_PATH = ROOT / "envs" / "color_map.json"
OUT_DIR = ROOT / "envs" / "ood_colors"

ORDER = ["block1", "sphere", "cylinder"]
TASKS = [
    "stack_blocks_two_cylinder",
    "stack_blocks_two_sphere",
    "stack_blocks_two_sphere_distractor",
    "stack_blocks_two_cylinder_distractor",
]

# 14 broadly-recognizable colors (brown excluded — table is brown). C(14,2) = 91 unique pairs.
ALLOWED_COLORS = [
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan",
    "white", "black", "gray", "magenta", "peach", "indigo",
]

with open(COLOR_MAP_PATH) as f:
    _FULL_COLOR_MAP = json.load(f)
COLOR_MAP = {c: _FULL_COLOR_MAP[c] for c in ALLOWED_COLORS if c in _FULL_COLOR_MAP}
assert len(COLOR_MAP) >= 10, f"Too few allowed colors found in color_map.json: {list(COLOR_MAP)}"


def _display_name(name):
    return "block" if name == "block1" else name


def build_for_seed(task_name, seed):
    """Return info dict for (task, seed). Color choice deterministic via random.seed(seed)."""
    rng = random.Random(seed)
    orig, other = rng.sample(list(COLOR_MAP.keys()), 2)

    if task_name == "stack_blocks_two_cylinder_distractor":
        args = {"block1": orig, "cylinder": other, "block2": other}
    elif task_name == "stack_blocks_two_cylinder":
        args = {"block1": orig, "cylinder": other}
    elif task_name == "stack_blocks_two_sphere_distractor":
        args = {"block1": orig, "sphere": other, "block2": other}
    elif task_name == "stack_blocks_two_sphere":
        args = {"block1": orig, "sphere": other}
    else:
        raise ValueError(f"Unknown OOD task: {task_name}")

    objs = sorted(
        [(k, v) for k, v in args.items() if v is not None and k != "block2"],
        key=lambda x: ORDER.index(x[0]),
    )
    (n1, c1), (n2, c2) = objs
    task = (f"Pick up the {c1} {_display_name(n1)} and place it in the center of the table. "
            f"Pick up the {c2} {_display_name(n2)} and stack it on top of the {c1} {_display_name(n1)}.")

    info = {f"{name}_color": COLOR_MAP[color] for name, color in args.items() if color is not None}
    info["task"] = task
    return info


def info_path_for(task_name, seed):
    return OUT_DIR / task_name / f"seed_{seed}.json"


def pregenerate(seeds):
    for task in TASKS:
        d = OUT_DIR / task
        d.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            with (d / f"seed_{s}.json").open("w") as f:
                json.dump(build_for_seed(task, s), f, indent=2)
    print(f"Wrote {len(TASKS)} × {len(seeds)} = {len(TASKS) * len(seeds)} per-seed color files under {OUT_DIR}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_seeds", type=int, default=50,
                   help="Number of seeds starting from --start_seed (default: 50).")
    p.add_argument("--start_seed", type=int, default=1000000,
                   help="First seed value (matches run_ood_test.py seed_base default).")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Explicit list of seeds. Overrides --num_seeds/--start_seed.")
    args = p.parse_args()
    seeds = args.seeds if args.seeds is not None else list(range(args.start_seed, args.start_seed + args.num_seeds))
    pregenerate(seeds)


if __name__ == "__main__":
    main()
