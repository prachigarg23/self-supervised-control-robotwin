import json
import argparse
from pathlib import Path
import random

ORDER = ["block1", "sphere", "cylinder"]

with open("./envs/color_map.json", "r") as f:
    COLOR_MAP = json.load(f)

def display_name(name):
    return "block" if name == "block1" else name


def get_task_name(task_name):
    random.seed(0)
    orig, other = random.sample(list(COLOR_MAP.keys()), 2)

    if task_name == "stack_blocks_two_cylinder_distractor":
        return generate_task(block1=orig, cylinder=other, block2=other)   # block2 = distractor
    elif task_name == "stack_blocks_two_cylinder":
        return generate_task(block1=orig, cylinder=other)
    elif task_name == "stack_blocks_two_sphere_distractor":
        return generate_task(block1=orig, sphere=other, block2=other)     # block2 = distractor
    elif task_name == "stack_blocks_two_sphere":
        return generate_task(block1=orig, sphere=other)
    else:
        raise ValueError(f"Unknown task name: {task_name}")


def generate_task(block1=None, block2=None, sphere=None, cylinder=None,
                  info_path="envs/info.json"):
    

    args = {"block1": block1, "block2": block2, "sphere": sphere, "cylinder": cylinder}

    # Validate all provided colors exist in the map
    for name, color in args.items():
        if color is not None and color not in COLOR_MAP:
            raise ValueError(f"Unknown color '{color}' for {name}. Choose from: {list(COLOR_MAP.keys())}")

    task_objects = sorted(
        [(k, v) for k, v in args.items() if v is not None and k != "block2"],
        key=lambda x: ORDER.index(x[0])
    )

    (name1, color1), (name2, color2) = task_objects

    task = (f"Pick up the {color1} {display_name(name1)} and place it in the center of the table. "
            f"Pick up the {color2} {display_name(name2)} and stack it on top of the {color1} {display_name(name1)}.")

    info_path = Path(info_path)
    if info_path.exists():
        try:
            with info_path.open("r") as f:
                info = json.load(f)
            if not isinstance(info, dict):
                info = {}
        except (json.JSONDecodeError, OSError):
            info = {}
    else:
        info = {}

    info.update({f"{name}_color": COLOR_MAP[color] for name, color in args.items() if color is not None})
    info["task"] = task

    with info_path.open("w") as f:
        json.dump(info, f, indent=2)

    print(task)
    return task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block1", metavar="COLOR")
    parser.add_argument("--block2", metavar="COLOR")
    parser.add_argument("--sphere", metavar="COLOR")
    parser.add_argument("--cylinder", metavar="COLOR")
    args = parser.parse_args()
    generate_task(**vars(args))