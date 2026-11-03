#!/usr/bin/env python3
"""
Save replay videos for all tasks in the stage 1 checkpoint.

    python eval_tasks_replay.py \
        --ckpt <path> --cfg <name> \
        --log_dir <dir>

Env vars:
    SEED_BASE            (default: 1000000)
    TEST_NUM             (default: 10)
    SEEDS_CACHE_DIR      (default: <robotwin>/cached_seeds/test_seeds)
"""

import argparse, csv, json, os, re, subprocess, sys
from pathlib import Path
import datetime

ROBOTWIN = Path(__file__).resolve().parent
SEEDS_CACHE = Path(os.environ.get("SEEDS_CACHE_DIR", str(ROBOTWIN / "cached_seeds" / "policy_generation_seeds")))

TEST_BASE = 1000000
TRAIN_BASE = 0
VALIDATION_BASE = 100000000
GENERALIZATION_BASE = 5000

# ── Edit these to change which tasks run ──────────────────────────────────────
TASKS_STAGE1 = [
    "place_dual_shoes", "beat_block_hammer", "pick_dual_bottles", "place_container_plate", "handover_mic", "lift_pot", "place_bread_basket", "place_empty_cup", "open_microwave", "put_bottles_dustbin" 
]

INSTRUCTIONS = json.load(open(ROBOTWIN / "task_prompts.json"))  # one prompt per task, shared by all data and eval scripts
# ─────────────────────────────────────────────────────────────────────────────

# ── Fixed summary column ordering ─────────────────────────────────────────────
# (display_name, task_key)  —  task_key=None means a computed average
SUMMARY_COLUMNS = [
    ("place dual shoes",           "place_dual_shoes"),
    ("beat block hammer",          "beat_block_hammer"),
    ("pick dual bottles",          "pick_dual_bottles"),
    ("place container plate",      "place_container_plate"),
    ("handover mic",               "handover_mic"),
    ("place empty cup",            "place_empty_cup"),
    ("place bread basket",         "place_bread_basket"),
    ("lift pot",                   "lift_pot"),
    ("open microwave",             "open_microwave"),
    ("put bottles dustbin",        "put_bottles_dustbin"),
    ("Average, Stage 1 tasks",     None),
    ("red on green",               "stack_blocks_two_red_on_green"),
    ("green on red",               "stack_blocks_two_green_on_red"),
    ("Average, Stage 2 new tasks", None),
    ("cylinder",                   "cylinder"),
    ("sphere",                     "sphere"),
    ("sphere distractor",          "sphere_distractor"),
    ("cylinder distractor",        "cylinder_distractor"),
    ("stack bowls 2",              "stack_bowls_two"),
    ("Average, OOD objects",       None),
    ("stack blocks three",         "stack_blocks_three"),
    ("stack bowls three",          "stack_bowls_three"),
]
AVERAGE_GROUPS = {
    "Average, Stage 1 tasks":     ["place_dual_shoes", "beat_block_hammer", "pick_dual_bottles", "place_container_plate", "handover_mic", "place_empty_cup", "place_bread_basket", "lift_pot", "open_microwave", "put_bottles_dustbin"],
    "Average, Stage 2 new tasks": ["stack_blocks_two_red_on_green", "stack_blocks_two_green_on_red"],
    "Average, OOD objects":       ["cylinder", "sphere", "sphere_distractor", "cylinder_distractor", "stack_bowls_two"],
}
# ─────────────────────────────────────────────────────────────────────────────

ANSI = re.compile(r"\x1b\[[0-9;]*m")

def parse_success(text):
    for line in reversed(text.splitlines()):
        m = re.search(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%", ANSI.sub("", line))
        if m:
            return m.group(1), m.group(2), m.group(3)
    return "", "", ""

def already_done(tsv_path, stage, ckpt, task, test_num):
    if not tsv_path.exists():
        return False
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if (row["stage"] == str(stage) and row["checkpoint"] == ckpt.replace(",", ";")
                    and row["task"] == task and row["total"] == str(test_num)
                    and row["successes"].isdigit()):
                return True
    return False

def run_stage(stage, cfg, ckpt, tasks, seed_base, test_num, log_dir, tsv_path, force, video_save_freq, hdf5_save_dir, exec_horizon=None, task_config="demo_clean_eval"):
    log = log_dir / f"stage{stage}.log"
    with open(log, "a") as f:
        f.write(f"\n=== stage{stage} cfg={cfg} ckpt={ckpt} ===\n")

    for i, task in enumerate(tasks):
        if not force and already_done(tsv_path, stage, ckpt, task, test_num):
            print(f"  SKIP {task} (done)")
            continue
        print(f"  [{i+1}/{len(tasks)}] {task}", end="", flush=True)

        cmd = [sys.executable, "-W", "ignore::UserWarning", "script/eval_policy_save.py",
               "--config", "policy/pi05/deploy_policy.yml", "--overrides",
               "--task_name", task, "--task_config", task_config,
               "--train_config_name", cfg, "--model_name", "pi05_base",
               "--ckpt_setting", "full_ft_frontview",
               "--seed", str(seed_base), "--test_num", str(test_num),
               "--policy_name", "pi05", "--checkpoint_id", ckpt,
               "--task_prompt", INSTRUCTIONS[task],
               "--seeds_cache_dir", str(SEEDS_CACHE),
               "--video_save_freq", str(video_save_freq), "--eval_dir", str(log_dir),
               "--hdf5_save_dir", str(hdf5_save_dir)]
        if exec_horizon is not None:
            cmd += ["--exec_horizon", str(exec_horizon)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(ROBOTWIN))
        out = []
        with open(log, "a") as lf:
            for line in proc.stdout:
                clean = ANSI.sub("", line).rstrip()
                out.append(line)
                if not line.startswith("step: "):
                    lf.write(line)
                # Print only lines that contain useful signal
                if re.search(r"(Success rate|Error|Traceback|Exception|WARNING)", clean, re.I):
                    print(f"    {clean}", flush=True)
        proc.wait()

        suc, total, pct = parse_success("".join(out))
        result = f"  {suc}/{total} = {pct}%" if suc else "  (no result)"
        rc_indicator = "" if proc.returncode == 0 else f"  [exit {proc.returncode}]"
        print(f"{result}{rc_indicator}")
        with open(tsv_path, "a", newline="") as f:
            csv.writer(f, delimiter="\t").writerow([stage, cfg, ckpt.replace(",", ";"), task, suc, total, pct])

def _get_pct(rows, stage, task_key):
    for r in rows:
        if r["stage"] == str(stage) and r["task"] == task_key and r["pct"]:
            try:
                return float(r["pct"])
            except ValueError:
                pass
    return None

def _cell(rows, stage, display, task_key):
    if task_key is not None:
        v = _get_pct(rows, stage, task_key)
        return f"{v:.1f}%" if v is not None else "--"
    vals = [_get_pct(rows, stage, t) for t in AVERAGE_GROUPS.get(display, [])]
    valid = [v for v in vals if v is not None]
    return f"{sum(valid)/len(valid):.1f}%" if valid else "--"

def print_table(tsv_path, tasks1):
    rows = []
    if tsv_path.exists():
        with open(tsv_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"\n{'task':<32} {'stage1':>10}")
    print("-" * 54)
    for display, task_key in SUMMARY_COLUMNS:
        p1 = _cell(rows, 1, display, task_key) if tasks1 else "—"
        print(f"{display:<32} {p1:>10}")

def write_summary(tsv_path, tasks1):
    rows = []
    if tsv_path.exists():
        with open(tsv_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    summary_path = tsv_path.parent / "summary.tsv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["stage"] + [d for d, _ in SUMMARY_COLUMNS])
        if tasks1:
            w.writerow(["stage1"] + [_cell(rows, 1, d, k) for d, k in SUMMARY_COLUMNS])
    print(f"Summary TSV: {summary_path}  <- paste into Google Sheets")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", help="Stage 1 checkpoint"); p.add_argument("--cfg", help="Stage 1 train config")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--test_num", type=int, default=int(os.environ.get("TEST_NUM", 50)))
    p.add_argument("--seed_base", type=int, default=int(os.environ.get("SEED_BASE", GENERALIZATION_BASE)))
    p.add_argument("--force", action="store_true", help="Rerun even if already in TSV")
    p.add_argument("--task_config", type=str, default="demo_clean_eval", help="task_config name to pass through to eval_policy_save.py")
    p.add_argument("--video_save_freq", type=int, default=10, help="Save video every N episodes. Set high (e.g. 999) to skip videos.")
    p.add_argument("--hdf5_save_dir", type=str, default="hdf5_saved_recent", help="Subdirectory name inside S1_policy_generated_data to save HDF5 files.")
    p.add_argument("--exec_horizon", type=int, default=None, help="Actions to execute per query (e.g. 25 or 50). Overrides deploy_policy.yml.")
    p.add_argument("--description", type=str, default=None, help="Description of the run")
    args = p.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    description = args.description if args.description else None
    current_command = " ".join(sys.argv)
    commands_csv = str(ROBOTWIN / "saved_runs.csv")
    if not os.path.exists(commands_csv):
        with open(commands_csv, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["command", "time", "description"])
    with open(commands_csv, "a", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([current_command, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description])
    if not args.ckpt:
        p.error("Provide at least --ckpt/--cfg")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    SEEDS_CACHE.mkdir(parents=True, exist_ok=True)
    tsv_path = log_dir / "results.tsv"
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        with open(tsv_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["stage", "train_config", "checkpoint", "task", "successes", "total", "pct"])

    os.chdir(ROBOTWIN)

    if args.ckpt and args.cfg:
        print(f"\n=== Stage 1: {args.cfg} ({len(TASKS_STAGE1)} tasks) ===")
        hdf5_save_dir = f"S1_policy_generated_data/{args.hdf5_save_dir}"
        run_stage(1, args.cfg, args.ckpt, TASKS_STAGE1, args.seed_base,
                  args.test_num, log_dir, tsv_path, args.force, args.video_save_freq, hdf5_save_dir, args.exec_horizon,
                  task_config=args.task_config)

    print(f"\nDone. TSV: {tsv_path}")
    print_table(tsv_path, TASKS_STAGE1 if args.ckpt else None)
    write_summary(tsv_path, TASKS_STAGE1 if args.ckpt else None)

if __name__ == "__main__":
    main()