#!/usr/bin/env python3
"""
Eval tasks against one or two checkpoints (stage1 / stage2).

Two stages (continual-learning comparison):
    python eval_tasks.py \
        --ckpt1 <path> --cfg1 <name> \
        --ckpt2 <path> --cfg2 <name> \
        --log_dir <dir>

Single stage:
    python eval_tasks.py --ckpt2 <path> --cfg2 <name> --log_dir <dir>

Env vars:
    SEED_BASE            (default: 999, multiplied inside eval_policy as 100000*(1+seed))
    TEST_NUM             (default: 50)
    SEEDS_CACHE_DIR      (default: <robotwin>/cached_seeds/test_seeds)

Resume: skips tasks already completed in results.tsv. Use --force to rerun.
Edit TASKS_STAGE1 / TASKS_STAGE2 at the top of this file to change which tasks run.
"""

import argparse, csv, json, os, re, subprocess, sys
from pathlib import Path
import datetime

ROBOTWIN = Path(__file__).resolve().parent
SEEDS_CACHE = Path(os.environ.get("SEEDS_CACHE_DIR", str(ROBOTWIN / "cached_seeds" / "test_seeds")))

TEST_BASE = 1000000
TRAIN_BASE = 0
VALIDATION_BASE = 100000000

# ── Edit these to change which tasks run ──────────────────────────────────────
TASKS_STAGE1 = ["stack_blocks_two_green_on_red", "stack_blocks_two_red_on_green", "stack_bowls_two", "stack_blocks_three", "stack_bowls_three"]
# TASKS_STAGE1 = [
#     "place_dual_shoes", "put_bottles_dustbin", "beat_block_hammer", "pick_dual_bottles", "place_container_plate", "handover_mic", "lift_pot", "place_bread_basket", "place_empty_cup", "open_microwave" 
# ]
# TASKS_STAGE2 = ["stack_blocks_two_green_on_red", "stack_blocks_two_red_on_green", "beat_block_hammer", "place_container_plate",]
TASKS_STAGE2 = [
    "stack_blocks_two_green_on_red", "stack_blocks_two_red_on_green", "place_dual_shoes", "put_bottles_dustbin", "beat_block_hammer", "handover_mic", "lift_pot", "open_microwave", "pick_dual_bottles", "place_bread_basket", "place_empty_cup", "place_container_plate", "stack_bowls_two", "stack_blocks_three", "stack_bowls_three",
]

TASKS_STAGE2_OOD_COMPOSITIONALITY = ["stack_bowls_two", "stack_blocks_three", "stack_bowls_three"]

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

def read_result_txt(run_dir, task, test_num, task_config="demo_clean_eval"):
    result_dir = Path(run_dir) / task / "pi05" / task_config / "full_ft_frontview"
    if not result_dir.exists():
        return "", "", ""
    for ts_dir in reversed(sorted(result_dir.iterdir())):
        rf = ts_dir / "_result.txt"
        if not rf.exists():
            continue
        for line in reversed(rf.read_text().splitlines()):
            try:
                rate = float(line.strip())
                return str(round(rate * test_num)), str(test_num), f"{rate * 100:.1f}"
            except ValueError:
                continue
    return "", "", ""

def already_done(run_dir, task, test_num, task_config="demo_clean_eval"):
    suc, total, _ = read_result_txt(run_dir, task, test_num, task_config)
    return total == str(test_num) and suc.isdigit()

def run_stage(stage, cfg, ckpt, tasks, seed_base, test_num, log_dir, tsv_path, force, video_save_freq, exec_horizon=None, task_config="demo_clean_eval"):
    log = log_dir / f"stage{stage}.log"
    with open(log, "a") as f:
        f.write(f"\n=== stage{stage} cfg={cfg} ckpt={ckpt} ===\n")

    for i, task in enumerate(tasks):
        if not force and already_done(log_dir, task, test_num, task_config):
            print(f"  SKIP {task} (done)")
            continue
        print(f"  [{i+1}/{len(tasks)}] {task}", flush=True)

        cmd = [sys.executable, "-W", "ignore::UserWarning", "script/eval_policy.py",
               "--config", "policy/pi05/deploy_policy.yml", "--overrides",
               "--task_name", task, "--task_config", task_config,
               "--train_config_name", cfg, "--model_name", "pi05_base",
               "--ckpt_setting", "full_ft_frontview",
               "--seed", str(seed_base), "--test_num", str(test_num),
               "--policy_name", "pi05", "--checkpoint_id", ckpt,
               "--task_prompt", INSTRUCTIONS[task],
               "--seeds_cache_dir", str(SEEDS_CACHE),
               "--video_save_freq", str(video_save_freq), "--eval_dir", str(log_dir)]
        if exec_horizon is not None:
            cmd += ["exec_horizon", str(exec_horizon)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(ROBOTWIN), env={**os.environ, "PYTHONUNBUFFERED": "1"})
        with open(log, "a") as lf:
            for line in proc.stdout:
                if not line.startswith("step: "):
                    lf.write(line)
        proc.wait()

        suc, total, pct = read_result_txt(log_dir, task, test_num, task_config)
        result = f"  {suc}/{total} = {pct}%" if suc else "  (no result)"
        print(f"{result}" + (f"  [exit {proc.returncode}]" if proc.returncode else ""))
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

def print_table(tsv_path, tasks1, tasks2):
    rows = []
    if tsv_path.exists():
        with open(tsv_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"\n{'task':<32} {'stage1':>10} {'stage2':>10}")
    print("-" * 54)
    for display, task_key in SUMMARY_COLUMNS:
        p1 = _cell(rows, 1, display, task_key) if tasks1 else "—"
        p2 = _cell(rows, 2, display, task_key) if tasks2 else "—"
        print(f"{display:<32} {p1:>10} {p2:>10}")

def write_summary(tsv_path, tasks1, tasks2):
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
        if tasks2:
            w.writerow(["stage2"] + [_cell(rows, 2, d, k) for d, k in SUMMARY_COLUMNS])
    print(f"Summary TSV: {summary_path}  <- paste into Google Sheets")

def main():
    

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt1", help="Stage 1 checkpoint"); p.add_argument("--cfg1", help="Stage 1 train config")
    p.add_argument("--ckpt2", help="Stage 2 checkpoint"); p.add_argument("--cfg2", help="Stage 2 train config")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--test_num", type=int, default=int(os.environ.get("TEST_NUM", 50)))
    p.add_argument("--seed_base", type=int, default=int(os.environ.get("SEED_BASE", TEST_BASE)))
    p.add_argument("--force", action="store_true", help="Rerun even if already in TSV")
    p.add_argument("--video_save_freq", type=int, default=10, help="Save video every N episodes. Set high (e.g. 999) to skip videos.")
    p.add_argument("--exec_horizon", type=int, default=None, help="Actions to execute per query (e.g. 25 or 50). Overrides deploy_policy.yml.")
    p.add_argument("--description", type=str, default=None, help="Description of the run")
    p.add_argument("--tasks2", nargs="+", default=None, help="Override TASKS_STAGE2 with specific tasks")
    p.add_argument("--run_id", type=int, default=None, help="Run index for multi-run variance (1, 2, 3...). Saves to log_dir/run_{id}/. Omit for single run (legacy).")
    p.add_argument("--task_config", type=str, default="demo_clean_eval", help="task_config passed to eval_policy.py (e.g. demo_clean_eval_topview for top-view embodiment). Defaults to demo_clean_eval (front view).")
    args = p.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    current_command = " ".join(sys.argv)
    description = args.description if args.description else None
    commands_csv = str(ROBOTWIN / "saved_runs.csv")
    if not os.path.exists(commands_csv):
        with open(commands_csv, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["command", "time", "description"])
    with open(commands_csv, "a", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([current_command, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description])
    if not args.ckpt1 and not args.ckpt2:
        p.error("Provide at least --ckpt1/--cfg1 or --ckpt2/--cfg2")

    log_dir = Path(args.log_dir)
    run_dir = log_dir / f"run_{args.run_id}" if args.run_id is not None else log_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    SEEDS_CACHE.mkdir(parents=True, exist_ok=True)
    tsv_path = run_dir / "results.tsv"
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        with open(tsv_path, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["stage", "train_config", "checkpoint", "task", "successes", "total", "pct"])

    os.chdir(ROBOTWIN)

    if args.ckpt1 and args.cfg1:
        print(f"\n=== Stage 1: {args.cfg1} ({len(TASKS_STAGE1)} tasks) ===")
        run_stage(1, args.cfg1, args.ckpt1, TASKS_STAGE1, args.seed_base,
                  args.test_num, run_dir, tsv_path, args.force, args.video_save_freq, args.exec_horizon, args.task_config)

    if args.ckpt2 and args.cfg2:
        tasks2 = args.tasks2 if args.tasks2 else TASKS_STAGE2
        print(f"\n=== Stage 2: {args.cfg2} ({len(tasks2)} tasks) ===")
        run_stage(2, args.cfg2, args.ckpt2, tasks2, args.seed_base,
                  args.test_num, run_dir, tsv_path, args.force, args.video_save_freq, args.exec_horizon, args.task_config)

    print(f"\nDone. TSV: {tsv_path}")
    print_table(tsv_path, TASKS_STAGE1 if args.ckpt1 else None, TASKS_STAGE2 if args.ckpt2 else None)
    write_summary(tsv_path, TASKS_STAGE1 if args.ckpt1 else None, TASKS_STAGE2 if args.ckpt2 else None)

if __name__ == "__main__":
    main()