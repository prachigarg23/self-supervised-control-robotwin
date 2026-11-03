#!/usr/bin/env python3
"""
Eval tasks using fixed validation seeds.

Validation seeds live in cached_seeds/val_seeds_cache/ with a per-task seed.
The 6 existing cached tasks reuse their current seeds; new tasks get
sequential seeds starting at 100700000.

Two stages:
    python validation_tasks.py \
        --ckpt1 <path> --cfg1 <name> \
        --ckpt2 <path> --cfg2 <name> \
        --log_dir <dir> \
        --exec_horizon 25 (optional)

Single stage:
    python validation_tasks.py --ckpt2 <path> --cfg2 <name> --log_dir <dir>
"""

import argparse, csv, json, os, re, subprocess, sys
from pathlib import Path

ROBOTWIN = Path(__file__).resolve().parent
SEEDS_CACHE = Path(os.environ.get("SEEDS_CACHE_DIR", str(ROBOTWIN / "cached_seeds" / "val_seeds_cache")))

# Per-task seeds. Existing 6 match the cached files already in cached_seeds/val_seeds_cache/.
# New tasks assigned sequentially from 100700000.
TASK_SEEDS = {
    "place_dual_shoes":              100100000,
    "beat_block_hammer":             100200000,
    "pick_dual_bottles":             100300000,
    "place_container_plate":         100400000,
    "stack_blocks_two_green_on_red": 100500000,
    "stack_blocks_two_red_on_green": 100600000,
    # from cached_seeds/val_seeds_cache
    "put_bottles_dustbin":           100000000,
    "handover_mic":                  100000000,
    "lift_pot":                      100000000,
    "place_bread_basket":            100000000,
    "place_empty_cup":               100000000,
    "open_microwave":                100000000,
}

# ── Edit these to change which tasks run ──────────────────────────────────────
TASKS_STAGE1 = [
    "place_dual_shoes", "put_bottles_dustbin", "beat_block_hammer", "pick_dual_bottles", "place_container_plate", "handover_mic", "lift_pot", "place_bread_basket", "place_empty_cup", "open_microwave"
]
TASKS_STAGE2 = [
    "stack_blocks_two_green_on_red", "stack_blocks_two_red_on_green", "place_dual_shoes", "lift_pot", "put_bottles_dustbin", "beat_block_hammer", "handover_mic", "open_microwave", "pick_dual_bottles", "place_bread_basket", "place_empty_cup", "place_container_plate",
]

INSTRUCTIONS = json.load(open(ROBOTWIN / "task_prompts.json"))  # one prompt per task, shared by all data and eval scripts
# ─────────────────────────────────────────────────────────────────────────────

ANSI = re.compile(r"\x1b\[[0-9;]*m")

def parse_success(text):
    for line in reversed(text.splitlines()):
        m = re.search(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%", ANSI.sub("", line))
        if m:
            return m.group(1), m.group(2), m.group(3)
    return "", "", ""

def already_done(csv_path, stage, ckpt, task, test_num):
    if not csv_path.exists():
        return False
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if (row["stage"] == str(stage) and row["checkpoint"] == ckpt.replace(",", ";")
                    and row["task"] == task and row["total"] == str(test_num)
                    and row["successes"].isdigit()):
                return True
    return False

def run_stage(stage, cfg, ckpt, tasks, test_num, log_dir, csv_path, force, video_save_freq, exec_horizon=None):
    log = log_dir / f"stage{stage}.log"
    with open(log, "a") as f:
        f.write(f"\n=== stage{stage} cfg={cfg} ckpt={ckpt} ===\n")

    for i, task in enumerate(tasks):
        if not force and already_done(csv_path, stage, ckpt, task, test_num):
            print(f"  SKIP {task} (done)")
            continue
        print(f"  [{i+1}/{len(tasks)}] {task}", end="", flush=True)

        seed = TASK_SEEDS[task]
        cmd = [sys.executable, "-W", "ignore::UserWarning", "script/eval_policy.py",
               "--config", "policy/pi05/deploy_policy.yml", "--overrides",
               "--task_name", task, "--task_config", "demo_clean_eval",
               "--train_config_name", cfg, "--model_name", "pi05_base",
               "--ckpt_setting", "full_ft_frontview",
               "--seed", str(seed), "--test_num", str(test_num),
               "--policy_name", "pi05", "--checkpoint_id", ckpt,
               "--task_prompt", INSTRUCTIONS[task],
               "--seeds_cache_dir", str(SEEDS_CACHE),
               "--video_save_freq", str(video_save_freq), "--eval_dir", str(log_dir)]
        if exec_horizon is not None:
            cmd += ["exec_horizon", str(exec_horizon)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(ROBOTWIN))
        out = []
        with open(log, "a") as lf:
            for line in proc.stdout:
                clean = ANSI.sub("", line).rstrip()
                out.append(line)
                if not line.startswith("step: "):
                    lf.write(line)
                if re.search(r"(Success rate|Error|Traceback|Exception|WARNING)", clean, re.I):
                    print(f"    {clean}", flush=True)
        proc.wait()

        suc, total, pct = parse_success("".join(out))
        result = f"  {suc}/{total} = {pct}%" if suc else "  (no result)"
        rc_indicator = "" if proc.returncode == 0 else f"  [exit {proc.returncode}]"
        print(f"{result}{rc_indicator}")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([stage, cfg, ckpt.replace(",", ";"), task, suc, total, pct])

def print_table(csv_path, tasks1, tasks2):
    rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    def pct(stage, task):
        for r in rows:
            if r["stage"] == str(stage) and r["task"] == task and r["pct"]:
                return r["pct"] + "%"
        return "n/a"
    all_tasks = list(dict.fromkeys((tasks1 or []) + (tasks2 or [])))
    print(f"\n{'task':<32} {'stage1':>10} {'stage2':>10}")
    print("-" * 54)
    for t in all_tasks:
        p1 = pct(1, t) if tasks1 and t in tasks1 else "—"
        p2 = pct(2, t) if tasks2 and t in tasks2 else "—"
        print(f"{t:<32} {p1:>10} {p2:>10}")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt1", help="Stage 1 checkpoint"); p.add_argument("--cfg1", help="Stage 1 train config")
    p.add_argument("--ckpt2", help="Stage 2 checkpoint"); p.add_argument("--cfg2", help="Stage 2 train config")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--test_num", type=int, default=int(os.environ.get("TEST_NUM", 50)))
    p.add_argument("--force", action="store_true", help="Rerun even if already in CSV")
    p.add_argument("--video_save_freq", type=int, default=10, help="Save video every N episodes. Set high (e.g. 999) to skip videos.")
    p.add_argument("--exec_horizon", type=int, default=None, help="Actions to execute per query (e.g. 25 or 50). Overrides deploy_policy.yml.")
    args = p.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if not args.ckpt1 and not args.ckpt2:
        p.error("Provide at least --ckpt1/--cfg1 or --ckpt2/--cfg2")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    SEEDS_CACHE.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "results.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["stage", "train_config", "checkpoint", "task", "successes", "total", "pct"])

    os.chdir(ROBOTWIN)

    if args.ckpt1 and args.cfg1:
        print(f"\n=== Stage 1: {args.cfg1} ({len(TASKS_STAGE1)} tasks) ===")
        run_stage(1, args.cfg1, args.ckpt1, TASKS_STAGE1, args.test_num,
                  log_dir, csv_path, args.force, args.video_save_freq, args.exec_horizon)

    if args.ckpt2 and args.cfg2:
        print(f"\n=== Stage 2: {args.cfg2} ({len(TASKS_STAGE2)} tasks) ===")
        run_stage(2, args.cfg2, args.ckpt2, TASKS_STAGE2, args.test_num,
                  log_dir, csv_path, args.force, args.video_save_freq, args.exec_horizon)

    print(f"\nDone. CSV: {csv_path}")
    print_table(csv_path, TASKS_STAGE1 if args.ckpt1 else None, TASKS_STAGE2 if args.ckpt2 else None)

if __name__ == "__main__":
    main()
