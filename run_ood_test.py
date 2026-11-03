#!/usr/bin/env python3
"""
OOD evaluation for the 4 stack_blocks_two_* tasks.

    python run_ood_test.py --cfg <name> --ckpt <path> --log_dir <dir> [--exec_horizon 25]

Results saved to <log_dir>/ood_results.csv (separate from other task results).
Resume: skips tasks already in ood_results.csv. Use --force to rerun all.
"""
import argparse, csv, os, re, signal, subprocess, sys
from pathlib import Path

ROBOTWIN = Path(__file__).resolve().parent

TASKS_OOD = [
    "stack_blocks_two_sphere",
    "stack_blocks_two_cylinder",
    "stack_blocks_two_sphere_distractor",
    "stack_blocks_two_cylinder_distractor",
]

ANSI    = re.compile(r"\x1b\[[0-9;]*m")
SUCCESS = re.compile(r"Success rate:\s+(\d+)/(\d+)\s+=>\s+([0-9.]+)%")


def parse_success(lines):
    for line in reversed(lines):
        m = SUCCESS.search(ANSI.sub("", line))
        if m:
            return int(m.group(1)), int(m.group(2)), float(m.group(3))
    return None, None, None


def run_one(task, args, results_csv):
    cmd = [
        "python", "script/eval_policy.py",
        "--config", "policy/pi05/deploy_policy.yml", "--overrides",
        "--task_name", task, "--task_config", "demo_clean_eval",
        "--train_config_name", args.cfg, "--model_name", "pi05_base",
        "--ckpt_setting", "full_ft_frontview",
        "--seed", "1000000", "--test_num", str(args.test_num),
        "--policy_name", "pi05", "--checkpoint_id", args.ckpt,
        "--seeds_cache_dir", str(ROBOTWIN / "cached_seeds" / "test_seeds"),
        "--video_save_freq", "10",
        "--eval_dir", args.log_dir,
    ]
    if args.exec_horizon is not None:
        cmd += ["exec_horizon", str(args.exec_horizon)]

    env = {**os.environ, "PYTHONWARNINGS": "ignore::UserWarning", "PYTHONUNBUFFERED": "1"}
    kwargs = dict(cwd=str(ROBOTWIN), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if os.name == "posix":
        kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(cmd, **kwargs)

    lines = []
    log_path = Path(args.log_dir) / "ood_eval.log"
    with open(log_path, "ab") as log_f:
        try:
            for raw in proc.stdout:
                sys.stdout.buffer.write(raw)
                sys.stdout.buffer.flush()
                decoded = raw.decode("utf-8", errors="replace")
                if not decoded.startswith("step: "):
                    log_f.write(raw)
                lines.append(decoded)
        except KeyboardInterrupt:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            raise
    proc.wait()

    suc, total, pct = parse_success(lines)
    with open(results_csv, "a", newline="") as f:
        csv.writer(f).writerow([args.cfg, args.ckpt, task, suc, total, pct])
    print(f"  => {suc}/{total} = {pct}%")
    return pct


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg",          required=True)
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--log_dir",      required=True)
    p.add_argument("--test_num",     type=int, default=50)
    p.add_argument("--exec_horizon", type=int, default=None)
    p.add_argument("--force",        action="store_true")
    args = p.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    results_csv = os.path.join(args.log_dir, "ood_results.csv")

    if not Path(results_csv).exists():
        with open(results_csv, "w", newline="") as f:
            csv.writer(f).writerow(["train_config", "checkpoint", "task", "successes", "total", "pct"])

    done = set()
    if not args.force and Path(results_csv).exists():
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("pct"):
                    done.add(row["task"])

    pcts = {}
    for i, task in enumerate(TASKS_OOD):
        if task in done:
            print(f"[{i+1}/4] SKIP {task} (already done)")
            with open(results_csv, newline="") as f:
                for row in csv.DictReader(f):
                    if row["task"] == task and row.get("pct"):
                        pcts[task] = float(row["pct"])
            continue
        print(f"[{i+1}/4] {task}")
        pct = run_one(task, args, results_csv)
        if pct is not None:
            pcts[task] = pct

    print(f"\n{'task':<45} {'pct':>8}")
    print("-" * 55)
    for task in TASKS_OOD:
        print(f"{task:<45} {f'{pcts[task]:.1f}%' if task in pcts else 'n/a':>8}")
    if pcts:
        print(f"{'Average':<45} {sum(pcts.values())/len(pcts):.1f}%")


if __name__ == "__main__":
    main()
