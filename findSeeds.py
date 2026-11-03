import os
import subprocess
import time
import signal
import json


GPU_ID = "7"
TASK_CONFIG = "demo_clean_eval"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ST_SEED = 1000000

tasks = [
    "stack_bowls_three",
    "beat_block_hammer",
]
for task in tasks:
    print(f"Collecting data for {task}")
    task_name = task.rsplit(".", 1)[0]
    process = subprocess.Popen(
    ["bash", "collect_data.sh", task_name, TASK_CONFIG, GPU_ID, f"{ST_SEED}"],
    cwd=SCRIPT_DIR,
    start_new_session=True,  # ← required: isolates child into its own process group
    )

    seed_path = os.path.join(SCRIPT_DIR, "data", task_name, TASK_CONFIG, "seed.txt")
    seedCount = 0
    try:
        with open(seed_path, "r", encoding="utf-8") as f:
                seed_array = [int(x) for x in f.read().split() if x.strip()]
                seedCount = len(seed_array)
    except:
        seedCount = 0
    seed_array = []
    print(f"seedCount: {seedCount}")
    while not os.path.exists(seed_path) or seedCount < 50:
        if os.path.isfile(seed_path):
            with open(seed_path, "r", encoding="utf-8") as f:
                seed_array = [int(x) for x in f.read().split() if x.strip()]
            seedCount = len(seed_array)
            if seedCount > 50:
                #json.dump(seed_array, open("/home/yaugand2/sim2real/robotwin_aloha/openpi-generalization/third_party/robotwin/eval_seeds_cache_all/seed_${task_name}.json", "w"))
                break
        time.sleep(200)
        print(f"seedCount: {seedCount}")
    json.dump(seed_array, open(f"{SCRIPT_DIR}/cached_seeds/test_seeds/seed_{task_name}_seed{str(ST_SEED)}_n{str(50)}.json", "w"))
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # kills bash + all children
    process.wait()
    print(f"Seeds for {task}: {seed_array}")
    json.dump(seed_array, open(f"{SCRIPT_DIR}/cached_seeds/test_seeds/seed_{task_name}_seed{str(ST_SEED)}_n{str(50)}.json", "w"))
