import subprocess
import os
import sys

print("Starting parallel runs...")

os.makedirs("/workspace/autoresearch", exist_ok=True)

with open("/workspace/autoresearch/worker1.log", "w") as f1, \
     open("/workspace/autoresearch/worker2.log", "w") as f2:

    p1 = subprocess.Popen(
        ["python", "-u", "train.py"],
        cwd="/workspace/worker1",
        stdout=f1, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL  # prevent hanging on input reads
    )
    p2 = subprocess.Popen(
        ["python", "-u", "train.py"],
        cwd="/workspace/worker2",
        stdout=f2, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL
    )

    # Wait for both concurrently, with a timeout
    TIMEOUT = 3600  # 1 hour — adjust as needed
    try:
        p1.wait(timeout=TIMEOUT)
        print(f"Worker 1 finished with code {p1.returncode}")
    except subprocess.TimeoutExpired:
        print("Worker 1 timed out — killing")
        p1.kill()

    try:
        p2.wait(timeout=TIMEOUT)
        print(f"Worker 2 finished with code {p2.returncode}")
    except subprocess.TimeoutExpired:
        print("Worker 2 timed out — killing")
        p2.kill()

print("Runs finished, merging logs...")

with open("/workspace/autoresearch/run.log", "a") as f_out:
    for label, path in [("WORKER 1", "/workspace/autoresearch/worker1.log"),
                        ("WORKER 2", "/workspace/autoresearch/worker2.log")]:
        f_out.write(f"=== {label} ===\n")
        with open(path, "r") as f:
            f_out.write(f.read())

print("Done!")