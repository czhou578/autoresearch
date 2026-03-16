import subprocess
import os
import time
import shutil
import signal

def kill_proc_tree(p):
    """Safely kills process and its children, handling OS-specific process groups."""
    try:
        if os.name == 'posix':
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        else:
            # Fallback for Windows to kill process tree
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.kill()
    except Exception as e:
        print(f"Error killing process tree: {e}")
        p.kill() # absolute fallback
        
    try:
        p.wait(timeout=5)  # give the OS 5 seconds to reap it
    except subprocess.TimeoutExpired:
        print(f"Warning: Process {p.pid} is stuck in an uninterruptible state.")

print("Starting parallel runs...")

os.makedirs("/workspace/autoresearch", exist_ok=True)

TIMEOUT = 3600  # 1 hour shared deadline
LIVENESS_TIMEOUT = 600  # 10 minutes without output means stalled

# Keep log files open until both processes are fully reaped
with open("/workspace/autoresearch/worker1.log", "w") as f1, \
     open("/workspace/autoresearch/worker2.log", "w") as f2:

    p1 = subprocess.Popen(
        ["python", "-u", "train.py"],
        cwd="/workspace/worker1",
        stdout=f1, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,  # prevent hanging on input reads
        start_new_session=True     # put in a new process group for clean tree killing
    )
    p2 = subprocess.Popen(
        ["python", "-u", "train.py"],
        cwd="/workspace/worker2",
        stdout=f2, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )

    deadline = time.monotonic() + TIMEOUT
    
    # Track state for liveness probing
    processes = [
        {"name": "Worker 1", "p": p1, "log_path": "/workspace/autoresearch/worker1.log", "last_size": 0, "last_active": time.monotonic()},
        {"name": "Worker 2", "p": p2, "log_path": "/workspace/autoresearch/worker2.log", "last_size": 0, "last_active": time.monotonic()}
    ]

    # Shared polling loop
    while processes:
        if time.monotonic() > deadline:
            for info in processes:
                print(f"{info['name']} timed out (1 hour) — killing")
                kill_proc_tree(info['p'])
            break
            
        active_processes = []
        for info in processes:
            p = info["p"]
            
            # Check if process naturally exited
            if p.poll() is not None:
                print(f"{info['name']} finished with code {p.returncode}")
                continue
                
            # Liveness check (monitor stdout log size)
            try:
                size = os.path.getsize(info["log_path"])
                if size > info["last_size"]:
                    info["last_size"] = size
                    info["last_active"] = time.monotonic()
                elif time.monotonic() - info["last_active"] > LIVENESS_TIMEOUT:
                    print(f"{info['name']} considered stalled (no output for 10m) — killing")
                    kill_proc_tree(p)
                    continue
            except OSError:
                pass
                
            active_processes.append(info)
            
        processes = active_processes
        time.sleep(1)  # prevent busy waiting

print("Runs finished, merging logs...")

# Use shutil.copyfileobj to avoid loading 50GB log files into RAM natively
with open("/workspace/autoresearch/run.log", "ab") as f_out: # Open in binary for shutil
    for label, path in [("WORKER 1", "/workspace/autoresearch/worker1.log"),
                        ("WORKER 2", "/workspace/autoresearch/worker2.log")]:
        f_out.write(f"=== {label} ===\n".encode('utf-8'))
        with open(path, "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)

print("Done!")