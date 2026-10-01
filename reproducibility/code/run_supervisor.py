"""Keep the three long-running drivers alive until the analysis is complete.

Two earlier attempts at this pipeline lost about half an hour each because the
driver processes were reaped while they were still working. Each driver is
already restart-safe -- it skips whatever is finished -- so the cheapest
insurance is a supervisor that notices a driver has gone and starts it again.

When every driver has produced its final output the supervisor runs
run_finish.py once and exits.

    python run_supervisor.py
"""

import os
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
OUT = os.path.join(ANALYSIS, "outputs")
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))

POLL_SECONDS = 120

# (script, extra argv, the output that means this driver is finished, log stem)
DRIVERS = [
    ("run_finetune_all.py", [], "a3_finetune_chemberta_zinc_source.json", "a3_finetune_all2"),
    ("run_matrix_all.py", ["--no-wait"], "a5_matrix_source_partA.csv", "a5_rest2"),
    ("run_gpu_rest.py", [], "a6_attribution.json", "gpu_rest2"),
]


def running(script):
    """True when a python process is already executing this script."""
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == me or not p.info["cmdline"]:
            continue
        if any(script in str(c) for c in p.info["cmdline"]):
            return True
    return False


def start(script, extra, stem):
    log = open(os.path.join(OUT, f"{stem}.log"), "a", buffering=1)
    err = open(os.path.join(OUT, f"{stem}.err.log"), "a", buffering=1)
    subprocess.Popen(
        [sys.executable, "-u", os.path.join(HERE, script), *extra],
        cwd=ROOT, stdout=log, stderr=err,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    print(f"[{time.strftime('%H:%M:%S')}] restarted {script}", flush=True)


def main():
    while True:
        pending = []
        for script, extra, product, stem in DRIVERS:
            if os.path.exists(os.path.join(OUT, product)):
                continue
            pending.append(script)
            if not running(script):
                start(script, extra, stem)
        if not pending:
            break
        print(f"[{time.strftime('%H:%M:%S')}] waiting on: {', '.join(pending)}", flush=True)
        time.sleep(POLL_SECONDS)

    print(f"[{time.strftime('%H:%M:%S')}] all drivers complete; running run_finish.py", flush=True)
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_finish.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
