"""Driver for the GPU work that follows fine-tuning.

The graph baseline, the pooling-sensitivity scan and the attribution analysis
all need the accelerator, so they are run strictly after the fine-tuning
schedule and strictly one at a time.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")
CKPT = os.path.join(os.path.dirname(HERE), "checkpoints")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# the last job of the fine-tuning schedule
SENTINEL = os.path.join(OUT, "a3_finetune_chemberta_zinc_source.json")

JOBS = [
    ("pooling sensitivity", ["a2b_pooling.py"], "a2b_pooling.json"),
    ("graph network, stratified", ["a5b_gnn.py", "--split", "stratified"], "a5b_gnn.json"),
    ("graph network, scaffold", ["a5b_gnn.py", "--split", "scaffold"], "a5b_gnn_scaffold.json"),
    ("graph network, cluster", ["a5b_gnn.py", "--split", "cluster"], "a5b_gnn_cluster.json"),
    ("graph network, dissimilarity", ["a5b_gnn.py", "--split", "dissimilarity"], "a5b_gnn_dissimilarity.json"),
    ("graph network, source", ["a5b_gnn.py", "--split", "source"], "a5b_gnn_source.json"),
    ("attribution", ["a6_attribution.py"], "a6_attribution.json"),
]


def main():
    if "--no-wait" not in sys.argv:
        while not os.path.exists(SENTINEL):
            print("waiting for the fine-tuning schedule to finish ...", flush=True)
            time.sleep(60)
        time.sleep(20)
    for label, argv, product in JOBS:
        if os.path.exists(os.path.join(OUT, product)):
            print(f"##### {label} already present, skipped", flush=True)
            continue
        print(f"\n##### {label}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-u", os.path.join(HERE, argv[0]), *argv[1:]], cwd=ROOT, check=False)
        print(f"##### {label} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
