"""Driver for the remaining benchmark passes.

The representation x learner grid is expensive, so it is run in passes: the
first pass (already complete) covers the single-representation cells and the
matched descriptor ensemble, this driver covers the concatenated
representations and then the four auxiliary partitions. Passes are strictly
sequential because each one saturates the available cores.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

HYBRID = ["ECFP4+ChemBERTa-ZINC", "ChemBERTa-ZINC+Mordred", "ECFP4+Mordred"]
# On the auxiliary partitions we carry the leading cell of each family rather
# than the whole grid: the question there is how each family degrades, not
# which learner wins.
AUX_REPS = ["ECFP4", "RDKit-desc", "Mordred", "ChemBERTa-ZINC", "MoLFormer-XL", "ECFP4+ChemBERTa-ZINC"]

JOBS = [
    ("stratified", HYBRID, ["Ridge", "SVR", "ExtraTrees", "CatBoost"], "a5_matrix_stratified_partB"),
    ("scaffold", AUX_REPS, ["Ridge", "SVR", "ExtraTrees"], "a5_matrix_scaffold_partA"),
    ("cluster", AUX_REPS, ["Ridge", "SVR", "ExtraTrees"], "a5_matrix_cluster_partA"),
    ("dissimilarity", AUX_REPS, ["Ridge", "SVR", "ExtraTrees"], "a5_matrix_dissimilarity_partA"),
    ("source", AUX_REPS, ["Ridge", "SVR", "ExtraTrees"], "a5_matrix_source_partA"),
]


def wait_for_partA():
    """Do not compete with the first pass for cores."""
    target = os.path.join(OUT, "a5_matrix_stratified_partA.csv")
    while not os.path.exists(target):
        print("waiting for the first benchmark pass to finish ...", flush=True)
        time.sleep(30)
    time.sleep(15)


def main():
    if "--no-wait" not in sys.argv:
        wait_for_partA()
    for split, reps, learners, tag in JOBS:
        if os.path.exists(os.path.join(OUT, f"{tag}.csv")):
            print(f"##### {tag} already present, skipped", flush=True)
            continue
        print(f"\n##### {split}: {len(reps)} representations x {len(learners)} learners -> {tag}", flush=True)
        t0 = time.time()
        cmd = [
            sys.executable, "-u", os.path.join(HERE, "a5_matrix.py"),
            "--split", split, "--reps", *reps, "--learners", *learners,
            "--no-ensemble", "--out", tag,
        ]
        subprocess.run(cmd, cwd=ROOT, check=False)
        print(f"##### {tag} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
