"""Check that the second-revision workers reproduce first-revision values.

The repeated partitions are compared with the primary ones, so the code that
fits a model on a repeated partition has to give, on a primary partition, the
value the first revision reported there. This script refits a few inexpensive
grid cells on primary partitions with c2_grid.run_partition and compares the
held-out R2 with the stored first-revision matrix; with --gnn it does the same
for the graph network through c4_gnn.run_partition.

    python c0_reproduce_primary.py
    python c0_reproduce_primary.py --gnn scaffold
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

from common import OUT, R1_OUT, read_json, write_json

CELLS = [("MoLFormer-XL", "SVR"), ("RDKit-desc", "ExtraTrees"), ("ECFP4", "Ridge"), ("ChemBERTa-ZINC", "SVR")]
SPLITS = ("scaffold", "cluster")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", nargs="*", default=None, help="primary partitions on which to refit the graph network")
    a = ap.parse_args()
    primary = read_json(os.path.join(R1_OUT, "a4_splits.json"))["splits"]
    report = read_json(os.path.join(OUT, "c0_reproduce_primary.json")) if os.path.exists(
        os.path.join(OUT, "c0_reproduce_primary.json")) else {"grid": [], "gnn": []}

    if a.gnn is None:
        import c2_grid
        y = pd.read_csv(os.path.join(R1_OUT, "curated_dataset.csv"))["pIC50"].to_numpy(float)
        reps = c2_grid.load_representations()
        report["grid"] = []
        for split in SPLITS:
            pid = f"_primary_{split}"
            path = os.path.join(c2_grid.RESULTS, f"{pid}.json")
            if os.path.exists(path):
                os.remove(path)
            c2_grid.run_partition(pid, primary[split], reps, y, 4, CELLS)
            got = read_json(path)["cells"]
            os.remove(path)
            old = pd.read_csv(os.path.join(R1_OUT, f"a5_matrix_{split}.csv"))
            for rep_name, learner in CELLS:
                row = old[(old["representation"] == rep_name) & (old["learner"] == learner)]
                new = got[f"{rep_name} | {learner}"]["test_r2"]
                ref = float(row["test_r2"].iloc[0])
                report["grid"].append({"partition": split, "cell": f"{rep_name} | {learner}",
                                       "first_revision": ref, "second_revision_code": new,
                                       "abs_difference": abs(new - ref)})
                print(f"{split:9s} {rep_name + ' | ' + learner:28s} first revision {ref:.4f}  now {new:.4f}  "
                      f"|diff| {abs(new - ref):.1e}")
        worst = max(r["abs_difference"] for r in report["grid"])
        print(f"largest difference over {len(report['grid'])} grid cells: {worst:.1e}")
    else:
        import c4_gnn
        _, y, _, data = c4_gnn.load_dataset()
        for split in a.gnn or ["scaffold"]:
            pid = f"_primary_{split}"
            path = os.path.join(c4_gnn.RESULTS, f"{pid}.json")
            if os.path.exists(path):
                os.remove(path)
            c4_gnn.run_partition(pid, primary[split], data["graphs"], y)
            new = read_json(path)["test_r2"]
            os.remove(path)
            suffix = "" if split == "stratified" else f"_{split}"
            ref = read_json(os.path.join(R1_OUT, f"a5b_gnn{suffix}.json"))["test_r2"]
            report["gnn"] = [r for r in report["gnn"] if r["partition"] != split]
            report["gnn"].append({"partition": split, "first_revision": ref, "second_revision_code": new,
                                  "abs_difference": abs(new - ref)})
            print(f"{split:9s} graph network: first revision {ref:.4f}  now {new:.4f}  |diff| {abs(new - ref):.1e}")

    write_json(os.path.join(OUT, "c0_reproduce_primary.json"), report)


if __name__ == "__main__":
    sys.exit(main())
