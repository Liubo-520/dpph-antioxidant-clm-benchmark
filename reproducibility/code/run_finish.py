"""Everything that happens once the model fitting is finished.

The expensive work (a2--a6) is driven by run_finetune_all.py, run_matrix_all.py
and run_gpu_rest.py. This driver performs the cheap, strictly ordered
downstream steps: merge each partition's benchmark, compute the statistics,
applicability domain and utility analyses, then regenerate every macro, table
and figure of the revision package and rebuild the deposit.

    python run_finish.py            run every step, stop at the first failure
    python run_finish.py --check    report what is present without running

It is safe to run more than once.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
OUT = os.path.join(ANALYSIS, "outputs")
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))

SPLITS = ["stratified", "scaffold", "cluster", "dissimilarity", "source"]

# (label, argv, the file that proves the step ran)
STEPS = [
    # every split now has a partB: the fingerprint--descriptor hybrid, which
    # leads the stratified partition and had never been scored on the harder
    # ones, so the generalisation table had no row for the best model of the
    # study outside the easiest partition
    (f"merge benchmark, {s}", ["a5c_merge.py", "--split", s, "--parts", "partA", "partB"],
     f"a5_matrix_{s}.csv")
    for s in SPLITS
] + [
    ("statistics, applicability domain, stacking", ["a7_stats_ad.py"], "a7_stats_ad.json"),
    ("chemical space, matched pairs, triage", ["a8_chemspace_utility.py"], "a8_chemspace_utility.json"),
    ("bootstrap intervals and assay-window sensitivity", ["a10_robustness.py"],
     "a10_metric_cis.csv"),
]

RENDER = [
    ("manuscript cross-reference numbers", ["b6_crossrefs.py"]),
    ("LaTeX macros and tables", ["b1_make_macros.py"]),
    ("manuscript figures", ["b2_make_figures.py"]),
    ("response-letter figures", ["b2b_response_figures.py"]),
    ("response letter as DOCX", ["b7_response_docx.py"]),
    ("released model and its usage instructions", ["b9_release_model.py"]),
    # after the macros and tables exist, so the comparison is made against the
    # manuscript as it will be submitted
    ("marked-up manuscript against the submitted version", ["b8_marked_manuscript.py"]),
    ("reproducibility deposit", ["b4_package.py"]),
]

# run last, and never fatal: it reports which directional sentences the data
# actually supports, which is a judgement for the author, not the driver
AUDIT = ("directional claims in the text", ["b5_verify_claims.py"])

# inputs the downstream steps need, with the driver that produces each
PREREQS = {
    "a5_matrix_stratified_partA.csv": "a5_matrix.py (first benchmark pass)",
    "a5_matrix_stratified_partB.csv": "run_matrix_all.py",
    "a5_matrix_scaffold_partA.csv": "run_matrix_all.py",
    "a5_matrix_cluster_partA.csv": "run_matrix_all.py",
    "a5_matrix_dissimilarity_partA.csv": "run_matrix_all.py",
    "a5_matrix_source_partA.csv": "run_matrix_all.py",
    "a3_finetune_chemberta_zinc.json": "run_finetune_all.py",
    "a3_finetune_chemberta_mtr.json": "run_finetune_all.py",
    "a3_finetune_chemberta_zinc_scaffold.json": "run_finetune_all.py",
    "a3_finetune_chemberta_zinc_cluster.json": "run_finetune_all.py",
    "a3_finetune_chemberta_zinc_dissimilarity.json": "run_finetune_all.py",
    "a3_finetune_chemberta_zinc_source.json": "run_finetune_all.py",
    "a2b_pooling.json": "run_gpu_rest.py",
    "a5b_gnn.json": "run_gpu_rest.py",
    "a6_attribution.json": "run_gpu_rest.py",
}


def check():
    missing = []
    for name, producer in PREREQS.items():
        ok = os.path.exists(os.path.join(OUT, name))
        print(f"  [{'x' if ok else ' '}] {name:52s} {'' if ok else '<- ' + producer}")
        if not ok:
            missing.append(name)
    return missing


def run(label, argv):
    print(f"\n##### {label}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, argv[0]), *argv[1:]], cwd=ROOT)
    print(f"##### {label}: rc={r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report readiness and exit")
    ap.add_argument("--force", action="store_true", help="run even if prerequisites are missing")
    a = ap.parse_args()

    print("Prerequisites:")
    missing = check()
    if a.check:
        return
    if missing and not a.force:
        raise SystemExit(
            f"\n{len(missing)} prerequisite(s) still missing; wait for the driver named "
            "beside each, or pass --force to build a partial package."
        )

    for label, argv, product in STEPS:
        if run(label, argv) != 0:
            print(f"!!! {label} failed; stopping", flush=True)
            return
    for label, argv in RENDER:
        if run(label, argv) != 0:
            print(f"!!! {label} failed; stopping", flush=True)
            return
    run(*AUDIT)
    print("\nAll downstream steps complete. Read the claim audit above, then build "
          "the PDFs with Revision_R1_new\\build.ps1.")


if __name__ == "__main__":
    main()
