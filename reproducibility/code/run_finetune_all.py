"""Driver for the complete leak-free fine-tuning schedule.

Stage 1 fine-tunes both chemical language models on the primary
chemistry-stratified partition with three independent seeds, which supplies the
headline numbers and the run-to-run variability. Stage 2 repeats the best model
once on each of the four harder partitions so that generalisation can be
reported under scaffold, cluster, dissimilarity and source-disjoint hold-outs.

Jobs whose output already carries every requested seed are skipped, so the
driver can be restarted after an interruption without repeating hours of
finished work.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")
SCRIPT = os.path.join(HERE, "a3_finetune_clean.py")

TAG = {"ChemBERTa-ZINC": "chemberta_zinc", "ChemBERTa-MTR": "chemberta_mtr"}

JOBS = [
    # (models, seeds, split, folds) -- the same five-fold protocol everywhere
    (["ChemBERTa-ZINC"], [42, 1, 2026], "stratified", 5),
    (["ChemBERTa-MTR"], [42], "stratified", 5),
    (["ChemBERTa-ZINC"], [42], "scaffold", 5),
    (["ChemBERTa-ZINC"], [42], "cluster", 5),
    (["ChemBERTa-ZINC"], [42], "dissimilarity", 5),
    (["ChemBERTa-ZINC"], [42], "source", 5),
]


def already_done(models, seeds, split):
    """True when every requested model/seed combination is already on disk."""
    suffix = "" if split == "stratified" else f"_{split}"
    for m in models:
        path = os.path.join(OUT, f"a3_finetune_{TAG[m]}{suffix}.json")
        if not os.path.exists(path):
            return False
        try:
            have = set(json.load(open(path, encoding="utf-8"))["per_seed"])
        except Exception:
            return False
        if not {str(s) for s in seeds} <= have:
            return False
    return True


for models, seeds, split, folds in JOBS:
    if already_done(models, seeds, split):
        print(f"\n########## {split}: {', '.join(models)} already complete, skipped", flush=True)
        continue
    cmd = [sys.executable, "-u", SCRIPT, "--split", split, "--folds", str(folds),
           "--models", *models, "--seeds", *[str(s) for s in seeds]]
    print(f"\n########## {split}: {', '.join(models)} seeds {seeds} ##########", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"########## {split} finished rc={r.returncode} in {(time.time()-t0)/60:.1f} min",
          flush=True)
