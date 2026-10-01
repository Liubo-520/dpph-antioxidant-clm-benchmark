"""Representation x learner grid on the repeated partitions.

Every cell is tuned exactly as in the first revision -- GridSearchCV, five-fold
shuffled KFold inside the training partition, scoring on R2, refit on the whole
training partition -- and the held-out compounds are scored once, after the
refit. The estimator pipelines and search spaces are imported from
a5_matrix.py so they cannot drift from the first revision.

What is stored per cell is what training-only model selection needs: the mean
inner cross-validated R2 of the selected hyperparameters (the quantity the grid
search itself maximised) and the held-out predictions. The representative of a
family in a partition is later taken as the cell with the highest inner score
in that family (c8_repeated_summary.py), so the hold-out plays no part in
choosing it.

The cells are those of the four harder partitions of the first revision
(Supplementary Table S14), run in three passes so that the analysis the
reviewer asked for is available first and each later pass only widens the
candidate set:

  pass 1  the five cells that training-set Q2_CV selected as family
          representatives on the primary partitions (one per grid family);
  pass 2  the thirteen linear and kernel cells of the same six inputs, which
          with pass 1 give 18 candidates for nested re-selection in every
          partition;
  pass 3  the four remaining tree-ensemble cells. They are the most expensive
          cells of the grid, and on none of the five primary partitions did
          training-only selection choose any of them, so they are run last and
          only as far as the time budget allows.

    python c2_grid.py --worker 0 --of 3 --threads 4 --passes 1 2
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from common import (DESIGNS, N_FOLDS, OUT, REP_FOLDS, family_key, load_partitions, pin, r1_import, read_json,
                    write_json)

r1_import()
from a5_matrix import INNER_CV, REPRESENTATION_FAMILY, learner_grid, load_representations  # noqa: E402

RESULTS = os.path.join(OUT, "grid")
os.makedirs(RESULTS, exist_ok=True)

PASSES = {
    1: [("ECFP4", "ExtraTrees"), ("RDKit-desc", "ExtraTrees"), ("MoLFormer-XL", "SVR"),
        ("ECFP4+Mordred", "ExtraTrees"), ("ECFP4+ChemBERTa-ZINC", "ExtraTrees")],
    2: [("ECFP4", "Ridge"), ("ECFP4", "SVR"),
        ("RDKit-desc", "Ridge"), ("RDKit-desc", "SVR"),
        ("Mordred", "Ridge"), ("Mordred", "SVR"),
        ("ChemBERTa-ZINC", "Ridge"), ("ChemBERTa-ZINC", "SVR"),
        ("MoLFormer-XL", "Ridge"),
        ("ECFP4+Mordred", "Ridge"), ("ECFP4+Mordred", "SVR"),
        ("ECFP4+ChemBERTa-ZINC", "Ridge"), ("ECFP4+ChemBERTa-ZINC", "SVR")],
    3: [("Mordred", "ExtraTrees"), ("ChemBERTa-ZINC", "ExtraTrees"), ("MoLFormer-XL", "ExtraTrees"),
        ("ECFP4+Mordred", "CatBoost")],
}


def run_partition(pid, part, reps, y, threads, cells):
    out_path = os.path.join(RESULTS, f"{pid}.json")
    done = read_json(out_path) if os.path.exists(out_path) else {"partition": pid, "cells": {}}
    train_idx = np.asarray(part["train_idx"])
    test_idx = np.asarray(part["test_idx"])
    for rep_name, learner in cells:
        key = f"{rep_name} | {learner}"
        if key in done["cells"]:
            continue
        X = reps[rep_name]
        est, grid = learner_grid(learner, X.shape[1])
        heavy = learner in ("ExtraTrees", "CatBoost")
        if learner == "ExtraTrees":
            est.set_params(mdl__n_jobs=threads)
        elif learner == "CatBoost":
            est.set_params(mdl__thread_count=threads)
        t0 = time.perf_counter()
        # the linear and kernel cells are fitted in this process, one fold at a
        # time: a pool of worker processes costs more memory than it saves time
        search = GridSearchCV(est, grid, cv=INNER_CV, scoring="r2", n_jobs=1, refit=True)
        search.fit(X[train_idx], y[train_idx])
        pred = search.best_estimator_.predict(X[test_idx])
        b = search.best_index_
        family = REPRESENTATION_FAMILY.get(rep_name, "Other")
        # another worker may have added cells to this partition in the meantime
        if os.path.exists(out_path):
            done["cells"] = {**read_json(out_path)["cells"], **done["cells"]}
        done["cells"][key] = {
            "representation": rep_name,
            "learner": learner,
            "family": family,
            "family_key": family_key(family, rep_name),
            "n_features": int(X.shape[1]),
            "best_params": {k: (v if isinstance(v, str) else float(v))
                            for k, v in search.best_params_.items()},
            "cv_r2_inner_mean": float(search.best_score_),
            "cv_r2_inner_sd": float(search.cv_results_["std_test_score"][b]),
            "test_r2": float(r2_score(y[test_idx], pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
            "test_mae": float(mean_absolute_error(y[test_idx], pred)),
            "test_predictions": pred.tolist(),
            "fit_seconds": float(time.perf_counter() - t0),
        }
        write_json(out_path, done)
        c = done["cells"][key]
        print(f"[{pid}] {key:34s} inner R2={c['cv_r2_inner_mean']:.4f} "
              f"held-out R2={c['test_r2']:.4f} ({c['fit_seconds']:.0f}s)", flush=True)


def main():
    pin("cpu")
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--passes", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS))
    ap.add_argument("--folds", nargs="+", type=int, default=list(REP_FOLDS))
    ap.add_argument("--partitions", nargs="+", default=None,
                    help="explicit partition ids; overrides --designs and --folds")
    ap.add_argument("--reverse", action="store_true",
                    help="take the queue from the end, to help a worker already running")
    a = ap.parse_args()

    # the activity values in dataset order; read from the curated table so that
    # this CPU-only worker does not have to load the graph objects
    import pandas as pd
    from common import R1_OUT
    y = pd.read_csv(os.path.join(R1_OUT, "curated_dataset.csv"))["pIC50"].to_numpy(float)

    reps = load_representations()
    parts = load_partitions()["partitions"]
    queue = a.partitions or [f"{d}_f{k:02d}" for k in a.folds for d in a.designs]
    mine = queue[a.worker::a.of]
    if a.reverse:
        mine = mine[::-1]
    print(f"worker {a.worker}/{a.of}: {len(mine)} partitions, {a.threads} threads, passes {a.passes}",
          flush=True)
    for number in a.passes:
        t0 = time.perf_counter()
        for pid in mine:
            try:
                run_partition(pid, parts[pid], reps, y, a.threads, PASSES[number])
            except Exception as exc:
                print(f"[{pid}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(f"worker {a.worker}: pass {number} finished in {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
