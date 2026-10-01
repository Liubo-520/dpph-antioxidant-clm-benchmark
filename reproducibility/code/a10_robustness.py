"""Two robustness results the revision claims but did not yet report.

1. Bootstrap confidence intervals for RMSE and MAE, not only for R-squared, on
   every model of every partition. The response letter tells the reviewers that
   every reported metric carries an interval; a7 computes all three on the
   primary partition but only R-squared reaches a table, and the harder
   partitions carried point estimates only.

2. The endpoint is described throughout as a 30-minute DPPH readout. Most of the
   records say exactly that, but a minority give a window ("after 20 to 30
   mins", "30 to 60 mins", "up to 300 mins"). This counts them exactly and
   refits the two leading grid cells on the strictly single-timepoint subset, so
   the text can state what the heterogeneity is worth instead of asserting
   homogeneity.

    python a10_robustness.py            both analyses
    python a10_robustness.py --cis      intervals only (seconds)
"""

import argparse
import json
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

import a5_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")

SPLITS = ["stratified", "scaffold", "cluster", "dissimilarity", "source"]
N_BOOT = 2000
SEED = 42

# A description states a single 30-minute readout unless it gives a range of
# times. The range has to be tied to a time unit: "at 10 to 60 ug/ml after 30
# mins" is a concentration range on a 30-minute readout, not a window.
TIME = r"(?:min|mins|minute|minutes|hr|hrs|hour|hours)"
RANGE = re.compile(r"\b\d+\s*to\s*\d+\s*" + TIME + r"\b|\bup to\s*\d+\s*" + TIME + r"\b", re.I)


# ------------------------------------------------------- 1. metric intervals
def metric_cis():
    rng = np.random.default_rng(SEED)
    rows = []
    for split in SPLITS:
        path = os.path.join(OUT, f"a5_matrix_{split}_predictions.json")
        if not os.path.exists(path):
            print(f"  (no predictions for {split})")
            continue
        payload = json.load(open(path, encoding="utf-8"))
        y = np.asarray(payload["y_test"], float)
        idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
        for model, pred in payload["predictions"].items():
            # each entry holds the held-out predictions and the out-of-fold ones
            p = np.asarray(pred["test"] if isinstance(pred, dict) else pred, float)
            if p.shape != y.shape:
                continue
            r2 = np.array([r2_score(y[i], p[i]) for i in idx])
            rmse = np.array([np.sqrt(mean_squared_error(y[i], p[i])) for i in idx])
            mae = np.array([mean_absolute_error(y[i], p[i]) for i in idx])
            q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
            rows.append({
                "split": split,
                "model": model,
                "r2": r2_score(y, p),
                "r2_lo": q(r2)[0], "r2_hi": q(r2)[1],
                "rmse": float(np.sqrt(mean_squared_error(y, p))),
                "rmse_lo": q(rmse)[0], "rmse_hi": q(rmse)[1],
                "mae": float(mean_absolute_error(y, p)),
                "mae_lo": q(mae)[0], "mae_hi": q(mae)[1],
            })
        print(f"  {split}: {sum(r['split'] == split for r in rows)} models")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a10_metric_cis.csv"), index=False)
    print(f"wrote a10_metric_cis.csv ({len(df)} rows, {N_BOOT} resamples)")
    return df


# --------------------------------------------- 2. assay-window sensitivity
def assay_sensitivity(cells=(("ECFP4+Mordred", "ExtraTrees"), ("ECFP4", "ExtraTrees"))):
    data = pd.read_csv(os.path.join(OUT, "curated_dataset.csv"))
    windowed = data["assay_description"].astype(str).apply(lambda s: bool(RANGE.search(s)))
    wording = (data.loc[windowed, "assay_description"].value_counts().to_dict())

    result = {
        "n_records": int(len(data)),
        "n_single_timepoint": int((~windowed).sum()),
        "n_window_wording": int(windowed.sum()),
        "pct_window_wording": float(100 * windowed.mean()),
        "window_descriptions": wording,
        "cells": [],
    }
    print("records with a window wording: %d of %d (%.1f%%)"
          % (result["n_window_wording"], result["n_records"], result["pct_window_wording"]))
    for text, n in sorted(wording.items(), key=lambda kv: -kv[1]):
        print("   %4d  %s" % (n, text[:110]))

    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        y = pickle.load(fh)["df"]["pIC50"].to_numpy(float)
    splits = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))["splits"]
    train_idx = np.array(splits["stratified"]["train_idx"])
    test_idx = np.array(splits["stratified"]["test_idx"])
    reps = a5_matrix.load_representations("stratified")

    keep = (~windowed).to_numpy()
    train_keep = train_idx[keep[train_idx]]
    test_keep = test_idx[keep[test_idx]]
    result["n_train_full"] = int(len(train_idx))
    result["n_test_full"] = int(len(test_idx))
    result["n_train_single_timepoint"] = int(len(train_keep))
    result["n_test_single_timepoint"] = int(len(test_keep))
    print("\nstratified partition: %d/%d training and %d/%d held-out compounds are "
          "single-timepoint" % (len(train_keep), len(train_idx), len(test_keep), len(test_idx)))

    full = pd.read_csv(os.path.join(OUT, "a5_matrix_stratified.csv"))
    for rep_name, learner in cells:
        X = reps[rep_name]
        est, grid = a5_matrix.learner_grid(learner, X.shape[1])
        t0 = time.perf_counter()
        search = GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=SEED),
                              scoring="r2", n_jobs=1, refit=True)
        search.fit(X[train_keep], y[train_keep])
        pred = search.best_estimator_.predict(X[test_keep])
        row = full[(full["representation"] == rep_name) & (full["learner"] == learner)]
        entry = {
            "representation": rep_name,
            "learner": learner,
            "r2_full": float(row["test_r2"].iloc[0]) if len(row) else None,
            "rmse_full": float(row["test_rmse"].iloc[0]) if len(row) else None,
            "mae_full": float(row["test_mae"].iloc[0]) if len(row) else None,
            "r2_single_timepoint": float(r2_score(y[test_keep], pred)),
            "rmse_single_timepoint": float(np.sqrt(mean_squared_error(y[test_keep], pred))),
            "mae_single_timepoint": float(mean_absolute_error(y[test_keep], pred)),
            "refit_seconds": time.perf_counter() - t0,
        }
        entry["delta_r2"] = entry["r2_single_timepoint"] - entry["r2_full"]
        result["cells"].append(entry)
        print("  %-22s %-11s  R2 %.4f (all) -> %.4f (single-timepoint), delta %+.4f"
              % (rep_name, learner, entry["r2_full"], entry["r2_single_timepoint"],
                 entry["delta_r2"]))

    with open(os.path.join(OUT, "a10_assay_sensitivity.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("wrote a10_assay_sensitivity.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cis", action="store_true", help="intervals only")
    ap.add_argument("--sensitivity", action="store_true", help="assay sensitivity only")
    a = ap.parse_args()
    if not a.sensitivity:
        metric_cis()
    if not a.cis:
        print()
        assay_sensitivity()


if __name__ == "__main__":
    main()
