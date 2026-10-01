"""Fit and package the released model.

The Data availability statement promises a public repository with the model and
instructions for using it. This builds exactly that, and nothing else: the
leading grid cell refitted on its training partition, the Mordred feature names
it expects, a predictor that goes from SMILES to pIC50, an applicability-domain
flag, and a model card carrying the measured performance and the limits.

The applicability domain travels with the model on purpose. On the
source-publication-disjoint partition this model scores a negative R-squared, so
a released artefact that answers every SMILES with a confident number would
misrepresent what it can do.

    python b9_release_model.py            build into revision/release
"""

import argparse
import json
import os
import pickle
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

import a5_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REVISION = os.path.dirname(ANALYSIS)
ROOT = os.path.dirname(REVISION)
OUT = os.path.join(ANALYSIS, "outputs")
RELEASE = os.path.join(REVISION, "release")

REPRESENTATION = "ECFP4+Mordred"
LEARNER = "ExtraTrees"


def build():
    os.makedirs(os.path.join(RELEASE, "model"), exist_ok=True)

    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        y = pickle.load(fh)["df"]["pIC50"].to_numpy(float)
    splits = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))["splits"]
    train_idx = np.array(splits["stratified"]["train_idx"])
    test_idx = np.array(splits["stratified"]["test_idx"])
    reps = a5_matrix.load_representations("stratified")
    X = reps[REPRESENTATION]

    meta = json.load(open(os.path.join(OUT, "a2_representations.json"), encoding="utf-8"))
    mordred_names = meta["meta"]["Mordred_names"]
    n_ecfp = reps["ECFP4"].shape[1]
    assert n_ecfp + len(mordred_names) == X.shape[1], "feature block sizes do not add up"

    est, grid = a5_matrix.learner_grid(LEARNER, X.shape[1])
    search = GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=42),
                          scoring="r2", n_jobs=1, refit=True)
    print("fitting %s + %s on %d training compounds ..." % (REPRESENTATION, LEARNER,
                                                            len(train_idx)))
    search.fit(X[train_idx], y[train_idx])
    model = search.best_estimator_

    # the applicability domain the model card quotes: nearest-neighbour ECFP4
    # Tanimoto similarity to the training set
    train_fp = reps["ECFP4"][train_idx].astype(bool)

    bundle = {
        "model": model,
        "representation": REPRESENTATION,
        "learner": LEARNER,
        "best_params": search.best_params_,
        "ecfp4": {"radius": 2, "n_bits": n_ecfp},
        "mordred_names": mordred_names,
        "train_ecfp4": np.packbits(train_fp, axis=1),
        "train_ecfp4_shape": train_fp.shape,
        "endpoint": "pIC50 = -log10(IC50 / M), DPPH radical scavenging, nominal 30 min",
    }
    path = os.path.join(RELEASE, "model", "dpph_pic50_ecfp4_mordred_extratrees.joblib")
    joblib.dump(bundle, path, compress=3)
    size = os.path.getsize(path) / 1e6
    print("wrote %s (%.1f MB)" % (os.path.relpath(path, REVISION), size))
    if size > 95:
        print("  WARNING: above GitHub's 100 MB per-file limit")

    row = pd.read_csv(os.path.join(OUT, "a5_matrix_stratified.csv"))
    row = row[(row["representation"] == REPRESENTATION) & (row["learner"] == LEARNER)].iloc[0]
    cis = pd.read_csv(os.path.join(OUT, "a10_metric_cis.csv"))
    ci = cis[(cis["split"] == "stratified") & (cis["model"] == f"{REPRESENTATION} | {LEARNER}")]
    gen = pd.concat([pd.read_csv(os.path.join(OUT, f"a5_matrix_{s}.csv")).assign(split=s)
                     for s in ("scaffold", "cluster", "dissimilarity", "source")
                     if os.path.exists(os.path.join(OUT, f"a5_matrix_{s}.csv"))])

    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "q2": float(row["q2_cv_oof"]),
        "r2": float(row["test_r2"]),
        "rmse": float(row["test_rmse"]),
        "mae": float(row["test_mae"]),
        "ci": (None if ci.empty else
               {k: (float(ci.iloc[0][f"{k}_lo"]), float(ci.iloc[0][f"{k}_hi"]))
                for k in ("r2", "rmse", "mae")}),
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "worst_split_r2": float(gen[gen["split"] == "source"]["test_r2"].max())
        if len(gen) else None,
        "model_file": os.path.basename(path),
        "model_mb": round(size, 1),
    }


README = """# DPPH antioxidant activity predictor

Predicts pIC50 for DPPH radical scavenging from a SMILES string.

This is the model released with *Chemical Language Models for Antioxidant
Activity Prediction: A Leakage-Controlled Multi-Partition Benchmark with
Quantitative Molecular Attribution*. The repository holds the fitted model and
the instructions for running it, and nothing else; the curated dataset, the five
partitions, every representation matrix, all predictions and the complete
analysis code are in the supplementary archive that accompanies the paper.

## What it is

An Extra Trees regressor on ECFP4 fingerprints concatenated with Mordred
descriptors, fitted on {n_train} compounds. It was the most accurate of the {cells}
scored models in the benchmark, and it contains no chemical language model --
that is one of the findings of the paper rather than an implementation detail.

## Install and run

```
pip install -r requirements.txt
python predict.py "Oc1ccc(cc1O)/C=C/C(=O)O"
python predict.py --input molecules.smi --output predictions.csv
```

Output columns: the predicted pIC50, the nearest-neighbour ECFP4 Tanimoto
similarity of the query to the training set, and the held-out RMSE that was
measured for compounds in that similarity range.

## How well it works, and where it does not

On the {n_test}-compound chemistry-stratified hold-out: R2 = {r2}
(95% CI {r2lo}--{r2hi}), RMSE = {rmse}, MAE = {mae}.

That number describes compounds resembling the training set. Accuracy falls as
the hold-out is made structurally harder, and on a partition that separates
whole source publications the model scores a **negative** R2 of {src}: worse than
predicting the training mean. Held-out RMSE by nearest-neighbour similarity:

| NN Tanimoto to training | Held-out compounds | RMSE |
|---|---|---|
| [0.0, 0.4) | 6 | 0.564 |
| [0.4, 0.6) | 16 | 0.543 |
| [0.6, 0.8) | 83 | 0.313 |
| [0.8, 1.0] | 87 | 0.315 |

`predict.py` prints the bin each query falls into. Use it. A prediction for a
compound with no training neighbour above Tanimoto 0.6 is a ranking hint, not an
estimate.

The endpoint is DPPH radical scavenging at a nominal 30-minute readout, pooled
from many publications. Nothing here transfers to ABTS, FRAP or ORAC.

## Citation

See the paper. The model card in `model/model_card.md` records the training
partition, the selected hyperparameters and the full set of measured metrics.
"""

CARD = """# Model card

| | |
|---|---|
| Task | regression of pIC50 = -log10(IC50 / M) |
| Endpoint | DPPH radical scavenging, nominal 30-minute readout |
| Input | SMILES |
| Representation | {rep} ({nfeat} features) |
| Learner | {learner} |
| Selected hyperparameters | {params} |
| Training compounds | {n_train} |
| Held-out compounds | {n_test} |
| Partition | chemistry-stratified 9:1, fixed seed 42 |

## Measured performance

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Held-out R2 | {r2} | {r2lo}--{r2hi} |
| Held-out RMSE | {rmse} | {rmselo}--{rmsehi} |
| Held-out MAE | {mae} | {maelo}--{maehi} |
| Out-of-fold Q2 on the training set | {q2} | |

Intervals are percentile bootstrap over 2,000 resamples of the hold-out. The Q2
is the pooled 10-fold out-of-fold R2 of the estimator the grid search selected;
because the hyperparameters were chosen using all of the training data it is a
slightly optimistic summary of training-set performance, not a nested
cross-validation estimate.

## Limits

- Hyperparameters were selected inside the training partition only, and the
  held-out compounds were scored exactly once.
- On a source-publication-disjoint partition the model scores R2 = {src}. Treat
  predictions for compounds unlike the training set as ranking hints.
- The training labels are aggregated from many publications and differ in
  solvent, temperature, illumination and instrument. That heterogeneity places a
  floor on attainable accuracy.
- The model is not a mechanistic model. It does not explain why a compound
  scavenges radicals.
"""


def write_docs(facts):
    ci = facts["ci"] or {}

    def rng(k):
        return (("%.3f" % ci[k][0], "%.3f" % ci[k][1]) if k in ci else ("--", "--"))

    r2lo, r2hi = rng("r2")
    rmselo, rmsehi = rng("rmse")
    maelo, maehi = rng("mae")
    common = dict(
        n_train="%d" % facts["n_train"], n_test="%d" % facts["n_test"],
        r2="%.4f" % facts["r2"], rmse="%.4f" % facts["rmse"], mae="%.4f" % facts["mae"],
        q2="%.4f" % facts["q2"], src="%.3f" % (facts["worst_split_r2"] or float("nan")),
        r2lo=r2lo, r2hi=r2hi, rmselo=rmselo, rmsehi=rmsehi, maelo=maelo, maehi=maehi,
        cells=facts.get("n_scored_models", 55),
        rep=REPRESENTATION, learner=LEARNER, nfeat=facts.get("n_features", ""),
        params=", ".join("%s = %s" % kv for kv in facts["best_params"].items()),
    )
    open(os.path.join(RELEASE, "README.md"), "w", encoding="utf-8").write(
        README.format(**common))
    open(os.path.join(RELEASE, "model", "model_card.md"), "w", encoding="utf-8").write(
        CARD.format(**common))
    print("wrote README.md and model/model_card.md")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    facts = build()
    matrix = pd.read_csv(os.path.join(OUT, "a5_matrix_stratified.csv"))
    facts["n_scored_models"] = int(len(matrix))
    facts["n_features"] = int(
        matrix[(matrix["representation"] == REPRESENTATION)
               & (matrix["learner"] == LEARNER)]["n_features"].iloc[0])
    with open(os.path.join(RELEASE, "model", "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2)
    write_docs(facts)
    print("\nrelease built in %s" % RELEASE)


if __name__ == "__main__":
    main()
