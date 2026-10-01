"""
Representation x learner benchmark on the primary chemistry-stratified split.

Answers Reviewer 4 Q6/Q7/Q13, Reviewer 5 Q4/Q6 and Reviewer 3: every model is
tuned by grid search with 5-fold cross-validation *inside the training set
only*; the held-out compounds are touched exactly once, when the refitted best
estimator is scored. The same learner grid is applied to every representation
so that the contribution of the representation can be separated from the
contribution of the downstream algorithm.
"""

import argparse
import json
import os
import pickle
import time
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")

INNER_CV = KFold(n_splits=5, shuffle=True, random_state=42)
OOF_CV = KFold(n_splits=10, shuffle=True, random_state=42)


# --------------------------------------------------------------- learner grid
def learner_grid(name, n_features):
    if name == "Ridge":
        return (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold(0.0)),
                    ("sc", StandardScaler()),
                    ("mdl", Ridge(random_state=None)),
                ]
            ),
            {"mdl__alpha": [1.0, 10.0, 100.0, 1000.0, 10000.0]},
        )
    if name == "SVR":
        return (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold(0.0)),
                    ("sc", StandardScaler()),
                    ("mdl", SVR(kernel="rbf")),
                ]
            ),
            {"mdl__C": [1.0, 10.0, 100.0], "mdl__gamma": ["scale"], "mdl__epsilon": [0.1]},
        )
    if name == "ExtraTrees":
        return (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold(0.0)),
                    ("mdl", ExtraTreesRegressor(n_estimators=400, random_state=1, n_jobs=-1)),
                ]
            ),
            {"mdl__max_features": [0.3, 0.6]},
        )
    if name == "CatBoost":
        return (
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold(0.0)),
                    (
                        "mdl",
                        CatBoostRegressor(
                            iterations=500,
                            learning_rate=0.05,
                            loss_function="RMSE",
                            random_seed=2,
                            verbose=False,
                            allow_writing_files=False,
                            thread_count=-1,
                        ),
                    ),
                ]
            ),
            {"mdl__depth": [4, 6]},
        )
    raise ValueError(name)


def metrics(y_true, y_pred):
    return {
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def load_representations(split_name="stratified"):
    """Label-free representations only.

    Fold-averaged embeddings of fine-tuned encoders are deliberately excluded:
    for a training molecule such an embedding is produced partly by encoders
    that saw that molecule's label, so feeding it to a downstream learner would
    put label information into the training features. The fine-tuned models
    enter the benchmark as end-to-end predictors instead (see a5c_merge.py).
    """
    z = np.load(os.path.join(FEAT, "representations.npz"))
    reps = {k: z[k].astype(np.float64) for k in z.files}
    reps["MoLFormer-XL+Mordred"] = np.hstack([reps["MoLFormer-XL"], reps["Mordred"]])
    reps["ECFP4+Mordred"] = np.hstack([reps["ECFP4"], reps["Mordred"]])
    reps["ECFP4+ChemBERTa-ZINC"] = np.hstack([reps["ECFP4"], reps["ChemBERTa-ZINC"]])
    return reps


REPRESENTATION_FAMILY = {
    "ECFP4": "Fingerprint baseline",
    "ECFP6": "Fingerprint baseline",
    "ECFP4-count": "Fingerprint baseline",
    "MACCS": "Fingerprint baseline",
    "RDKit-desc": "Descriptor baseline",
    "Mordred": "Descriptor baseline",
    "ChemBERTa-ZINC": "Frozen chemical language model",
    "ChemBERTa-MTR": "Frozen chemical language model",
    "ChemBERTa-MLM": "Frozen chemical language model",
    "MoLFormer-XL": "Frozen chemical language model",
    "ChemBERTa-MTR-199": "Frozen chemical language model",
    "ChemBERTa-ZINC+Mordred": "Hybrid",
    "ChemBERTa-MTR-199+Mordred": "Hybrid",
    "MoLFormer-XL+Mordred": "Hybrid",
    "ECFP4+Mordred": "Hybrid",
    "ECFP4+ChemBERTa-ZINC": "Hybrid",
}

DEFAULT_REPS = [
    "ECFP4",
    "ECFP6",
    "ECFP4-count",
    "MACCS",
    "RDKit-desc",
    "Mordred",
    "ChemBERTa-ZINC",
    "ChemBERTa-MTR",
    "ChemBERTa-MLM",
    "MoLFormer-XL",
    "ChemBERTa-MTR-199",
    "ChemBERTa-ZINC+Mordred",
    "MoLFormer-XL+Mordred",
    "ECFP4+Mordred",
    "ECFP4+ChemBERTa-ZINC",
]
DEFAULT_LEARNERS = ["Ridge", "SVR", "ExtraTrees", "CatBoost"]

# Tree ensembles are an order of magnitude more expensive than the linear and
# kernel learners, so they are run on the representations that matter for the
# comparison rather than on every fingerprint variant.
TREE_REPS = {
    "ECFP4",
    "RDKit-desc",
    "Mordred",
    "ChemBERTa-ZINC",
    "MoLFormer-XL",
    "ECFP4+Mordred",
    "ECFP4+ChemBERTa-ZINC",
    "ChemBERTa-ZINC+Mordred",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="stratified")
    ap.add_argument("--reps", nargs="+", default=DEFAULT_REPS)
    ap.add_argument("--learners", nargs="+", default=DEFAULT_LEARNERS)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--no-ensemble",
        action="store_true",
        help="skip the matched descriptor-ensemble reproduction (only needed on the primary split)",
    )
    args = ap.parse_args()

    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        y = pickle.load(fh)["df"]["pIC50"].to_numpy(float)
    splits = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))["splits"]
    train_idx = np.array(splits[args.split]["train_idx"])
    test_idx = np.array(splits[args.split]["test_idx"])
    reps = load_representations(args.split)

    rows, predictions = [], {}
    for rep_name in args.reps:
        if rep_name not in reps:
            print(f"  (skipping missing representation {rep_name})")
            continue
        X = reps[rep_name]
        for learner in args.learners:
            if learner in ("ExtraTrees", "CatBoost") and rep_name not in TREE_REPS:
                continue
            key = f"{rep_name} | {learner}"
            est, grid = learner_grid(learner, X.shape[1])
            t0 = time.perf_counter()
            search = GridSearchCV(
                est, grid, cv=INNER_CV, scoring="r2", n_jobs=1 if learner in ("ExtraTrees", "CatBoost") else -1,
                refit=True,
            )
            search.fit(X[train_idx], y[train_idx])
            best = search.best_estimator_
            oof = cross_val_predict(
                best, X[train_idx], y[train_idx], cv=OOF_CV,
                n_jobs=1 if learner in ("ExtraTrees", "CatBoost") else -1,
            )
            pred = best.predict(X[test_idx])
            elapsed = time.perf_counter() - t0
            m = metrics(y[test_idx], pred)
            row = {
                "representation": rep_name,
                "family": REPRESENTATION_FAMILY.get(rep_name, "Other"),
                "learner": learner,
                "n_features": int(X.shape[1]),
                "best_params": {k: v for k, v in search.best_params_.items()},
                "cv_r2_inner_mean": float(search.best_score_),
                "q2_cv_oof": float(r2_score(y[train_idx], oof)),
                "oof_rmse": float(np.sqrt(mean_squared_error(y[train_idx], oof))),
                **m,
                "fit_seconds": float(elapsed),
            }
            rows.append(row)
            predictions[key] = {"test": pred.tolist(), "oof": oof.tolist()}
            print(
                f"{key:44s} Q2cv={row['q2_cv_oof']:.4f} testR2={row['test_r2']:.4f} "
                f"RMSE={row['test_rmse']:.4f} ({elapsed:.0f}s)"
            )

    # --------------------------------------------- matched descriptor ensemble
    # Re-implementation of the published Mordred descriptor-ensemble workflow,
    # retrained and evaluated on exactly the same partition as every other row.
    ens_members = {} if args.no_ensemble else {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=1000, max_features=0.5, random_state=11, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=1000, max_features=0.33, random_state=12, n_jobs=-1),
        "CatBoost": CatBoostRegressor(iterations=1200, depth=5, learning_rate=0.03, random_seed=13,
                                      verbose=False, allow_writing_files=False, thread_count=-1),
        "XGBoost": XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.03, subsample=0.9,
                                colsample_bytree=0.8, reg_lambda=3.0, random_state=14, n_jobs=-1),
    }
    Xm = reps["Mordred"]
    member_pred, member_oof = [], []
    t0 = time.perf_counter()
    for name, mdl in ens_members.items():
        pipe = Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)), ("mdl", mdl)]
        )
        oof = cross_val_predict(pipe, Xm[train_idx], y[train_idx], cv=OOF_CV, n_jobs=1)
        pipe.fit(Xm[train_idx], y[train_idx])
        p = pipe.predict(Xm[test_idx])
        member_pred.append(p)
        member_oof.append(oof)
        rows.append(
            {
                "representation": "Mordred",
                "family": "Descriptor ensemble member",
                "learner": f"Ensemble-{name}",
                "n_features": int(Xm.shape[1]),
                "best_params": {},
                "cv_r2_inner_mean": float("nan"),
                "q2_cv_oof": float(r2_score(y[train_idx], oof)),
                "oof_rmse": float(np.sqrt(mean_squared_error(y[train_idx], oof))),
                **metrics(y[test_idx], p),
                "fit_seconds": float(time.perf_counter() - t0),
            }
        )
        print(f"{'Mordred | Ensemble-' + name:44s} testR2={rows[-1]['test_r2']:.4f}")

    if member_pred:
        ens_test = np.mean(member_pred, axis=0)
        ens_oof = np.mean(member_oof, axis=0)
        rows.append(
            {
                "representation": "Mordred",
                "family": "Matched descriptor ensemble",
                "learner": "Ensemble mean (reproduced reference workflow)",
                "n_features": int(Xm.shape[1]),
                "best_params": {},
                "cv_r2_inner_mean": float("nan"),
                "q2_cv_oof": float(r2_score(y[train_idx], ens_oof)),
                "oof_rmse": float(np.sqrt(mean_squared_error(y[train_idx], ens_oof))),
                **metrics(y[test_idx], ens_test),
                "fit_seconds": float(time.perf_counter() - t0),
            }
        )
        predictions["Mordred descriptor ensemble (matched)"] = {
            "test": ens_test.tolist(),
            "oof": ens_oof.tolist(),
        }
        print(f"{'Mordred | matched descriptor ensemble':44s} testR2={rows[-1]['test_r2']:.4f}")

    tag = args.out or f"a5_matrix_{args.split}"
    df = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    df.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    with open(os.path.join(OUT, f"{tag}_predictions.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "split": args.split,
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "y_test": y[test_idx].tolist(),
                "y_train": y[train_idx].tolist(),
                "predictions": predictions,
                "tuning_protocol": (
                    "GridSearchCV, 5-fold shuffled KFold inside the training set, scoring = R2, "
                    "refit on the full training set. Q2_CV is the pooled 10-fold out-of-fold R2 "
                    "of the selected estimator on the training compounds. The held-out set is "
                    "used only for the final score."
                ),
            },
            fh,
            indent=2,
        )
    print("\nWrote", os.path.join(OUT, f"{tag}.csv"))


if __name__ == "__main__":
    main()
