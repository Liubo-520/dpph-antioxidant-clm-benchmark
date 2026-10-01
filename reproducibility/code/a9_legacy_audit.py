"""
Audit of the feature matrix behind the originally submitted headline result.

Reviewer 2 and Reviewer 5 (comment 2) questioned whether the reported hold-out
performance reflects genuine generalisation. Re-deriving the archived feature
matrix showed that it cannot be reproduced by any label-free extraction from the
pretrained checkpoint, so this script quantifies where the archived features
came from and what their provenance implies for the reported metric.

Three diagnostics are run:
  1. agreement between the archived matrix and candidate extraction protocols;
  2. how much activity information a linear probe can read out of the archived
     matrix compared with label-free representations;
  3. the decisive test - hold-out performance split by whether a nominally
     held-out molecule had contributed to the supervised training of the
     encoders that produced the archived features.
"""

import json
import os
import pickle
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")
sys.path.insert(0, os.path.join(ROOT, "code"))
from llm_models import HFSmilesRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT, "hf_local", "chemberta_zinc_safetensors")


def ridge_pipe(alpha=100.0):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("sc", StandardScaler()),
            ("mdl", Ridge(alpha=alpha)),
        ]
    )


@torch.no_grad()
def fold_cls(smiles, fold, max_length=256):
    path = os.path.join(ROOT, "results", "models", f"random_llm_chemberta_zinc_fold{fold}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = HFSmilesRegressor(
        model_name=MODEL_PATH,
        dropout=ck["dropout"],
        pooling=ck["pooling"],
        freeze_encoder=True,
        local_files_only=True,
    )
    m.load_state_dict(ck["model_state_dict"])
    m.to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    out = []
    for i in range(0, len(smiles), 64):
        enc = tok(
            smiles[i : i + 64], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(DEVICE)
        out.append(m.encoder(**enc).last_hidden_state[:, 0].float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.vstack(out)


def main():
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        data = pickle.load(fh)
    smiles = data["df"]["smiles"].tolist()
    y = data["df"]["pIC50"].to_numpy(float)
    ft_train = np.asarray(data["random_split"]["train_idx"])

    strat = json.load(open(os.path.join(ROOT, "results", "stratified_split_llm_results.json"), encoding="utf-8"))
    tr = np.asarray(strat["train_idx"])
    te = np.asarray(strat["test_idx"])

    legacy = np.load(os.path.join(ROOT, "results", "chemberta_avg_embeddings.npy")).astype(float)
    report = {}

    # ------------------------------------------------------ 1. provenance match
    folds = [fold_cls(smiles, f) for f in range(1, 11)]
    candidates = {
        "mean CLS over the ten supervised fold encoders": np.mean(folds, axis=0),
        "fold-1 supervised encoder CLS": folds[0],
    }
    reps = np.load(os.path.join(FEAT, "representations.npz"))
    candidates["frozen pretrained encoder, mean pooling"] = reps["ChemBERTa-ZINC"].astype(float)

    match = {}
    for name, cand in candidates.items():
        per_mol = np.array([np.corrcoef(legacy[i], cand[i])[0, 1] for i in range(len(smiles))])
        match[name] = {
            "median_per_molecule_pearson_r": float(np.median(per_mol)),
            "mean_column_pearson_r": float(
                np.nanmean(
                    [
                        np.corrcoef(legacy[:, j], cand[:, j])[0, 1]
                        for j in range(min(legacy.shape[1], cand.shape[1]))
                    ]
                )
            ),
            "max_abs_difference": float(np.abs(legacy - cand).max()),
        }
    report["provenance_match"] = match

    # -------------------------------------------- 2. how much label information
    probe = {}
    for name, X in [
        ("archived feature matrix used for the submitted result", legacy),
        ("frozen pretrained encoder, mean pooling", reps["ChemBERTa-ZINC"].astype(float)),
        ("Mordred descriptors", reps["Mordred"].astype(float)),
        ("ECFP4 fingerprints", reps["ECFP4"].astype(float)),
    ]:
        p = ridge_pipe()
        p.fit(X[tr], y[tr])
        pred = p.predict(X[te])
        probe[name] = {
            "test_r2": float(r2_score(y[te], pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
            "test_mae": float(mean_absolute_error(y[te], pred)),
        }
    report["ridge_probe_on_stratified_split"] = probe

    # ---------------------------------------------------- 3. the decisive test
    seen = np.isin(te, ft_train)

    def split_errors(X, alpha=100.0):
        q = ridge_pipe(alpha)
        q.fit(X[tr], y[tr])
        pc = q.predict(X[te])
        e_seen = np.abs(y[te][seen] - pc[seen])
        e_unseen = np.abs(y[te][~seen] - pc[~seen])
        u = stats.mannwhitneyu(e_seen, e_unseen, alternative="less")
        return {
            "rmse_seen": float(np.sqrt(np.mean(e_seen**2))),
            "rmse_unseen": float(np.sqrt(np.mean(e_unseen**2))),
            "rmse_ratio_unseen_over_seen": float(
                np.sqrt(np.mean(e_unseen**2)) / np.sqrt(np.mean(e_seen**2))
            ),
            "median_abs_error_seen": float(np.median(e_seen)),
            "median_abs_error_unseen": float(np.median(e_unseen)),
            "mannwhitney_p_seen_better": float(u.pvalue),
            "r2_seen": float(r2_score(y[te][seen], pc[seen])),
            "r2_unseen": float(r2_score(y[te][~seen], pc[~seen])),
        }

    report["contamination_test"] = {
        "n_test": int(len(te)),
        "n_test_seen_by_supervised_encoders": int(seen.sum()),
        "n_test_never_seen": int((~seen).sum()),
        "sd_pIC50_seen_subset": float(y[te][seen].std(ddof=1)),
        "sd_pIC50_unseen_subset": float(y[te][~seen].std(ddof=1)),
        "note_on_r2": (
            "R2 is compared with caution because the 30 never-seen molecules have a different "
            "activity spread; RMSE and the RMSE ratio are the primary comparison."
        ),
        "archived_features": split_errors(legacy),
        "label_free_controls": {
            "Mordred descriptors": split_errors(reps["Mordred"].astype(float)),
            "frozen pretrained encoder": split_errors(reps["ChemBERTa-ZINC"].astype(float)),
            "ECFP4 fingerprints": split_errors(reps["ECFP4"].astype(float)),
        },
    }
    report["conclusion"] = (
        "The archived feature matrix matches the mean CLS representation of the ten encoders "
        "that had been supervised on DPPH labels under an earlier 80:20 random partition, and "
        "does not match any label-free extraction from the same checkpoint. Because that earlier "
        "partition overlaps the chemistry-stratified hold-out, "
        f"{int(np.isin(te, ft_train).sum())} of the 192 nominally held-out molecules had already "
        "contributed to encoder training. Splitting the hold-out accordingly reproduces the "
        "signature of information leakage: the archived features predict the previously seen "
        "molecules far better than the unseen ones, whereas label-free representations show no "
        "such gap. All results in the revised manuscript are therefore recomputed with encoders "
        "that are fitted inside the training partition only."
    )

    with open(os.path.join(OUT, "a9_legacy_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
