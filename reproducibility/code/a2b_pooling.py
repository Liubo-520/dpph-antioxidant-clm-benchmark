"""Sensitivity of the frozen ChemBERTa-ZINC representation to how it is extracted.

Reviewer 4 Q14 and Reviewer 5 Q4 ask which layer and which pooling strategy were
used, on the grounds that different choices give substantially different
results. This script measures how large that effect actually is on this dataset,
so that the manuscript can state it rather than assert that the protocol was
specified.
"""

import json
import os
import pickle
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
CKPT = os.path.join(ROOT, "hf_local", "chemberta_zinc_safetensors")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256

STRATEGIES = [
    ("Final layer, mean over non-padding tokens", -1, "mean"),
    ("Final layer, [CLS] token", -1, "cls"),
    ("Second-to-last layer, mean over non-padding tokens", -2, "mean"),
    ("Second-to-last layer, [CLS] token", -2, "cls"),
    ("Mean of the final four layers, mean pooling", "last4", "mean"),
]


@torch.no_grad()
def extract(smiles, layer, pooling):
    tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
    backbone = AutoModelForMaskedLM.from_pretrained(CKPT, local_files_only=True)
    model = getattr(backbone, backbone.base_model_prefix).to(DEVICE).eval()
    vecs = []
    for i in range(0, len(smiles), 64):
        enc = tok(
            smiles[i : i + 64],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        out = model(**enc, output_hidden_states=True)
        if layer == "last4":
            h = torch.stack(out.hidden_states[-4:]).mean(0)
        else:
            h = out.hidden_states[layer]
        if pooling == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        vecs.append(v.float().cpu().numpy())
    del model, backbone
    torch.cuda.empty_cache()
    return np.vstack(vecs)


def main():
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        df = pickle.load(fh)["df"]
    smiles = df["smiles"].tolist()
    y = df["pIC50"].to_numpy(float)
    splits = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))["splits"]
    tr = np.array(splits["stratified"]["train_idx"])
    te = np.array(splits["stratified"]["test_idx"])

    rows = []
    for label, layer, pooling in STRATEGIES:
        X = extract(smiles, layer, pooling).astype(np.float64)
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("mdl", Ridge()),
            ]
        )
        search = GridSearchCV(
            pipe,
            {"mdl__alpha": [1.0, 10.0, 100.0, 1000.0, 10000.0]},
            cv=KFold(5, shuffle=True, random_state=42),
            scoring="r2",
            n_jobs=-1,
        )
        search.fit(X[tr], y[tr])
        best = search.best_estimator_
        oof = cross_val_predict(best, X[tr], y[tr], cv=KFold(10, shuffle=True, random_state=42), n_jobs=-1)
        pred = best.predict(X[te])
        rows.append(
            {
                "strategy": label,
                "layer": str(layer),
                "pooling": pooling,
                "n_features": int(X.shape[1]),
                "alpha": float(search.best_params_["mdl__alpha"]),
                "q2_cv_oof": float(r2_score(y[tr], oof)),
                "test_r2": float(r2_score(y[te], pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
            }
        )
        print(f"{label:52s} Q2cv={rows[-1]['q2_cv_oof']:.4f} testR2={rows[-1]['test_r2']:.4f}")

    payload = {
        "model": "ChemBERTa-ZINC, frozen",
        "learner": "Ridge with alpha selected by five-fold cross-validation inside the training set",
        "split": "stratified",
        "strategies": rows,
        "spread_test_r2": float(max(r["test_r2"] for r in rows) - min(r["test_r2"] for r in rows)),
    }
    with open(os.path.join(OUT, "a2b_pooling.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nspread in held-out R2 across extraction strategies:", payload["spread_test_r2"])


if __name__ == "__main__":
    main()
