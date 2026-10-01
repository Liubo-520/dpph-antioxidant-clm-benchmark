"""Assemble the benchmark table for one partition.

Three sources are joined. The representation x learner grid comes from
a5_matrix.py, which may be run in several passes. The end-to-end fine-tuned
chemical language models come from a3_finetune_clean.py, and the graph network
from a5b_gnn.py; both are added here rather than as representations, because a
fold-averaged encoder embedding of a *training* molecule carries that molecule's
own label and must not be fed to a downstream learner. The end-to-end fold
ensembles have no such problem: their held-out predictions come only from models
that never saw those compounds, and their training predictions are strictly
out-of-fold.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")

FT_MODELS = {
    "chemberta_zinc": "ChemBERTa-ZINC (fine-tuned)",
    "chemberta_mtr": "ChemBERTa-MTR (fine-tuned)",
}


def finetune_rows(split, y_train, y_test):
    """One row per fine-tuned model: the fold ensemble, plus the seed mean."""
    suffix = "" if split == "stratified" else f"_{split}"
    rows, preds = [], {}
    for tag, label in FT_MODELS.items():
        path = os.path.join(OUT, f"a3_finetune_{tag}{suffix}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        per = d["per_seed"]
        test_stack = [np.asarray(v["test_predictions"], float) for v in per.values()]
        oof_stack = [np.asarray(v["oof_predictions"], float) for v in per.values()]
        variants = [(f"{label}, fold ensemble", np.mean(test_stack, axis=0), np.mean(oof_stack, axis=0))]
        if len(per) > 1:
            single = next(iter(per.values()))
            variants.append(
                (
                    f"{label}, fold ensemble (single seed)",
                    np.asarray(single["test_predictions"], float),
                    np.asarray(single["oof_predictions"], float),
                )
            )
            variants[0] = (f"{label}, fold ensemble ({len(per)}-seed mean)", *variants[0][1:])
        for name, pt, po in variants:
            rows.append(
                {
                    "representation": label,
                    "family": "Fine-tuned chemical language model",
                    "learner": name.split(", ", 1)[1],
                    "n_features": int(d["hyperparameters"].get("max_length", 0)),
                    "best_params": {"protocol": f"{d['n_folds']}-fold nested fine-tuning"},
                    "cv_r2_inner_mean": float(np.mean([v["cv_r2_mean"] for v in per.values()])),
                    "q2_cv_oof": float(r2_score(y_train, po)),
                    "oof_rmse": float(np.sqrt(mean_squared_error(y_train, po))),
                    "test_r2": float(r2_score(y_test, pt)),
                    "test_rmse": float(np.sqrt(mean_squared_error(y_test, pt))),
                    "test_mae": float(mean_absolute_error(y_test, pt)),
                    "fit_seconds": float(60 * sum(v["minutes"] for v in per.values())),
                }
            )
            preds[f"{label} | {rows[-1]['learner']}"] = {"test": pt.tolist(), "oof": po.tolist()}
    return rows, preds


def gnn_rows(split, y_train, y_test):
    """The graph network is also an end-to-end predictor, so it is added here."""
    suffix = "" if split == "stratified" else f"_{split}"
    path = os.path.join(OUT, f"a5b_gnn{suffix}.json")
    if not os.path.exists(path):
        return [], {}
    d = json.load(open(path, encoding="utf-8"))
    pt = np.asarray(d["test_predictions"], float)
    po = np.asarray(d["oof_predictions"], float)
    row = {
        "representation": "Molecular graph",
        "family": "Graph neural network",
        "learner": "AttentiveFP, fold ensemble",
        "n_features": 0,
        "best_params": {"protocol": f"{len(d['fold_val_r2'])}-fold nested training with early stopping"},
        "cv_r2_inner_mean": float(np.mean(d["fold_val_r2"])),
        "q2_cv_oof": float(r2_score(y_train, po)),
        "oof_rmse": float(np.sqrt(mean_squared_error(y_train, po))),
        "test_r2": float(r2_score(y_test, pt)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pt))),
        "test_mae": float(mean_absolute_error(y_test, pt)),
        "fit_seconds": float(60 * d["minutes"]),
    }
    return [row], {"Molecular graph | AttentiveFP, fold ensemble": {"test": pt.tolist(), "oof": po.tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="stratified")
    ap.add_argument("--parts", nargs="+", default=["partA", "partB"])
    a = ap.parse_args()

    frames, preds, meta = [], {}, None
    for part in a.parts:
        tag = f"a5_matrix_{a.split}_{part}"
        csv = os.path.join(OUT, f"{tag}.csv")
        if not os.path.exists(csv):
            print(f"  (missing {tag}, skipped)")
            continue
        frames.append(pd.read_csv(csv))
        payload = json.load(open(os.path.join(OUT, f"{tag}_predictions.json"), encoding="utf-8"))
        preds.update(payload["predictions"])
        meta = payload
    if meta is None:
        raise SystemExit(f"no benchmark parts found for split '{a.split}'")

    # a5_matrix.py stores the reproduced reference workflow under a display
    # name of its own; the benchmark row calls it "<representation> | <learner>"
    # like every other row. Align them, or the matched comparison the reviewers
    # asked about has a table row with no predictions behind it and drops out of
    # the paired significance tests.
    legacy = "Mordred descriptor ensemble (matched)"
    aligned = "Mordred | Ensemble mean (reproduced reference workflow)"
    if legacy in preds and aligned not in preds:
        preds[aligned] = preds.pop(legacy)

    y_train = np.asarray(meta["y_train"], float)
    y_test = np.asarray(meta["y_test"], float)
    ft_rows, ft_preds = finetune_rows(a.split, y_train, y_test)
    g_rows, g_preds = gnn_rows(a.split, y_train, y_test)
    preds.update(ft_preds)
    preds.update(g_preds)

    extra = [pd.DataFrame(r) for r in (ft_rows, g_rows) if r]
    df = pd.concat(frames + extra, ignore_index=True)
    df["key"] = df["representation"] + " | " + df["learner"]
    df = df.drop_duplicates("key", keep="last").drop(columns="key")
    df = df.sort_values("test_r2", ascending=False)
    df.to_csv(os.path.join(OUT, f"a5_matrix_{a.split}.csv"), index=False)

    meta["predictions"] = preds
    with open(os.path.join(OUT, f"a5_matrix_{a.split}_predictions.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"merged {len(df)} rows and {len(preds)} prediction vectors for split '{a.split}'")
    cols = ["representation", "learner", "family", "q2_cv_oof", "test_r2", "test_rmse"]
    print(df.head(16)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
