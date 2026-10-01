"""
Leak-free supervised fine-tuning of the chemical language models.

Every encoder is fine-tuned inside 10-fold cross-validation restricted to the
1,719 training compounds of the fixed 9:1 stratified split; the 192 held-out
compounds are never seen by any encoder, tokenizer statistic or early-stopping
decision. Three independent seeds quantify run-to-run variability
(Reviewer 5 Q4, Reviewer 4 Q4/Q7).

Outputs per model and seed:
  * out-of-fold predictions on the training compounds  -> honest Q2_CV
  * fold-ensemble predictions on the held-out compounds
  * fold-ensemble encoder embeddings for all 1,911 molecules
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")
CKPT = os.path.join(ANALYSIS, "checkpoints")
for d in (OUT, FEAT, CKPT):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "code"))

from llm_models import HFSmilesRegressor
from llm_training import make_loader, metrics, predict, set_seed, train_fold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "ChemBERTa-ZINC": os.path.join(ROOT, "hf_local", "chemberta_zinc_safetensors"),
    "ChemBERTa-MTR": os.path.join(ROOT, "hf_local", "chemberta_77m_mtr_safetensors"),
}


class Args:
    def __init__(self, model_name, seed):
        self.model_name = model_name
        self.tokenizer_name = model_name
        self.batch_size = 32
        self.max_length = 256
        self.max_epochs = 60
        self.patience = 10
        self.lr = 2e-5
        self.weight_decay = 0.01
        self.dropout = 0.15
        self.warmup_ratio = 0.08
        self.pooling = "cls"
        self.seed = seed
        self.freeze_encoder = False
        self.local_files_only = True
        self.use_amp = True
        self.verbose = False


@torch.no_grad()
def encode_all(state_dict, args, smiles):
    """Fold-encoder representation of every molecule (CLS of the final layer)."""
    model = HFSmilesRegressor(
        model_name=args.model_name,
        dropout=args.dropout,
        pooling=args.pooling,
        freeze_encoder=True,
        local_files_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    vecs = []
    for i in range(0, len(smiles), 64):
        enc = tok(
            smiles[i : i + 64],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(DEVICE)
        vecs.append(model.encoder(**enc).last_hidden_state[:, 0].float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.vstack(vecs)


def get_split(name):
    payload = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))
    s = payload["splits"][name]
    return np.array(s["train_idx"]), np.array(s["test_idx"])


def run_model(tag, path, seeds, n_folds=10, split_name="stratified"):
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        data = pickle.load(fh)
    df = data["df"]
    graphs = data["graphs"]
    smiles = df["smiles"].tolist()
    y = df["pIC50"].to_numpy(float)

    train_idx, test_idx = get_split(split_name)
    test_graphs = [graphs[i] for i in test_idx]
    suffix = "" if split_name == "stratified" else f"_{split_name}"

    seed_results = {}
    for seed in seeds:
        args = Args(path, seed)
        t_start = time.time()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof = np.zeros(len(train_idx))
        test_fold_preds, emb_folds, fold_info = [], [], []

        for fold, (tr_i, va_i) in enumerate(kf.split(train_idx)):
            tr_graphs = [graphs[i] for i in train_idx[tr_i]]
            va_graphs = [graphs[i] for i in train_idx[va_i]]
            prefix = os.path.join(CKPT, f"{tag.replace('-', '_').lower()}{suffix}_seed{seed}")
            info = train_fold(tr_graphs, va_graphs, test_graphs, fold, args, DEVICE, prefix)

            ck = torch.load(f"{prefix}_fold{fold + 1}.pt", map_location="cpu", weights_only=False)
            state = ck["model_state_dict"]

            model = HFSmilesRegressor(
                model_name=args.model_name,
                dropout=args.dropout,
                pooling=args.pooling,
                freeze_encoder=True,
                local_files_only=True,
            )
            model.load_state_dict(state)
            model.to(DEVICE).eval()
            tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
            val_pred, _ = predict(model, make_loader(va_graphs, tok, 64, False, args.max_length), DEVICE)
            oof[va_i] = val_pred
            del model
            torch.cuda.empty_cache()

            test_fold_preds.append(np.asarray(info["test_predictions"], dtype=float))
            emb_folds.append(encode_all(state, args, smiles))
            fold_info.append({"fold": fold + 1, "val_r2": info["val_r2"], "best_epoch": info["best_epoch"]})

        test_pred = np.mean(test_fold_preds, axis=0)
        emb = np.mean(emb_folds, axis=0).astype(np.float32)
        np.save(
            os.path.join(FEAT, f"ft_emb_{tag.replace('-', '_').lower()}{suffix}_seed{seed}.npy"),
            emb,
        )

        seed_results[str(seed)] = {
            "fold_val_r2": [f["val_r2"] for f in fold_info],
            "cv_r2_mean": float(np.mean([f["val_r2"] for f in fold_info])),
            "cv_r2_sd": float(np.std([f["val_r2"] for f in fold_info], ddof=1)),
            "q2_cv_pooled_oof": float(r2_score(y[train_idx], oof)),
            "oof_rmse": float(np.sqrt(mean_squared_error(y[train_idx], oof))),
            "test_ensemble": {
                "test_r2": float(r2_score(y[test_idx], test_pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y[test_idx], test_pred))),
                "test_mae": float(mean_absolute_error(y[test_idx], test_pred)),
            },
            "oof_predictions": oof.tolist(),
            "test_predictions": test_pred.tolist(),
            "folds": fold_info,
            "minutes": float((time.time() - t_start) / 60),
        }
        print(
            f"[{tag} seed {seed}] Q2_CV(OOF)={seed_results[str(seed)]['q2_cv_pooled_oof']:.4f} "
            f"test R2={seed_results[str(seed)]['test_ensemble']['test_r2']:.4f} "
            f"({seed_results[str(seed)]['minutes']:.1f} min)"
        )

    out = {
        "model": tag,
        "split": split_name,
        "checkpoint_path": path,
        "protocol": (
            f"{n_folds}-fold cross-validated supervised fine-tuning restricted to the "
            f"{len(train_idx)} training compounds of the '{split_name}' partition; the "
            f"{len(test_idx)} held-out compounds are excluded from every gradient update and "
            "from early stopping. Test predictions are the mean of the fold models."
        ),
        "hyperparameters": {
            k: v for k, v in vars(Args(path, seeds[0])).items() if not k.startswith("_")
        },
        "n_folds": n_folds,
        "seeds": seeds,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "per_seed": seed_results,
    }
    fname = f"a3_finetune_{tag.replace('-', '_').lower()}{suffix}.json"
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ChemBERTa-ZINC", "ChemBERTa-MTR"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2026])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--split", default="stratified")
    a = ap.parse_args()
    for tag in a.models:
        print(f"\n===== {tag} / {a.split} =====")
        run_model(tag, MODELS[tag], a.seeds, a.folds, a.split)
