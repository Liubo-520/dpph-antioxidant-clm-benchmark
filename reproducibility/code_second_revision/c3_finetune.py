"""Leak-free fine-tuned ChemBERTa-ZINC on the repeated partitions.

Identical protocol to a3_finetune_clean.py of the first revision (same
hyperparameters, five folds nested inside the training partition, early
stopping on the inner validation fold only, held-out prediction as the mean of
the five fold models, seed 42). The only differences are bookkeeping: the
partition comes from c1_partitions.json, fold checkpoints are deleted once
their predictions are stored, and no fold-encoder embeddings are written,
because nothing downstream uses them.

    python c3_finetune.py --worker 0 --of 3
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from transformers import AutoTokenizer

from common import pin  # noqa: F401
from common import DESIGNS, N_FOLDS, OUT, REP_FOLDS, ROOT, load_dataset, load_partitions, r1_import, write_json

r1_import()
from llm_models import HFSmilesRegressor                      # noqa: E402
from llm_training import make_loader, predict, train_fold     # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT, "hf_local", "chemberta_zinc_safetensors")
RESULTS = os.path.join(OUT, "finetune")
TMP = os.path.join(OUT, "finetune", "_ckpt_tmp")
os.makedirs(TMP, exist_ok=True)

SEED = 42
INNER_FOLDS = 5


class Args:
    """The fine-tuning hyperparameters of a3_finetune_clean.py, unchanged."""

    def __init__(self):
        self.model_name = MODEL_PATH
        self.tokenizer_name = MODEL_PATH
        self.batch_size = 32
        self.max_length = 256
        self.max_epochs = 60
        self.patience = 10
        self.lr = 2e-5
        self.weight_decay = 0.01
        self.dropout = 0.15
        self.warmup_ratio = 0.08
        self.pooling = "cls"
        self.seed = SEED
        self.freeze_encoder = False
        self.local_files_only = True
        self.use_amp = True
        self.verbose = False


def run_partition(pid, part, graphs, y):
    out_path = os.path.join(RESULTS, f"{pid}.json")
    if os.path.exists(out_path):
        return
    train_idx = np.asarray(part["train_idx"])
    test_idx = np.asarray(part["test_idx"])
    test_graphs = [graphs[i] for i in test_idx]
    args = Args()
    t0 = time.time()

    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_idx))
    fold_test, fold_info = [], []
    for fold, (tr_i, va_i) in enumerate(kf.split(train_idx)):
        tr_graphs = [graphs[i] for i in train_idx[tr_i]]
        va_graphs = [graphs[i] for i in train_idx[va_i]]
        prefix = os.path.join(TMP, f"{pid}_seed{SEED}")
        info = train_fold(tr_graphs, va_graphs, test_graphs, fold, args, DEVICE, prefix)

        ckpt = f"{prefix}_fold{fold + 1}.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        model = HFSmilesRegressor(model_name=MODEL_PATH, dropout=args.dropout, pooling=args.pooling,
                                  freeze_encoder=True, local_files_only=True)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        val_pred, _ = predict(model, make_loader(va_graphs, tok, 64, False, args.max_length), DEVICE)
        oof[va_i] = val_pred
        del model, state
        torch.cuda.empty_cache()
        os.remove(ckpt)

        fold_test.append(np.asarray(info["test_predictions"], dtype=float))
        fold_info.append({"fold": fold + 1, "val_r2": info["val_r2"], "best_epoch": info["best_epoch"]})

    test_pred = np.mean(fold_test, axis=0)
    result = {
        "partition": pid,
        "model": "ChemBERTa-ZINC (fine-tuned), fold ensemble",
        "seed": SEED,
        "inner_folds": INNER_FOLDS,
        "fold_val_r2": [f["val_r2"] for f in fold_info],
        "best_epochs": [f["best_epoch"] for f in fold_info],
        "cv_r2_inner_mean": float(np.mean([f["val_r2"] for f in fold_info])),
        "q2_cv_oof": float(r2_score(y[train_idx], oof)),
        "test_r2": float(r2_score(y[test_idx], test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y[test_idx], test_pred))),
        "test_mae": float(mean_absolute_error(y[test_idx], test_pred)),
        "oof_predictions": oof.tolist(),
        "test_predictions": test_pred.tolist(),
        "minutes": float((time.time() - t0) / 60),
    }
    write_json(out_path, result)
    print(f"[{pid}] fine-tuned CLM: inner R2={result['cv_r2_inner_mean']:.4f} "
          f"held-out R2={result['test_r2']:.4f} ({result['minutes']:.1f} min)", flush=True)


def main():
    pin("gpu")
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS))
    ap.add_argument("--folds", nargs="+", type=int, default=list(REP_FOLDS))
    ap.add_argument("--partitions", nargs="+", default=None,
                    help="explicit partition ids; overrides --designs and --folds")
    ap.add_argument("--reverse", action="store_true",
                    help="take the queue from the end, to help a worker already running")
    a = ap.parse_args()

    _, y, _, data = load_dataset()
    graphs = data["graphs"]
    parts = load_partitions()["partitions"]
    # fold-major order, so every design advances together
    queue = a.partitions or [f"{d}_f{k:02d}" for k in a.folds for d in a.designs]
    mine = queue[a.worker::a.of]
    if a.reverse:
        mine = mine[::-1]
    print(f"worker {a.worker}/{a.of}: {len(mine)} partitions on {DEVICE}", flush=True)
    for pid in mine:
        try:
            run_partition(pid, parts[pid], graphs, y)
        except Exception as exc:                      # keep the queue moving
            print(f"[{pid}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
