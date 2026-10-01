"""
Graph neural network baseline (AttentiveFP) under the identical protocol.

Answers Reviewer 4 Q13 and Reviewer 5 Q6, which ask for a comparison against
learned graph representations rather than descriptors alone. The network is
trained with the same fold-nested cross-validation inside the training compounds of
each partition as the fine-tuned language models, with early stopping on the
inner validation fold only. The held-out compounds are scored once, by the mean
of the fold models.
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
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    preds, targets, total = [], [], 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y.view(-1))
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss) * batch.num_graphs
        preds.append(out.detach().cpu().numpy())
        targets.append(batch.y.view(-1).detach().cpu().numpy())
    return total / len(loader.dataset), np.concatenate(preds), np.concatenate(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="stratified")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        data = pickle.load(fh)
    graphs = data["graphs"]
    y = data["df"]["pIC50"].to_numpy(float)
    splits = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))["splits"]
    train_idx = np.array(splits[args.split]["train_idx"])
    test_idx = np.array(splits[args.split]["test_idx"])
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=128, shuffle=False)

    torch.manual_seed(args.seed)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train_idx))
    fold_test, fold_val_r2 = [], []
    t0 = time.time()

    for fold, (tr_i, va_i) in enumerate(kf.split(train_idx), start=1):
        tr_loader = DataLoader([graphs[i] for i in train_idx[tr_i]], batch_size=64, shuffle=True)
        va_loader = DataLoader([graphs[i] for i in train_idx[va_i]], batch_size=128, shuffle=False)
        model = AttentiveFP(
            in_channels=graphs[0].x.shape[1],
            hidden_channels=200,
            out_channels=1,
            edge_dim=graphs[0].edge_attr.shape[1],
            num_layers=3,
            num_timesteps=2,
            dropout=0.15,
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best, best_state, wait = -np.inf, None, 0
        for _ in range(args.epochs):
            run_epoch(model, tr_loader, opt)
            _, vp, vt = run_epoch(model, va_loader)
            r2 = r2_score(vt, vp)
            if r2 > best:
                best, wait = r2, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.patience:
                    break
        model.load_state_dict(best_state)
        _, vp, _ = run_epoch(model, va_loader)
        oof[va_i] = vp
        _, tp, _ = run_epoch(model, test_loader)
        fold_test.append(tp)
        fold_val_r2.append(float(best))
        print(f"  AttentiveFP fold {fold:2d}: val R2={best:.4f}")

    test_pred = np.mean(fold_test, axis=0)
    result = {
        "model": "AttentiveFP graph neural network",
        "split": args.split,
        "architecture": {
            "hidden_channels": 200,
            "num_layers": 3,
            "num_timesteps": 2,
            "dropout": 0.15,
            "optimiser": "AdamW, lr 1e-3, weight decay 1e-5",
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
        },
        "fold_val_r2": fold_val_r2,
        "q2_cv_oof": float(r2_score(y[train_idx], oof)),
        "test_r2": float(r2_score(y[test_idx], test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y[test_idx], test_pred))),
        "test_mae": float(mean_absolute_error(y[test_idx], test_pred)),
        "oof_predictions": oof.tolist(),
        "test_predictions": test_pred.tolist(),
        "minutes": float((time.time() - t0) / 60),
    }
    suffix = "" if args.split == "stratified" else f"_{args.split}"
    with open(os.path.join(OUT, f"a5b_gnn{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(
        f"AttentiveFP [{args.split}] Q2cv={result['q2_cv_oof']:.4f} "
        f"test R2={result['test_r2']:.4f} ({result['minutes']:.1f} min)"
    )


if __name__ == "__main__":
    main()
