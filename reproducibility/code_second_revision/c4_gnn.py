"""AttentiveFP graph network on the repeated partitions.

Identical architecture, optimiser and fold-nested protocol to a5b_gnn.py of the
first revision (five folds inside the training partition, early stopping on the
inner validation fold, held-out prediction as the mean of the fold models,
seed 42); only the partition source differs.

    python c4_gnn.py --worker 0 --of 2
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

from common import pin  # noqa: F401
from common import DESIGNS, N_FOLDS, OUT, REP_FOLDS, load_dataset, load_partitions, write_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = os.path.join(OUT, "gnn")
os.makedirs(RESULTS, exist_ok=True)

SEED, INNER_FOLDS, EPOCHS, PATIENCE = 42, 5, 200, 30


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


def run_partition(pid, part, graphs, y):
    out_path = os.path.join(RESULTS, f"{pid}.json")
    if os.path.exists(out_path):
        return
    train_idx = np.asarray(part["train_idx"])
    test_idx = np.asarray(part["test_idx"])
    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=128, shuffle=False)

    torch.manual_seed(SEED)
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_idx))
    fold_test, fold_val_r2 = [], []
    t0 = time.time()
    for tr_i, va_i in kf.split(train_idx):
        tr_loader = DataLoader([graphs[i] for i in train_idx[tr_i]], batch_size=64, shuffle=True)
        va_loader = DataLoader([graphs[i] for i in train_idx[va_i]], batch_size=128, shuffle=False)
        model = AttentiveFP(
            in_channels=graphs[0].x.shape[1], hidden_channels=200, out_channels=1,
            edge_dim=graphs[0].edge_attr.shape[1], num_layers=3, num_timesteps=2, dropout=0.15,
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best, best_state, wait = -np.inf, None, 0
        for _ in range(EPOCHS):
            run_epoch(model, tr_loader, opt)
            _, vp, vt = run_epoch(model, va_loader)
            r2 = r2_score(vt, vp)
            if r2 > best:
                best, wait = r2, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        _, vp, _ = run_epoch(model, va_loader)
        oof[va_i] = vp
        _, tp, _ = run_epoch(model, test_loader)
        fold_test.append(tp)
        fold_val_r2.append(float(best))

    test_pred = np.mean(fold_test, axis=0)
    result = {
        "partition": pid,
        "model": "Molecular graph + AttentiveFP, fold ensemble",
        "seed": SEED,
        "inner_folds": INNER_FOLDS,
        "fold_val_r2": fold_val_r2,
        "cv_r2_inner_mean": float(np.mean(fold_val_r2)),
        "q2_cv_oof": float(r2_score(y[train_idx], oof)),
        "test_r2": float(r2_score(y[test_idx], test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y[test_idx], test_pred))),
        "test_mae": float(mean_absolute_error(y[test_idx], test_pred)),
        "oof_predictions": oof.tolist(),
        "test_predictions": test_pred.tolist(),
        "minutes": float((time.time() - t0) / 60),
    }
    write_json(out_path, result)
    print(f"[{pid}] AttentiveFP: inner R2={result['cv_r2_inner_mean']:.4f} "
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
    queue = a.partitions or [f"{d}_f{k:02d}" for k in a.folds for d in a.designs]
    mine = queue[a.worker::a.of]
    if a.reverse:
        mine = mine[::-1]
    print(f"worker {a.worker}/{a.of}: {len(mine)} partitions on {DEVICE}", flush=True)
    for pid in mine:
        try:
            run_partition(pid, parts[pid], graphs, y)
        except Exception as exc:
            print(f"[{pid}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
