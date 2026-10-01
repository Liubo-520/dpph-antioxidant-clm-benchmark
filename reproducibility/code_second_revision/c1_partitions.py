"""Repeated group-aware 9:1 partitions (Reviewer 2, second round, point 2).

The first revision reported one 9:1 partition per design. A bootstrap over the
held-out compounds of a single partition measures sampling noise within that
hold-out; it says nothing about how much the result depends on *which*
scaffolds, clusters or publications happened to be held out. This script builds
ten 9:1 partitions for each repeatable design so that the assignment itself is
resampled:

  stratified  ten-fold split stratified on the chemistry-aware category
  scaffold    ten-fold group split on Bemis-Murcko scaffold
  cluster     ten-fold group split on Butina cluster (ECFP4, Tanimoto 0.6)
  source      ten-fold group split on source publication (DOI)

In the three group designs every group is assigned whole to one fold, so each
fold is a group-disjoint hold-out of about 10% of the compounds, and every
compound is held out exactly once per design. Groups are assigned at random
irrespective of size (largest first so that the folds come out equal in size,
ties and fold choice by a seeded generator). This differs deliberately from the
primary group-disjoint partitions, which fill the hold-out with the smallest
groups: under that rule the scaffold hold-out is drawn only from singleton
scaffolds and 128 of the 188 source-disjoint compounds are the same for every
seed, so reseeding it would not resample the assignment.

The maximum-dissimilarity partition is deterministic and is not repeated.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedKFold

from common import N_FOLDS, OUT, PARTITION_SEED, load_light, write_json

RDLogger.DisableLog("rdApp.*")


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def butina_clusters(fps, cutoff=0.4):
    """Identical to a4_splits.py: distance cutoff 0.4, i.e. Tanimoto 0.6."""
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - np.asarray(sims))
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


def balanced_group_folds(groups, k, seed):
    """Assign whole groups to k folds of near-equal size, at random.

    Groups are taken largest first, the order among equally sized groups being
    random, and each goes to the currently smallest fold, ties between folds
    broken at random. Every compound of a group therefore lands in one fold.
    """
    rng = np.random.default_rng(seed)
    uniq, inverse, counts = np.unique(np.asarray(groups).astype(str), return_inverse=True,
                                      return_counts=True)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    sizes = np.zeros(k, dtype=int)
    fold_of_group = np.empty(len(uniq), dtype=int)
    for g in order:
        smallest = np.flatnonzero(sizes == sizes.min())
        f = int(rng.choice(smallest))
        fold_of_group[g] = f
        sizes[f] += counts[g]
    return fold_of_group[inverse]


def characterise(fps, scaffolds, y, train, test):
    train_fps = [fps[i] for i in train]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[t], train_fps)) for t in test])
    train_sc = set(np.asarray(scaffolds)[train])
    seen = np.array([s in train_sc for s in np.asarray(scaffolds)[test]])
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "nn_tanimoto_mean": float(nn.mean()),
        "nn_tanimoto_median": float(np.median(nn)),
        "nn_tanimoto_p10": float(np.percentile(nn, 10)),
        "pct_nn_ge_0.4": float(100 * (nn >= 0.4).mean()),
        "pct_nn_ge_0.7": float(100 * (nn >= 0.7).mean()),
        "pct_scaffold_seen": float(100 * seen.mean()),
        "n_test_scaffolds": int(len(set(np.asarray(scaffolds)[test].tolist()))),
        "test_mean_pIC50": float(y[test].mean()),
        "test_sd_pIC50": float(y[test].std(ddof=1)),
        "train_mean_pIC50": float(y[train].mean()),
        "train_sd_pIC50": float(y[train].std(ddof=1)),
    }


def main():
    smiles, y, curated, _ = load_light()
    n = len(smiles)
    categories = curated["chemistry_category"].to_numpy()
    scaffolds = curated["bemis_murcko_scaffold"].fillna("").astype(str).to_numpy()
    sources = curated["source_doi"].fillna("NA").astype(str).to_numpy()
    fps = fingerprints(smiles)
    clusters = butina_clusters(fps, cutoff=0.4)
    print(f"{n} compounds; {len(set(scaffolds))} scaffolds, {len(set(clusters.tolist()))} Butina "
          f"clusters, {len(set(sources))} source publications")

    groupings = {"scaffold": scaffolds, "cluster": clusters, "source": sources}
    fold_labels = {}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=PARTITION_SEED)
    lab = np.empty(n, dtype=int)
    for k, (_, te) in enumerate(skf.split(np.zeros(n), categories)):
        lab[te] = k
    fold_labels["stratified"] = lab
    for offset, (name, groups) in enumerate(groupings.items(), start=1):
        fold_labels[name] = balanced_group_folds(groups, N_FOLDS, PARTITION_SEED + offset)

    partitions, rows = {}, []
    for design, lab in fold_labels.items():
        groups = groupings.get(design)
        for k in range(N_FOLDS):
            test = np.flatnonzero(lab == k)
            train = np.flatnonzero(lab != k)
            if groups is not None:
                shared = set(np.asarray(groups)[train].astype(str)) & set(np.asarray(groups)[test].astype(str))
                assert not shared, f"{design} fold {k}: {len(shared)} groups on both sides"
            pid = f"{design}_f{k:02d}"
            partitions[pid] = {"design": design, "fold": k,
                               "train_idx": train.tolist(), "test_idx": test.tolist()}
            c = characterise(fps, scaffolds, y, train, test)
            if groups is not None:
                size_of = Counter(np.asarray(groups).astype(str))
                held = [size_of[g] for g in set(np.asarray(groups)[test].astype(str))]
                c["n_test_groups"] = int(len(held))
                c["largest_test_group"] = int(max(held))
            rows.append({"partition": pid, "design": design, "fold": k, **c})
        sizes = [int((lab == k).sum()) for k in range(N_FOLDS)]
        print(f"{design:11s} fold sizes {sizes}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "c1_partition_characterisation.csv"), index=False)
    summary = table.groupby("design", sort=False).agg(
        n_test_min=("n_test", "min"), n_test_max=("n_test", "max"),
        nn_median_mean=("nn_tanimoto_median", "mean"),
        nn_median_min=("nn_tanimoto_median", "min"), nn_median_max=("nn_tanimoto_median", "max"),
        pct_nn_ge_07=("pct_nn_ge_0.7", "mean"),
        scaffold_seen_mean=("pct_scaffold_seen", "mean"),
        scaffold_seen_min=("pct_scaffold_seen", "min"), scaffold_seen_max=("pct_scaffold_seen", "max"),
        test_mean_min=("test_mean_pIC50", "min"), test_mean_max=("test_mean_pIC50", "max"),
        test_sd_min=("test_sd_pIC50", "min"), test_sd_max=("test_sd_pIC50", "max"),
    )
    print(summary.round(3).to_string())

    write_json(os.path.join(OUT, "c1_partitions.json"), {
        "n_molecules": n,
        "n_folds": N_FOLDS,
        "seed": PARTITION_SEED,
        "rule": {
            "stratified": "StratifiedKFold on the chemistry-aware category, shuffled",
            "scaffold": "whole Bemis-Murcko scaffold groups, random size-balanced assignment",
            "cluster": "whole Butina clusters (ECFP4, Tanimoto 0.6), random size-balanced assignment",
            "source": "whole source publications (DOI), random size-balanced assignment",
        },
        "n_groups": {"scaffold": int(len(set(scaffolds))),
                     "cluster": int(len(set(clusters.tolist()))),
                     "source": int(len(set(sources)))},
        "fold_of_compound": {d: lab.tolist() for d, lab in fold_labels.items()},
        "butina_cluster_of_compound": clusters.tolist(),
        "partitions": partitions,
    })
    print("wrote", os.path.join(OUT, "c1_partitions.json"))


if __name__ == "__main__":
    main()
