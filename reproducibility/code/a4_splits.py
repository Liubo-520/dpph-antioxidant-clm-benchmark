"""
Construct and characterise every train/test partition used in the revision.

Answers Reviewer 1 Q2, Reviewer 2, Reviewer 3, Reviewer 4 Q2/Q3/Q4 and
Reviewer 5 Q2: besides the original chemistry-stratified 9:1 split we add a
Bemis-Murcko scaffold split, a Butina cluster split, a dissimilarity-based
hold-out, a publication-disjoint external split, and 25 repeated stratified
splits. Each partition is characterised by nearest-neighbour Tanimoto
similarity and scaffold overlap between training and test compounds.
"""

import json
import os
import pickle
import warnings
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedShuffleSplit

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
os.makedirs(OUT, exist_ok=True)

TEST_FRACTION = 0.10
PRIMARY_SEED = 42
N_REPEATS = 25


def load():
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        data = pickle.load(fh)
    curated = pd.read_csv(os.path.join(OUT, "curated_dataset.csv"))
    return data["df"]["smiles"].tolist(), data["df"]["pIC50"].to_numpy(float), curated


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


# ---------------------------------------------------------------- split rules
def stratified_split(categories, seed, test_fraction=TEST_FRACTION):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    return next(sss.split(np.zeros(len(categories)), categories))


def group_disjoint_split(groups, y, seed, test_fraction=TEST_FRACTION):
    """Assign whole groups to the test set until the target size is reached.

    Groups are visited in a seeded random order among the small groups first so
    that large congeneric families stay in training, which is the standard
    scaffold-split convention.
    """
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i, g in enumerate(groups):
        buckets[g].append(i)
    order = sorted(buckets.items(), key=lambda kv: (len(kv[1]), rng.random()))
    target = int(round(test_fraction * len(groups)))
    test, used = [], set()
    for name, idx in order:
        if len(test) + len(idx) > target and test:
            continue
        test.extend(idx)
        used.add(name)
        if len(test) >= target:
            break
    test = np.array(sorted(test))
    train = np.array(sorted(set(range(len(groups))) - set(test.tolist())))
    return train, test


def butina_clusters(fps, cutoff=0.4):
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


def dissimilarity_split(fps, test_fraction=TEST_FRACTION):
    """Hold out the compounds that are least similar to the rest of the library."""
    n = len(fps)
    max_sim = np.zeros(n)
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        sims[i] = -1.0
        max_sim[i] = sims.max()
    order = np.argsort(max_sim)  # least similar first
    k = int(round(test_fraction * n))
    test = np.array(sorted(order[:k]))
    train = np.array(sorted(set(range(n)) - set(test.tolist())))
    return train, test, max_sim


# ------------------------------------------------------------ characterisation
def nn_similarity(fps, train, test):
    out = np.zeros(len(test))
    train_fps = [fps[i] for i in train]
    for j, t in enumerate(test):
        out[j] = max(DataStructs.BulkTanimotoSimilarity(fps[t], train_fps))
    return out


def characterise(fps, scaffolds, categories, y, train, test, name):
    nn = nn_similarity(fps, train, test)
    train_sc = set(np.asarray(scaffolds)[train])
    test_sc = np.asarray(scaffolds)[test]
    shared = np.array([s in train_sc for s in test_sc])
    return {
        "split": name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_nn_tanimoto_mean": float(nn.mean()),
        "test_nn_tanimoto_median": float(np.median(nn)),
        "test_nn_tanimoto_p10": float(np.percentile(nn, 10)),
        "test_nn_tanimoto_p90": float(np.percentile(nn, 90)),
        "pct_test_with_nn_sim_ge_0.7": float(100 * (nn >= 0.7).mean()),
        "pct_test_with_nn_sim_ge_0.4": float(100 * (nn >= 0.4).mean()),
        "pct_test_scaffold_seen_in_training": float(100 * shared.mean()),
        "n_test_unique_scaffolds": int(len(set(test_sc.tolist()))),
        "test_mean_pIC50": float(y[test].mean()),
        "test_sd_pIC50": float(y[test].std(ddof=1)),
        "train_mean_pIC50": float(y[train].mean()),
        "test_category_counts": {k: int(v) for k, v in Counter(np.asarray(categories)[test]).items()},
        "test_nn_tanimoto": nn.tolist(),
    }


def main():
    smiles, y, curated = load()
    categories = curated["chemistry_category"].to_numpy()
    scaffolds = curated["bemis_murcko_scaffold"].to_numpy()
    sources = curated["source_doi"].to_numpy()
    fps = fingerprints(smiles)

    splits, summary = {}, []

    # 1. primary chemistry-stratified split, reproduced from the first submission
    original = json.load(open(os.path.join(ROOT, "results", "stratified_split_llm_results.json"), encoding="utf-8"))
    tr, te = stratified_split(categories, PRIMARY_SEED)
    assert sorted(te.tolist()) == sorted(original["test_idx"]), "primary split no longer reproduces"
    splits["stratified"] = (tr, te)

    # 2. Bemis-Murcko scaffold split
    splits["scaffold"] = group_disjoint_split(scaffolds, y, PRIMARY_SEED)[:2]

    # 3. Butina cluster split on ECFP4 Tanimoto distance
    labels = butina_clusters(fps, cutoff=0.4)
    splits["cluster"] = group_disjoint_split(labels, y, PRIMARY_SEED)[:2]

    # 4. dissimilarity hold-out: the 10% most structurally isolated molecules
    d_tr, d_te, max_sim = dissimilarity_split(fps)
    splits["dissimilarity"] = (d_tr, d_te)

    # 5. publication-disjoint external split
    splits["source"] = group_disjoint_split(sources, y, PRIMARY_SEED)[:2]

    for name, (a, b) in splits.items():
        summary.append(characterise(fps, scaffolds, categories, y, a, b, name))
        print(
            f"{name:15s} train={len(a):5d} test={len(b):4d} "
            f"NN-Tanimoto median={summary[-1]['test_nn_tanimoto_median']:.3f} "
            f"scaffold seen={summary[-1]['pct_test_scaffold_seen_in_training']:.1f}%"
        )

    # 6. repeated stratified splits for split-to-split variability
    repeats = []
    for r in range(N_REPEATS):
        a, b = stratified_split(categories, 1000 + r)
        repeats.append({"seed": 1000 + r, "train_idx": a.tolist(), "test_idx": b.tolist()})

    n_clusters = len(set(labels.tolist()))
    payload = {
        "n_molecules": len(smiles),
        "test_fraction": TEST_FRACTION,
        "primary_seed": PRIMARY_SEED,
        "cluster_algorithm": "Butina, ECFP4 (r=2, 2048 bit) Tanimoto distance, cutoff 0.4",
        "n_butina_clusters": int(n_clusters),
        "n_source_publications": int(len(set(sources.tolist()))),
        "dissimilarity_rule": "hold out the 10% of molecules with the lowest maximum Tanimoto similarity to any other molecule",
        "max_intra_library_similarity": {
            "median": float(np.median(max_sim)),
            "p10": float(np.percentile(max_sim, 10)),
        },
        "splits": {k: {"train_idx": v[0].tolist(), "test_idx": v[1].tolist()} for k, v in splits.items()},
        "repeated_stratified": repeats,
        "characterisation": summary,
    }
    with open(os.path.join(OUT, "a4_splits.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    rows = []
    for s in summary:
        rows.append({k: v for k, v in s.items() if k not in ("test_nn_tanimoto", "test_category_counts")})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "a4_split_characterisation.csv"), index=False)
    print("\nWrote", os.path.join(OUT, "a4_splits.json"))


if __name__ == "__main__":
    main()
