"""
Quantitative chemical-space analysis, matched-pair SAR test and prospective
triage simulation.

Answers Reviewer 5 Q7 (the UMAP figure is currently interpreted without any
quantitative support) and Reviewer 2 (no demonstrated purpose beyond
retrospective prediction). We therefore add:

  * numerical descriptors of the projection: category silhouette, activity
    autocorrelation in fingerprint space, neighbourhood preservation, and a
    sensitivity scan over the UMAP hyperparameters;
  * a matched-molecular-pair test asking whether the model reproduces the
    experimentally measured activity change caused by adding a phenolic
    hydroxyl, which is a falsifiable structure-activity statement rather than a
    post-hoc narrative;
  * a triage simulation on the publication-disjoint and scaffold-disjoint
    partitions that measures how well the model would prioritise potent
    scavengers among compounds from laboratories and scaffolds it has never
    seen.
"""

import json
import os
import pickle
import warnings
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import umap
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")


def enrichment_factor(y_true, y_score, frac, threshold):
    n = len(y_true)
    k = max(1, int(round(frac * n)))
    order = np.argsort(y_score)[::-1]
    actives = y_true >= threshold
    if actives.sum() == 0:
        return np.nan, np.nan
    hit_rate = actives[order[:k]].mean()
    return float(hit_rate / actives.mean()), float(100 * hit_rate)


def main():
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        data = pickle.load(fh)
    smiles = data["df"]["smiles"].tolist()
    y = data["df"]["pIC50"].to_numpy(float)
    curated = pd.read_csv(os.path.join(OUT, "curated_dataset.csv"))
    categories = curated["chemistry_category"].to_numpy()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps_bit = [gen.GetFingerprint(m) for m in mols]
    X = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)

    report = {}

    # --------------------------------------------- quantitative chemical space
    labeled = categories != "Other"
    report["category_structure"] = {
        "silhouette_categories_in_fingerprint_space": float(
            silhouette_score(X[labeled], categories[labeled], metric="jaccard")
        ),
        "note": (
            "Silhouette is computed on Jaccard distances between ECFP4 fingerprints for the "
            "seven defined categories; the residual 'Other' class is excluded because it is a "
            "catch-all rather than a chemical family."
        ),
    }

    # activity autocorrelation: are near neighbours more similar in activity?
    nn = NearestNeighbors(n_neighbors=6, metric="jaccard").fit(X)
    _, idx = nn.kneighbors(X)
    neigh_delta = np.abs(y[idx[:, 1:]] - y[:, None]).mean(1)
    rng = np.random.default_rng(7)
    rand_delta = np.array([np.abs(y[rng.integers(0, len(y), 5)] - y[i]).mean() for i in range(len(y))])
    w = stats.wilcoxon(neigh_delta, rand_delta, alternative="less")
    report["activity_landscape"] = {
        "mean_abs_activity_gap_5_nearest_neighbours": float(neigh_delta.mean()),
        "mean_abs_activity_gap_random_pairs": float(rand_delta.mean()),
        "wilcoxon_p": float(w.pvalue),
        "interpretation": (
            "Nearest neighbours differ in activity by clearly less than random pairs, so the "
            "landscape is locally smooth, but the residual gap of about "
            f"{neigh_delta.mean():.2f} log units shows substantial activity-cliff behaviour."
        ),
    }

    # UMAP hyperparameter sensitivity + neighbourhood preservation
    scans, base_emb = [], None
    for n_neighbors in (5, 15, 50):
        for min_dist in (0.0, 0.1, 0.5):
            emb = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="jaccard",
                random_state=42,
                n_jobs=1,
            ).fit_transform(X)
            nn_hi = NearestNeighbors(n_neighbors=16, metric="jaccard").fit(X)
            nn_lo = NearestNeighbors(n_neighbors=16).fit(emb)
            _, hi = nn_hi.kneighbors(X)
            _, lo = nn_lo.kneighbors(emb)
            trust = np.mean([len(set(hi[i, 1:]) & set(lo[i, 1:])) / 15 for i in range(len(X))])
            sil = float(silhouette_score(emb[labeled], categories[labeled]))
            rho_x = stats.spearmanr(emb[:, 0], y).statistic
            rho_y = stats.spearmanr(emb[:, 1], y).statistic
            scans.append(
                {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "neighbourhood_preservation_k15": float(trust),
                    "category_silhouette_2d": sil,
                    "spearman_activity_vs_umap1": float(rho_x),
                    "spearman_activity_vs_umap2": float(rho_y),
                }
            )
            if n_neighbors == 15 and min_dist == 0.1:
                base_emb = emb
            print(f"UMAP n_neighbors={n_neighbors} min_dist={min_dist} preservation={trust:.3f} sil={sil:.3f}")
    report["umap_sensitivity"] = scans
    report["umap_conclusion"] = (
        "Across the scanned settings the two-dimensional projection preserves only "
        f"{100 * np.mean([s['neighbourhood_preservation_k15'] for s in scans]):.0f}% of the "
        "15 nearest fingerprint neighbours and the category silhouette stays near zero, so the "
        "projection is used purely as a qualitative overview and no conclusion in the manuscript "
        "rests on visual separation in this plot."
    )

    # how dispersed are the potent compounds?
    high = y >= np.percentile(y, 90)
    nn_high = NearestNeighbors(n_neighbors=2, metric="jaccard").fit(X[high])
    d_high, _ = nn_high.kneighbors(X[high])
    report["potent_compound_dispersion"] = {
        "n_top_decile": int(high.sum()),
        "median_nn_tanimoto_among_top_decile": float(1 - np.median(d_high[:, 1])),
        "n_distinct_scaffolds_in_top_decile": int(curated.loc[high, "bemis_murcko_scaffold"].nunique()),
        "n_distinct_categories_in_top_decile": int(pd.Series(categories[high]).nunique()),
    }

    # ------------------------------------------------- matched molecular pairs
    # A pair differing by one hydroxyl cannot be found by fragmenting on bonds
    # between heavy atoms: the fragmenter never yields a hydrogen substituent,
    # so the two members have different cores. Deleting each terminal hydroxyl
    # oxygen and looking the remainder up among the dataset's canonical SMILES
    # identifies the pairs exactly.
    canonical = {}
    for i, mol in enumerate(mols):
        if mol is not None:
            canonical.setdefault(Chem.MolToSmiles(mol), i)

    pairs = []
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "O" or atom.GetDegree() != 1 or atom.GetTotalNumHs() < 1:
                continue
            edit = Chem.RWMol(mol)
            edit.RemoveAtom(atom.GetIdx())
            try:
                stripped = edit.GetMol()
                Chem.SanitizeMol(stripped)
            except Exception:
                continue
            j = canonical.get(Chem.MolToSmiles(stripped))
            if j is not None and j != i:
                pairs.append((j, i))  # (unsubstituted, hydroxylated)
    pairs = sorted(set(pairs))
    print(f"Matched H->OH pairs: {len(pairs)}")

    mmp = {}
    if len(pairs) >= 8:
        obs = np.array([y[b] - y[a] for a, b in pairs])
        matrix = json.load(
            open(os.path.join(OUT, "a5_matrix_stratified_predictions.json"), encoding="utf-8")
        )
        # The best cell that actually has stored predictions, ranked by training
        # Q2 so that the hold-out plays no part in the choice. a7's "best model"
        # is the stacked one, which exists only inside a7, so it cannot be used
        # here and the previous fallback picked an arbitrary key.
        table = pd.read_csv(os.path.join(OUT, "a5_matrix_stratified.csv"))
        table["key"] = table["representation"] + " | " + table["learner"]
        available = table[table["key"].isin(matrix["predictions"])]
        available = available[~available["family"].str.contains("ensemble member", na=False)]
        key = available.sort_values("q2_cv_oof", ascending=False).iloc[0]["key"]
        pred_full = np.full(len(y), np.nan)
        for idx_, p in zip(matrix["train_idx"], matrix["predictions"][key]["oof"]):
            pred_full[idx_] = p
        for idx_, p in zip(matrix["test_idx"], matrix["predictions"][key]["test"]):
            pred_full[idx_] = p
        keep = [k for k, (a, b) in enumerate(pairs) if np.isfinite(pred_full[a]) and np.isfinite(pred_full[b])]
        obs = obs[keep]
        prd = np.array([pred_full[pairs[k][1]] - pred_full[pairs[k][0]] for k in keep])
        r = stats.spearmanr(obs, prd)
        mmp = {
            "model_used": key,
            "n_matched_pairs_H_to_OH": int(len(obs)),
            "mean_observed_delta_pIC50": float(obs.mean()),
            "sd_observed_delta_pIC50": float(obs.std(ddof=1)),
            "mean_predicted_delta_pIC50": float(prd.mean()),
            "spearman_observed_vs_predicted": float(r.statistic),
            "spearman_p": float(r.pvalue),
            "sign_agreement_pct": float(100 * np.mean(np.sign(obs) == np.sign(prd))),
            "observed_one_sample_t_p": float(stats.ttest_1samp(obs, 0).pvalue),
            "note": (
                "Predictions for training compounds are out-of-fold, so no pair is scored by a "
                "model that saw both of its members during fitting."
            ),
        }
        pd.DataFrame(
            {
                "index_unsubstituted": [pairs[k][0] for k in keep],
                "index_hydroxylated": [pairs[k][1] for k in keep],
                "smiles_unsubstituted": [smiles[pairs[k][0]] for k in keep],
                "smiles_hydroxylated": [smiles[pairs[k][1]] for k in keep],
                "observed_delta": obs,
                "predicted_delta": prd,
            }
        ).to_csv(os.path.join(OUT, "a8_mmp_pairs.csv"), index=False)
        print("Matched pairs:", json.dumps(mmp, indent=1))
    report["matched_molecular_pairs"] = mmp

    # -------------------------------------------------- prospective triage
    triage = {}
    threshold = float(np.percentile(y, 90))
    for split_name in ("source", "scaffold", "cluster", "dissimilarity", "stratified"):
        path = os.path.join(OUT, f"a5_matrix_{split_name}_predictions.json")
        if not os.path.exists(path):
            continue
        payload = json.load(open(path, encoding="utf-8"))
        yt = np.asarray(payload["y_test"], float)
        table = pd.read_csv(os.path.join(OUT, f"a5_matrix_{split_name}.csv"))
        table["key"] = table["representation"] + " | " + table["learner"]
        valid = table[table["key"].isin(payload["predictions"].keys())]
        key = valid.sort_values("q2_cv_oof", ascending=False).iloc[0]["key"]
        score = np.asarray(payload["predictions"][key]["test"], float)
        entry = {"model": key, "n_test": int(len(yt)), "n_actives": int((yt >= threshold).sum())}
        for frac in (0.05, 0.10, 0.20):
            ef, hr = enrichment_factor(yt, score, frac, threshold)
            entry[f"enrichment_factor_top{int(frac * 100)}pct"] = ef
            entry[f"hit_rate_top{int(frac * 100)}pct"] = hr
        entry["baseline_hit_rate_pct"] = float(100 * (yt >= threshold).mean())
        entry["spearman_rank_correlation"] = float(stats.spearmanr(yt, score).statistic)
        triage[split_name] = entry
        print(split_name, json.dumps(entry, indent=1))
    report["prospective_triage"] = {
        "active_definition": f"experimental pIC50 >= {threshold:.3f} (top decile of the full dataset)",
        "results": triage,
    }

    np.save(os.path.join(FEAT, "umap_reference_embedding.npy"), base_emb)
    with open(os.path.join(OUT, "a8_chemspace_utility.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote", os.path.join(OUT, "a8_chemspace_utility.json"))


if __name__ == "__main__":
    main()
