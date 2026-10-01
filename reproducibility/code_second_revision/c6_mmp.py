"""Matched-molecular-pair test under pair-aware splitting (Reviewer 2, point 3).

The first revision scored each matched pair with conventional molecule-level
out-of-fold predictions. That guarantees no model saw the molecule it predicts,
but the model predicting one member of a pair was usually trained on the other
member, so the test did not show that the model generalises to a transformation
it has not seen applied to that molecule. This script

  1. counts, for the conventional protocol, how often the paired analogue was in
     the training data of the model that made each prediction;
  2. repeats the test under connected-component-aware cross-validation: the
     pairs form a graph, every connected component of that graph is assigned
     whole to one fold, and each molecule is therefore predicted by a model
     trained on neither its partner nor any molecule linked to it through a
     chain of hydroxyl substitutions;
  3. repeats it under scaffold-aware cross-validation, which is stricter still:
     both members of a pair share a Bemis-Murcko scaffold, so holding out whole
     scaffolds removes the partner and every other compound on that scaffold
     (predictions taken from the scaffold design of c2_grid.py when available).

Confidence intervals come from a cluster bootstrap over connected components,
because pairs that share a molecule are not independent.

    python c6_mmp.py            everything; the component-aware fits are cached
"""

import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.model_selection import GridSearchCV, KFold

from c1_partitions import balanced_group_folds
from common import pin  # noqa: F401
from common import N_FOLDS, OUT, R1_OUT, REP_FOLDS, load_light, r1_import, read_json, write_json

r1_import()
from a5_matrix import INNER_CV, learner_grid, load_representations  # noqa: E402

RDLogger.DisableLog("rdApp.*")

N_BOOT = 2000
SEED = 20260930
THREADS = 4
CACHE = os.path.join(OUT, "c6_mmp_component_cv_predictions.json")

# The model of the first revision first (highest training Q2_CV with stored
# predictions), then the Q2-selected representatives of the fingerprint and
# frozen-language-model families, so that the result is seen for a descriptor,
# a fingerprint and a learned representation.
MODELS = [
    ("RDKit-desc", "ExtraTrees"),
    ("ECFP4", "ExtraTrees"),
    ("MoLFormer-XL", "SVR"),
]


# ------------------------------------------------------------------ the pairs
def enumerate_pairs(smiles):
    """(unsubstituted, hydroxylated) index pairs; identical to a8 of round one."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    canonical = {}
    for i, mol in enumerate(mols):
        if mol is not None:
            canonical.setdefault(Chem.MolToSmiles(mol), i)
    pairs, kinds = [], {}
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "O" or atom.GetDegree() != 1 or atom.GetTotalNumHs() < 1:
                continue
            nbr = atom.GetNeighbors()[0]
            if nbr.GetIsAromatic():
                kind = "phenolic"
            elif any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(nbr).GetSymbol() == "O"
                     for b in nbr.GetBonds()):
                kind = "carboxylic"
            else:
                kind = "aliphatic or enolic"
            edit = Chem.RWMol(mol)
            edit.RemoveAtom(atom.GetIdx())
            try:
                stripped = edit.GetMol()
                Chem.SanitizeMol(stripped)
            except Exception:
                continue
            j = canonical.get(Chem.MolToSmiles(stripped))
            if j is not None and j != i:
                pairs.append((j, i))
                kinds[(j, i)] = kind
    pairs = sorted(set(pairs))
    return pairs, [kinds[p] for p in pairs], mols


def connected_components(n, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


# ------------------------------------------------------------------ statistics
def pair_stats(pairs, comp_of_pair, y, pred, rng):
    obs = np.array([y[b] - y[a] for a, b in pairs])
    prd = np.array([pred[b] - pred[a] for a, b in pairs])
    rho = stats.spearmanr(obs, prd)
    agree = np.sign(obs) == np.sign(prd)
    nz = obs != 0
    out = {
        "n_pairs": int(len(pairs)),
        "mean_observed_delta": float(obs.mean()),
        "sd_observed_delta": float(obs.std(ddof=1)),
        "mean_predicted_delta": float(prd.mean()),
        "spearman": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "pearson": float(stats.pearsonr(obs, prd).statistic),
        "sign_agreement_pct": float(100 * agree.mean()),
        "sign_agreement_binomial_p": float(stats.binomtest(int(agree[nz].sum()), int(nz.sum()), 0.5,
                                                           alternative="greater").pvalue),
        "mae_of_delta": float(np.mean(np.abs(obs - prd))),
        "predicted_delta_one_sample_t_p": float(stats.ttest_1samp(prd, 0).pvalue),
    }
    # cluster bootstrap: resample connected components, keep all their pairs
    comps = np.unique(comp_of_pair)
    members = {c: np.flatnonzero(comp_of_pair == c) for c in comps}
    b_rho, b_sign, b_mean = [], [], []
    for _ in range(N_BOOT):
        draw = rng.choice(comps, size=len(comps), replace=True)
        idx = np.concatenate([members[c] for c in draw])
        if np.std(obs[idx]) == 0 or np.std(prd[idx]) == 0:
            continue
        b_rho.append(stats.spearmanr(obs[idx], prd[idx]).statistic)
        b_sign.append(100 * np.mean(np.sign(obs[idx]) == np.sign(prd[idx])))
        b_mean.append(prd[idx].mean())
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out["spearman_ci95"] = q(b_rho)
    out["sign_agreement_ci95"] = q(b_sign)
    out["mean_predicted_delta_ci95"] = q(b_mean)
    return out, obs, prd


# ------------------------------------------------- conventional (round one)
def conventional_predictions(key, n):
    """Round-one protocol: 10-fold OOF on the stratified training set + hold-out."""
    payload = read_json(os.path.join(R1_OUT, "a5_matrix_stratified_predictions.json"))
    train_idx = np.asarray(payload["train_idx"])
    test_idx = np.asarray(payload["test_idx"])
    pred = np.full(n, np.nan)
    pred[train_idx] = payload["predictions"][key]["oof"]
    pred[test_idx] = payload["predictions"][key]["test"]
    # which fold model predicted each training compound (a5_matrix.OOF_CV)
    fold = np.full(n, -1)
    oof_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    for k, (_, va) in enumerate(oof_cv.split(train_idx)):
        fold[train_idx[va]] = k
    return pred, fold, set(test_idx.tolist())


def exposure(pairs, fold, test_set):
    """Was the partner in the training data of the model predicting each member?"""
    def partner_seen(target, partner):
        if target in test_set:            # predicted by the model fitted on all training compounds
            return partner not in test_set
        if partner in test_set:           # partner never used for fitting
            return False
        return fold[target] != fold[partner]

    both = one = none = 0
    for a, b in pairs:
        s = int(partner_seen(a, b)) + int(partner_seen(b, a))
        both += s == 2
        one += s == 1
        none += s == 0
    n = len(pairs)
    return {"pairs_partner_seen_for_both_members": int(both),
            "pairs_partner_seen_for_one_member": int(one),
            "pairs_partner_seen_for_neither": int(none),
            "pct_pairs_with_any_partner_exposure": float(100 * (both + one) / n),
            "pct_predictions_made_by_a_model_trained_on_the_partner": float(100 * (2 * both + one) / (2 * n))}


# ------------------------------------------------ component-aware predictions
def component_cv_predictions(reps, y, fold_of_compound):
    cache = read_json(CACHE) if os.path.exists(CACHE) else {}
    for rep_name, learner in MODELS:
        key = f"{rep_name} | {learner}"
        if key in cache:
            continue
        X = reps[rep_name]
        pred = np.full(len(y), np.nan)
        t0 = time.perf_counter()
        for k in range(N_FOLDS):
            te = np.flatnonzero(fold_of_compound == k)
            tr = np.flatnonzero(fold_of_compound != k)
            est, grid = learner_grid(learner, X.shape[1])
            heavy = learner in ("ExtraTrees", "CatBoost")
            if learner == "ExtraTrees":
                est.set_params(mdl__n_jobs=THREADS)
            search = GridSearchCV(est, grid, cv=INNER_CV, scoring="r2",
                                  n_jobs=1 if heavy else THREADS, refit=True)
            search.fit(X[tr], y[tr])
            pred[te] = search.best_estimator_.predict(X[te])
        cache[key] = pred.tolist()
        write_json(CACHE, cache)
        print(f"component-aware CV fitted for {key} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return {k: np.asarray(v) for k, v in cache.items()}


def scaffold_cv_predictions(key, n):
    """Held-out predictions from the scaffold-disjoint partitions of c2_grid.py.

    Compounds outside the scaffold folds in use (common.REP_FOLDS) stay NaN, so
    only pairs whose scaffold was held out are scored under this protocol; both
    members of a pair share a scaffold and are therefore held out together.
    """
    parts = read_json(os.path.join(OUT, "c1_partitions.json"))["partitions"]
    pred = np.full(n, np.nan)
    for k in REP_FOLDS:
        pid = f"scaffold_f{k:02d}"
        path = os.path.join(OUT, "grid", f"{pid}.json")
        if not os.path.exists(path):
            return None
        cell = read_json(path)["cells"].get(key)
        if cell is None:
            return None
        pred[np.asarray(parts[pid]["test_idx"])] = cell["test_predictions"]
    return pred


def main():
    pin("cpu")
    rng = np.random.default_rng(SEED)
    smiles, y, curated, _ = load_light()
    n = len(y)
    pairs, kinds, mols = enumerate_pairs(smiles)
    comp = connected_components(n, pairs)
    comp_of_pair = np.array([comp[a] for a, _ in pairs])
    in_pairs = sorted({i for p in pairs for i in p})
    comp_sizes = Counter(comp[in_pairs].tolist())

    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    same_scaffold = sum(scaffolds[a] == scaffolds[b] for a, b in pairs)

    report = {
        "n_pairs": len(pairs),
        "n_molecules_in_pairs": len(in_pairs),
        "n_connected_components": len(comp_sizes),
        "largest_component_molecules": int(max(comp_sizes.values())),
        "components_with_more_than_two_molecules": int(sum(v > 2 for v in comp_sizes.values())),
        "hydroxyl_type_counts": dict(Counter(kinds)),
        "pairs_sharing_bemis_murcko_scaffold": int(same_scaffold),
        "protocols": {},
    }
    print({k: v for k, v in report.items() if k != "protocols"})

    # ten folds over all compounds, connected components kept whole
    fold_of_compound = balanced_group_folds(comp, N_FOLDS, SEED)
    for a, b in pairs:
        assert fold_of_compound[a] == fold_of_compound[b]
    report["component_cv"] = {
        "rule": "ten folds over all compounds; every connected component of the pair graph is "
                "assigned whole to one fold (compounds in no pair form their own component)",
        "fold_sizes": [int((fold_of_compound == k).sum()) for k in range(N_FOLDS)],
        "seed": SEED,
    }

    reps = load_representations()
    comp_pred = component_cv_predictions(reps, y, fold_of_compound)

    rows = []
    for rep_name, learner in MODELS:
        key = f"{rep_name} | {learner}"
        entry = {}

        conv, fold, test_set = conventional_predictions(key, n)
        s, obs, prd = pair_stats(pairs, comp_of_pair, y, conv, rng)
        s["exposure"] = exposure(pairs, fold, test_set)
        entry["molecule_level"] = s
        if key == "RDKit-desc | ExtraTrees":
            pd.DataFrame({
                "index_unsubstituted": [a for a, _ in pairs],
                "index_hydroxylated": [b for _, b in pairs],
                "smiles_unsubstituted": [smiles[a] for a, _ in pairs],
                "smiles_hydroxylated": [smiles[b] for _, b in pairs],
                "hydroxyl_type": kinds,
                "connected_component": comp_of_pair,
                "observed_delta": obs,
                "predicted_delta_molecule_level": prd,
                "predicted_delta_component_aware": [comp_pred[key][b] - comp_pred[key][a] for a, b in pairs],
            }).to_csv(os.path.join(OUT, "c6_mmp_pairs.csv"), index=False)

        s, _, _ = pair_stats(pairs, comp_of_pair, y, comp_pred[key], rng)
        entry["component_aware"] = s

        sc = scaffold_cv_predictions(key, n)
        if sc is not None:
            held = [i for i, (a, b) in enumerate(pairs) if np.isfinite(sc[a]) and np.isfinite(sc[b])]
            if len(held) >= 8:
                s, _, _ = pair_stats([pairs[i] for i in held], comp_of_pair[held], y, sc, rng)
                entry["scaffold_aware"] = s
                # the same pairs under the pair-aware protocol, for a like-for-like comparison
                s, _, _ = pair_stats([pairs[i] for i in held], comp_of_pair[held], y, comp_pred[key], rng)
                entry["component_aware_same_pairs"] = s

        # the phenolic subset, since the text speaks of adding a phenolic hydroxyl
        ph = [i for i, k in enumerate(kinds) if k == "phenolic"]
        if len(ph) >= 8:
            sub_pairs = [pairs[i] for i in ph]
            s, _, _ = pair_stats(sub_pairs, comp_of_pair[ph], y, comp_pred[key], rng)
            entry["component_aware_phenolic_only"] = s

        report["protocols"][key] = entry
        for proto, s in entry.items():
            rows.append({"model": key, "protocol": proto, **{k: v for k, v in s.items()
                                                             if not isinstance(v, (dict, list))},
                         "spearman_lo": s["spearman_ci95"][0], "spearman_hi": s["spearman_ci95"][1],
                         "sign_lo": s["sign_agreement_ci95"][0], "sign_hi": s["sign_agreement_ci95"][1]})

    write_json(os.path.join(OUT, "c6_mmp.json"), report)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "c6_mmp_summary.csv"), index=False)
    pd.set_option("display.width", 240)
    print(table[["model", "protocol", "n_pairs", "mean_observed_delta", "mean_predicted_delta",
                 "spearman", "spearman_lo", "spearman_hi", "spearman_p", "sign_agreement_pct",
                 "sign_lo", "sign_hi", "sign_agreement_binomial_p"]].round(3).to_string(index=False))
    e = report["protocols"]["RDKit-desc | ExtraTrees"]["molecule_level"]["exposure"]
    print("exposure under the conventional protocol:", e)


if __name__ == "__main__":
    main()
