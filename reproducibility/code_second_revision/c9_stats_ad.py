"""Statistics and applicability domain for the model chosen from training data.

a7_stats_ad.py of the first revision took as "the reported model" the row with
the highest held-out R2. That is the same test-set selection Reviewer 2 objected
to for the family representatives, applied one level up, so it is removed here
as well: the reported model is the one with the highest training-set Q2_CV, and
the hold-out is used only to score it.

Everything else is computed exactly as in a7 -- its functions are imported, and
its random generator is consumed in the same order up to the model summary, so
the bootstrap intervals of every individual model are unchanged from the first
revision. What changes is

  * which model the applicability-domain, conformal, class-wise and largest-
    error analyses describe;
  * the paired-comparison table, which is now anchored on that model: against
    the reproduced reference workflow, against the stack without language-model
    members, and against the training-selected representative of every family.

Outputs (same names as a7, written to the second-round output folder):
  a7_stats_ad.json, a7_model_summary.csv, a7_pairwise_tests.csv,
  a7_classwise.csv, c9_reported_model_predictions.json
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import FAMILIES, OUT, R1_FEAT, R1_OUT, family_key, load_light, r1_import, write_json

r1_import()
import a7_stats_ad as a7  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REFERENCE = "Mordred | Ensemble mean (reproduced reference workflow)"
FULL_STACK = "Stacked model over representation families"
NO_CLM_STACK = "Stacked model without language-model members"


def main():
    smiles, _, curated, _ = load_light()
    categories = curated["chemistry_category"].to_numpy()
    y_train, y_test, train_idx, test_idx, reg_test, reg_oof = a7.load_registry()
    reps = np.load(os.path.join(R1_FEAT, "representations.npz"))

    # ------------------------------------------------- stack, exactly as in a7
    matrix = pd.read_csv(os.path.join(R1_OUT, "a5_matrix_stratified.csv"))
    matrix["key"] = matrix["representation"] + " | " + matrix["learner"]
    eligible = matrix[matrix["key"].isin(reg_oof.keys())]
    eligible = eligible[~eligible["family"].str.contains("ensemble member", na=False)]
    eligible = eligible[eligible["family"] != "Matched descriptor ensemble"]
    best_per_family = eligible.sort_values("q2_cv_oof", ascending=False).drop_duplicates("family")
    members = best_per_family["key"].tolist()
    stack_test, stack_oof, coefs, intercept = a7.build_stack(y_train, reg_test, reg_oof, members)
    reg_test[FULL_STACK], reg_oof[FULL_STACK] = stack_test, stack_oof

    clm_members = [
        k for k, fam, rep in zip(best_per_family["key"], best_per_family["family"],
                                 best_per_family["representation"])
        if "chemical language model" in str(fam) or any(t in str(rep) for t in ("ChemBERTa", "MoLFormer"))
    ]
    ablation = {"full_members": members, "language_model_members": clm_members}
    without = [m for m in members if m not in clm_members]
    wo_test, wo_oof, _, _ = a7.build_stack(y_train, reg_test, reg_oof, without)
    reg_test[NO_CLM_STACK], reg_oof[NO_CLM_STACK] = wo_test, wo_oof
    ablation["with_language_model"] = a7.metrics(y_test, stack_test)
    ablation["without_language_model"] = a7.metrics(y_test, wo_test)
    ablation["delta_r2"] = ablation["with_language_model"]["r2"] - ablation["without_language_model"]["r2"]
    ablation["paired_bootstrap"] = a7.paired_bootstrap(y_test, stack_test, wo_test)
    ablation["wilcoxon_p"] = float(stats.wilcoxon(np.abs(y_test - stack_test), np.abs(y_test - wo_test)).pvalue)

    # ------------------------------------- per-model metrics, exactly as in a7
    summary = []
    for name, pred in reg_test.items():
        oof = reg_oof.get(name)
        summary.append({"model": name, **a7.metrics(y_test, pred), **a7.bootstrap_ci(y_test, pred),
                        "q2_cv_oof": float(r2_score(y_train, oof)) if oof is not None else np.nan})
    summary = pd.DataFrame(summary)

    # ------------------------------ the reported model: highest training Q2_CV
    summary = summary.sort_values("q2_cv_oof", ascending=False).reset_index(drop=True)
    best_name = summary.iloc[0]["model"]
    by_test = summary.sort_values("r2", ascending=False).iloc[0]["model"]
    summary.to_csv(os.path.join(OUT, "a7_model_summary.csv"), index=False)
    print(f"reported model (highest Q2_CV): {best_name}  "
          f"Q2={summary.iloc[0]['q2_cv_oof']:.4f} held-out R2={summary.iloc[0]['r2']:.4f}")
    print(f"model with the highest held-out R2 (not used for selection): {by_test}")

    # training-selected representative of each family on this partition
    fam_of = {k: family_key(f, r) for k, f, r in zip(matrix["key"], matrix["family"], matrix["representation"])}
    q2_of = dict(zip(summary["model"], summary["q2_cv_oof"]))
    representatives = {}
    for fam, _label in FAMILIES:
        cands = [k for k in reg_test if fam_of.get(k) == fam]
        if cands:
            representatives[fam] = max(cands, key=lambda k: q2_of[k])

    # ------------------------------------------ paired comparisons, anchored
    comparators = [REFERENCE, NO_CLM_STACK] + [representatives[f] for f, _ in FAMILIES if f in representatives]
    comparators = [c for c in dict.fromkeys(comparators) if c != best_name]
    pairs, praw = [], []
    for b in comparators:
        pb = a7.paired_bootstrap(y_test, reg_test[best_name], reg_test[b])
        ea, eb = np.abs(y_test - reg_test[best_name]), np.abs(y_test - reg_test[b])
        w = stats.wilcoxon(ea, eb)
        role = ("reproduced reference workflow" if b == REFERENCE else
                "stack ablation" if b == NO_CLM_STACK else
                "family representative: " + next(lab for f, lab in FAMILIES if representatives.get(f) == b))
        pairs.append({"model_a": best_name, "model_b": b, "role": role, **pb,
                      "wilcoxon_abs_error_p": float(w.pvalue),
                      "median_abs_error_a": float(np.median(ea)), "median_abs_error_b": float(np.median(eb))})
        praw.append(w.pvalue)
    for row, q in zip(pairs, a7.bh(praw)):
        row["wilcoxon_p_bh"] = float(q)
    pd.DataFrame(pairs).to_csv(os.path.join(OUT, "a7_pairwise_tests.csv"), index=False)

    # ----------------------------------------- applicability domain (Williams)
    best_pred = reg_test[best_name]
    Xdesc = np.nan_to_num(reps["Mordred"].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=50, random_state=42).fit(Xdesc[train_idx])
    Ztr, Zte = pca.transform(Xdesc[train_idx]), pca.transform(Xdesc[test_idx])
    G = np.linalg.pinv(Ztr.T @ Ztr)
    lev = np.einsum("ij,jk,ik->i", Zte, G, Zte)
    h_star = 3.0 * (Ztr.shape[1] + 1) / len(train_idx)
    resid = y_test - best_pred
    std_resid = resid / resid.std(ddof=1)
    inside = (lev <= h_star) & (np.abs(std_resid) <= 3)
    ad = {
        "model": best_name,
        "n_test": int(len(y_test)),
        "leverage_space": "PCA(50) of Mordred descriptors fitted on the training compounds only",
        "variance_explained": float(pca.explained_variance_ratio_.sum()),
        "h_star": float(h_star),
        "n_high_leverage": int((lev > h_star).sum()),
        "n_high_standardised_residual": int((np.abs(std_resid) > 3).sum()),
        "n_inside_domain": int(inside.sum()),
        "pct_inside_domain": float(100 * inside.mean()),
        "leverage": lev.tolist(),
        "standardised_residual": std_resid.tolist(),
        "metrics_inside_domain": a7.metrics(y_test[inside], best_pred[inside]),
        "metrics_full_test_set": a7.metrics(y_test, best_pred),
    }

    # ------------------------------------ similarity-based domain and error
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    train_fps = [fps[i] for i in train_idx]
    nn_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in test_idx])
    abs_err = np.abs(resid)
    sp = stats.spearmanr(nn_sim, abs_err)
    bin_rows = []
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        m = (nn_sim >= lo) & (nn_sim < hi)
        if m.sum() >= 3:
            bin_rows.append({
                "nn_tanimoto_bin": f"[{lo:.1f}, {hi:.1f})", "n": int(m.sum()),
                "rmse": float(np.sqrt(mean_squared_error(y_test[m], best_pred[m]))),
                "mae": float(mean_absolute_error(y_test[m], best_pred[m])),
                "r2": float(r2_score(y_test[m], best_pred[m])) if m.sum() > 5 else np.nan,
            })
    similarity_domain = {
        "nn_tanimoto_mean": float(nn_sim.mean()),
        "spearman_similarity_vs_abs_error": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "bins": bin_rows,
        "nn_tanimoto": nn_sim.tolist(),
        "abs_error": abs_err.tolist(),
    }

    # -------------------------------------------- split-conformal uncertainty
    cal_resid = np.abs(y_train - reg_oof[best_name])
    conformal = {}
    for alpha in (0.10, 0.20):
        q = float(np.quantile(cal_resid, 1 - alpha, method="higher"))
        conformal[f"{int((1 - alpha) * 100)}%"] = {
            "half_width": q, "empirical_coverage": float(np.mean(np.abs(resid) <= q)),
            "target_coverage": 1 - alpha}
    train_nn = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in train_idx if j != i]))
        for i in train_idx])
    diff_train = np.clip(1.0 - train_nn, 1e-3, None)
    diff_test = np.clip(1.0 - nn_sim, 1e-3, None)
    norm_scores = cal_resid / diff_train
    for alpha in (0.10, 0.20):
        q = float(np.quantile(norm_scores, 1 - alpha, method="higher"))
        hw = q * diff_test
        conformal[f"{int((1 - alpha) * 100)}% novelty-normalised"] = {
            "mean_half_width": float(hw.mean()),
            "empirical_coverage": float(np.mean(np.abs(resid) <= hw)),
            "target_coverage": 1 - alpha}

    # ---------------------------------------------------- class-wise metrics
    cat_test = categories[test_idx]
    class_rows = []
    for cat in sorted(set(cat_test)):
        m = cat_test == cat
        row = {"category": cat, "n": int(m.sum()), "mean_pIC50": float(y_test[m].mean())}
        if m.sum() >= 3:
            row["rmse"] = float(np.sqrt(mean_squared_error(y_test[m], best_pred[m])))
            row["mae"] = float(mean_absolute_error(y_test[m], best_pred[m]))
            row["r2"] = float(r2_score(y_test[m], best_pred[m])) if m.sum() >= 6 else np.nan
        class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(os.path.join(OUT, "a7_classwise.csv"), index=False)

    write_json(os.path.join(OUT, "a7_stats_ad.json"), {
        "best_model": best_name,
        "selection_rule": "highest training-set Q2_CV among all scored models",
        "model_with_highest_held_out_r2": by_test,
        "family_representatives": representatives,
        "stack_members": members,
        "stack_coefficients": coefs,
        "stack_intercept": intercept,
        "stack_ablation": ablation,
        "bootstrap_resamples": a7.N_BOOT,
        "applicability_domain_williams": ad,
        "applicability_domain_similarity": similarity_domain,
        "conformal_prediction": conformal,
        "classwise": class_rows,
    })
    write_json(os.path.join(OUT, "c9_reported_model_predictions.json"), {
        "model": best_name,
        "train_idx": np.asarray(train_idx).tolist(), "test_idx": np.asarray(test_idx).tolist(),
        "y_test": y_test.tolist(), "test": best_pred.tolist(), "oof": reg_oof[best_name].tolist(),
        "other": {NO_CLM_STACK: {"test": wo_test.tolist(), "oof": wo_oof.tolist()}},
    })

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(summary.head(8)[["model", "q2_cv_oof", "r2", "r2_ci95", "rmse"]].to_string(index=False))
    print(pd.DataFrame(pairs)[["model_b", "role", "delta_r2", "delta_r2_ci95", "wilcoxon_abs_error_p",
                               "wilcoxon_p_bh"]].to_string(index=False))
    print("AD:", {k: v for k, v in ad.items() if not isinstance(v, list)})
    print("similarity rho:", sp.statistic, sp.pvalue, "conformal:", conformal)
    print("ablation:", {k: v for k, v in ablation.items() if k not in ("full_members",)})


if __name__ == "__main__":
    main()
