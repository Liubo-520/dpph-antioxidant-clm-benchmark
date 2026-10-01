"""
Uncertainty, applicability domain and statistical comparison of the models.

Answers Reviewer 1 Q2 (the Williams-plot inconsistency), Reviewer 4 Q8/Q9/Q12
and Reviewer 5 Q8: bootstrap confidence intervals for every metric, paired
compound-level significance tests with multiplicity control, a leverage- and
similarity-based applicability domain computed on the correct held-out set,
split-conformal prediction intervals with measured coverage, class-wise
performance, and the relationship between prediction error and structural
novelty.
"""

import json
import os
import pickle
import warnings
from collections import OrderedDict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")

N_BOOT = 2000
RNG = np.random.default_rng(20260908)


def metrics(y, p):
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
    }


def bootstrap_ci(y, p, n=N_BOOT):
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    r2 = np.array([r2_score(y[i], p[i]) for i in idx])
    rmse = np.array([np.sqrt(mean_squared_error(y[i], p[i])) for i in idx])
    mae = np.array([mean_absolute_error(y[i], p[i]) for i in idx])
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"r2_ci95": q(r2), "rmse_ci95": q(rmse), "mae_ci95": q(mae)}


def paired_bootstrap(y, pa, pb, n=N_BOOT):
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    d = np.array([r2_score(y[i], pa[i]) - r2_score(y[i], pb[i]) for i in idx])
    return {
        "delta_r2": float(r2_score(y, pa) - r2_score(y, pb)),
        "delta_r2_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_two_sided_bootstrap": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
    }


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        out[i] = prev
    return out


def load_registry():
    """Test-set predictions of every candidate model, keyed by display name."""
    payload = json.load(open(os.path.join(OUT, "a5_matrix_stratified_predictions.json"), encoding="utf-8"))
    y_test = np.asarray(payload["y_test"], float)
    y_train = np.asarray(payload["y_train"], float)
    test_idx = np.asarray(payload["test_idx"], int)
    train_idx = np.asarray(payload["train_idx"], int)

    reg_test, reg_oof = OrderedDict(), OrderedDict()
    for k, v in payload["predictions"].items():
        reg_test[k] = np.asarray(v["test"], float)
        reg_oof[k] = np.asarray(v["oof"], float)

    return y_train, y_test, train_idx, test_idx, reg_test, reg_oof


def build_stack(y_train, reg_test, reg_oof, members):
    Xo = np.column_stack([reg_oof[m] for m in members])
    Xt = np.column_stack([reg_test[m] for m in members])
    meta = RidgeCV(alphas=np.logspace(-3, 3, 25))
    meta.fit(Xo, y_train)
    return meta.predict(Xt), meta.predict(Xo), dict(zip(members, meta.coef_.tolist())), float(meta.intercept_)


def main():
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        df = pickle.load(fh)["df"]
    smiles = df["smiles"].tolist()
    curated = pd.read_csv(os.path.join(OUT, "curated_dataset.csv"))
    categories = curated["chemistry_category"].to_numpy()

    y_train, y_test, train_idx, test_idx, reg_test, reg_oof = load_registry()
    reps = np.load(os.path.join(FEAT, "representations.npz"))

    # ------------------------------------------------------------ stacked model
    # One member per representation family, chosen by training Q2_CV so that the
    # held-out set plays no part in the choice. The meta-learner is fitted on
    # out-of-fold predictions only.
    matrix = pd.read_csv(os.path.join(OUT, "a5_matrix_stratified.csv"))
    matrix["key"] = matrix["representation"] + " | " + matrix["learner"]
    eligible = matrix[matrix["key"].isin(reg_oof.keys())]
    # the reproduced reference workflow is a comparison target, not one of the
    # representation families the stack is built over
    eligible = eligible[~eligible["family"].str.contains("ensemble member", na=False)]
    eligible = eligible[eligible["family"] != "Matched descriptor ensemble"]
    best_per_family = eligible.sort_values("q2_cv_oof", ascending=False).drop_duplicates("family")
    members = best_per_family["key"].tolist()
    stack_test, stack_oof, coefs, intercept = build_stack(y_train, reg_test, reg_oof, members)
    reg_test["Stacked model over representation families"] = stack_test
    reg_oof["Stacked model over representation families"] = stack_oof

    # Ablation: does the chemical language model contribute anything the
    # fingerprints and descriptors do not already provide?
    #
    # Membership is decided by what the representation actually contains, not
    # by its family label. The "Hybrid" family holds both language-model
    # concatenations and plain fingerprint-plus-descriptor ones, and the best
    # hybrid on this dataset is ECFP4+Mordred, which contains no language model
    # at all. Removing it as though it did would attribute its contribution to
    # the language model and overstate the ablation.
    CLM_TOKENS = ("ChemBERTa", "MoLFormer")
    clm_members = [
        k
        for k, fam, rep in zip(
            best_per_family["key"], best_per_family["family"], best_per_family["representation"]
        )
        if "chemical language model" in str(fam) or any(t in str(rep) for t in CLM_TOKENS)
    ]
    ablation = {"full_members": members, "language_model_members": clm_members}
    without = [m for m in members if m not in clm_members]
    if without and clm_members:
        wo_test, wo_oof, _, _ = build_stack(y_train, reg_test, reg_oof, without)
        reg_test["Stacked model without language-model members"] = wo_test
        reg_oof["Stacked model without language-model members"] = wo_oof
        ablation["with_language_model"] = metrics(y_test, stack_test)
        ablation["without_language_model"] = metrics(y_test, wo_test)
        ablation["delta_r2"] = ablation["with_language_model"]["r2"] - ablation["without_language_model"]["r2"]
        ablation["paired_bootstrap"] = paired_bootstrap(y_test, stack_test, wo_test)
        ablation["wilcoxon_p"] = float(
            stats.wilcoxon(np.abs(y_test - stack_test), np.abs(y_test - wo_test)).pvalue
        )

    # ----------------------------------------------------- metrics + bootstrap
    summary = []
    for name, pred in reg_test.items():
        m = metrics(y_test, pred)
        ci = bootstrap_ci(y_test, pred)
        oof = reg_oof.get(name)
        summary.append(
            {
                "model": name,
                **m,
                **ci,
                "q2_cv_oof": float(r2_score(y_train, oof)) if oof is not None else np.nan,
            }
        )
    summary = pd.DataFrame(summary).sort_values("r2", ascending=False).reset_index(drop=True)
    summary.to_csv(os.path.join(OUT, "a7_model_summary.csv"), index=False)
    best_name = summary.iloc[0]["model"]
    print("Top models by held-out R2:")
    print(summary.head(12).to_string(index=False))

    # ------------------------------------------------- pairwise significance
    # The top ten by held-out R2, plus the comparisons the reviewers asked for
    # by name: the reproduced reference workflow (Reviewer 1 Q2, Reviewer 3,
    # Reviewer 4 Q5, Reviewer 5 Q1) and the fine-tuned language model, which
    # need a paired test whether or not they rank in the top ten.
    top = summary.head(10)["model"].tolist()
    for pattern in ("Ensemble mean (reproduced reference workflow)", "(fine-tuned)"):
        for name in summary["model"]:
            if pattern in str(name) and name not in top:
                top.append(name)
                break
    pairs, praw = [], []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            pb = paired_bootstrap(y_test, reg_test[a], reg_test[b])
            ea = np.abs(y_test - reg_test[a])
            eb = np.abs(y_test - reg_test[b])
            w = stats.wilcoxon(ea, eb)
            pairs.append(
                {
                    "model_a": a,
                    "model_b": b,
                    **pb,
                    "wilcoxon_abs_error_p": float(w.pvalue),
                    "median_abs_error_a": float(np.median(ea)),
                    "median_abs_error_b": float(np.median(eb)),
                }
            )
            praw.append(w.pvalue)
    if praw:
        for row, q in zip(pairs, bh(praw)):
            row["wilcoxon_p_bh"] = float(q)
    pd.DataFrame(pairs).to_csv(os.path.join(OUT, "a7_pairwise_tests.csv"), index=False)

    # ----------------------------------------- applicability domain (Williams)
    best_pred = reg_test[best_name]
    Xdesc = np.nan_to_num(reps["Mordred"].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=50, random_state=42).fit(Xdesc[train_idx])
    Ztr, Zte = pca.transform(Xdesc[train_idx]), pca.transform(Xdesc[test_idx])
    G = np.linalg.pinv(Ztr.T @ Ztr)
    lev = np.einsum("ij,jk,ik->i", Zte, G, Zte)
    p_dim = Ztr.shape[1]
    h_star = 3.0 * (p_dim + 1) / len(train_idx)
    resid = y_test - best_pred
    std_resid = resid / resid.std(ddof=1)
    ad = {
        "model": best_name,
        "n_test": int(len(y_test)),
        "leverage_space": "PCA(50) of Mordred descriptors fitted on the training compounds only",
        "variance_explained": float(pca.explained_variance_ratio_.sum()),
        "h_star": float(h_star),
        "n_high_leverage": int((lev > h_star).sum()),
        "n_high_standardised_residual": int((np.abs(std_resid) > 3).sum()),
        "n_inside_domain": int(((lev <= h_star) & (np.abs(std_resid) <= 3)).sum()),
        "pct_inside_domain": float(100 * ((lev <= h_star) & (np.abs(std_resid) <= 3)).mean()),
        "leverage": lev.tolist(),
        "standardised_residual": std_resid.tolist(),
    }
    inside = (lev <= h_star) & (np.abs(std_resid) <= 3)
    ad["metrics_inside_domain"] = metrics(y_test[inside], best_pred[inside])
    ad["metrics_full_test_set"] = metrics(y_test, best_pred)

    # ------------------------------------ similarity-based domain and error
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    train_fps = [fps[i] for i in train_idx]
    nn_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in test_idx])
    abs_err = np.abs(resid)
    sp = stats.spearmanr(nn_sim, abs_err)
    bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bin_rows = []
    for lo, hi in bins:
        m = (nn_sim >= lo) & (nn_sim < hi)
        if m.sum() >= 3:
            bin_rows.append(
                {
                    "nn_tanimoto_bin": f"[{lo:.1f}, {hi:.1f})",
                    "n": int(m.sum()),
                    "rmse": float(np.sqrt(mean_squared_error(y_test[m], best_pred[m]))),
                    "mae": float(mean_absolute_error(y_test[m], best_pred[m])),
                    "r2": float(r2_score(y_test[m], best_pred[m])) if m.sum() > 5 else np.nan,
                }
            )
    similarity_domain = {
        "nn_tanimoto_mean": float(nn_sim.mean()),
        "spearman_similarity_vs_abs_error": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "bins": bin_rows,
        "nn_tanimoto": nn_sim.tolist(),
        "abs_error": abs_err.tolist(),
    }

    # -------------------------------------------- split-conformal uncertainty
    best_oof = reg_oof[best_name]
    cal_resid = np.abs(y_train - best_oof)
    conformal = {}
    for alpha in (0.10, 0.20):
        q = float(np.quantile(cal_resid, 1 - alpha, method="higher"))
        covered = float(np.mean(np.abs(resid) <= q))
        conformal[f"{int((1 - alpha) * 100)}%"] = {
            "half_width": q,
            "empirical_coverage": covered,
            "target_coverage": 1 - alpha,
        }
    # difficulty-normalised variant: wider intervals for structurally novel molecules
    train_nn = np.array(
        [
            max(
                DataStructs.BulkTanimotoSimilarity(
                    fps[i], [fps[j] for j in train_idx if j != i]
                )
            )
            for i in train_idx
        ]
    )
    diff_train = np.clip(1.0 - train_nn, 1e-3, None)
    diff_test = np.clip(1.0 - nn_sim, 1e-3, None)
    norm_scores = cal_resid / diff_train
    for alpha in (0.10, 0.20):
        q = float(np.quantile(norm_scores, 1 - alpha, method="higher"))
        hw = q * diff_test
        conformal[f"{int((1 - alpha) * 100)}% novelty-normalised"] = {
            "mean_half_width": float(hw.mean()),
            "empirical_coverage": float(np.mean(np.abs(resid) <= hw)),
            "target_coverage": 1 - alpha,
        }

    # ---------------------------------------------------- class-wise metrics
    cat_test = categories[test_idx]
    class_rows = []
    for cat in sorted(set(cat_test)):
        m = cat_test == cat
        row = {"category": cat, "n": int(m.sum()), "mean_pIC50": float(y_test[m].mean())}
        if m.sum() >= 3:
            row.update(
                {
                    "rmse": float(np.sqrt(mean_squared_error(y_test[m], best_pred[m]))),
                    "mae": float(mean_absolute_error(y_test[m], best_pred[m])),
                }
            )
            row["r2"] = float(r2_score(y_test[m], best_pred[m])) if m.sum() >= 6 else np.nan
        class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(os.path.join(OUT, "a7_classwise.csv"), index=False)

    payload = {
        "best_model": best_name,
        "stack_members": members,
        "stack_coefficients": coefs,
        "stack_intercept": intercept,
        "stack_ablation": ablation,
        "bootstrap_resamples": N_BOOT,
        "applicability_domain_williams": ad,
        "applicability_domain_similarity": similarity_domain,
        "conformal_prediction": conformal,
        "classwise": class_rows,
        "note_on_previous_figure": (
            "In the first submission the Williams plot was inherited from an earlier "
            "80:20 random-split experiment and therefore showed 383 points, which did not "
            "match the 192-compound stratified hold-out reported in the text. The plot is "
            "now recomputed for the reported model on the 192 held-out compounds."
        ),
    }
    with open(os.path.join(OUT, "a7_stats_ad.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\nApplicability domain:", json.dumps({k: v for k, v in ad.items() if not isinstance(v, list)}, indent=1))
    print("Conformal:", json.dumps(conformal, indent=1))
    print("Similarity bins:", json.dumps(bin_rows, indent=1))
    print("Class-wise:", json.dumps(class_rows, indent=1))


if __name__ == "__main__":
    main()
