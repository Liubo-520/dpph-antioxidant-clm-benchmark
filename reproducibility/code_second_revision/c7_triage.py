"""Confidence intervals for the retrospective triage simulation (Reviewer 2, point 4).

Table 7 of the first revision reported enrichment factors as point estimates.
With 13-26 potent compounds in a hold-out of about 190, an enrichment factor in
the top 5% rests on nine or ten ranked compounds, so its sampling uncertainty
is large and has to be shown. This script adds, for every partition,

  * a 95% percentile bootstrap interval for each enrichment factor, for the
    top-10% hit rate and for the rank correlation, from 2,000 resamples of the
    held-out compounds (the potency threshold stays fixed at the top decile of
    the full dataset; resamples without any potent compound are discarded and
    counted);
  * the one-sided hypergeometric probability of drawing at least the observed
    number of potent compounds in the top fraction by chance.

The ranking model of each partition is the cell with the highest training-set
Q2_CV on that partition, as stated in the manuscript. The rule is applied to
the complete benchmark matrix of the first revision.

When the repeated partitions of c2_grid.py are available the same statistics
are computed fold by fold, with the ranking model of each fold chosen by its
inner cross-validation score, so that the enrichment is also seen across
different assignments of scaffolds, clusters and publications.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from c2_grid import PASSES
from common import DESIGNS, OUT, PRIMARY, R1_OUT, REP_FOLDS, load_light, read_json, write_json

N_BOOT = 2000
# the candidate cells available in every repeated partition (passes 1 and 2)
CANDIDATES = [f"{r} | {l}" for r, l in PASSES[1] + PASSES[2]]
SEED = 20260930
FRACTIONS = (0.05, 0.10, 0.20)


def enrichment(y_true, score, frac, threshold):
    n = len(y_true)
    k = max(1, int(round(frac * n)))
    order = np.argsort(score)[::-1]
    actives = y_true >= threshold
    if actives.sum() == 0:
        return np.nan, np.nan, k, 0
    hits = int(actives[order[:k]].sum())
    return float((hits / k) / actives.mean()), float(100 * hits / k), k, hits


def triage_stats(y_true, score, threshold, rng):
    n = len(y_true)
    actives = y_true >= threshold
    out = {"n_test": int(n), "n_potent": int(actives.sum()),
           "base_rate_pct": float(100 * actives.mean()),
           "spearman": float(stats.spearmanr(y_true, score).statistic)}
    idx = rng.integers(0, n, size=(N_BOOT, n))
    valid = np.array([(y_true[i] >= threshold).sum() > 0 for i in idx])
    out["n_boot_used"] = int(valid.sum())
    rho = np.array([stats.spearmanr(y_true[i], score[i]).statistic for i in idx])
    out["spearman_ci95"] = [float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))]
    for frac in FRACTIONS:
        tag = f"top{int(frac * 100)}"
        ef, hr, k, hits = enrichment(y_true, score, frac, threshold)
        b = np.array([enrichment(y_true[i], score[i], frac, threshold)[:2] for i in idx[valid]])
        out[f"ef_{tag}"] = ef
        out[f"ef_{tag}_ci95"] = [float(np.percentile(b[:, 0], 2.5)), float(np.percentile(b[:, 0], 97.5))]
        out[f"hit_rate_{tag}_pct"] = hr
        out[f"hit_rate_{tag}_ci95"] = [float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))]
        out[f"n_selected_{tag}"] = int(k)
        out[f"n_hits_{tag}"] = int(hits)
        # P(X >= hits) when k compounds are drawn at random from the hold-out
        out[f"p_hypergeom_{tag}"] = float(stats.hypergeom.sf(hits - 1, n, int(actives.sum()), k))
        out[f"ef_{tag}_max_possible"] = float(min(1.0, actives.sum() / k) / actives.mean())
    return out


def primary(y_all, threshold, rng):
    results = {}
    for split in PRIMARY:
        table = pd.read_csv(os.path.join(R1_OUT, f"a5_matrix_{split}.csv"))
        payload = read_json(os.path.join(R1_OUT, f"a5_matrix_{split}_predictions.json"))
        table["key"] = table["representation"] + " | " + table["learner"]
        valid = table[table["key"].isin(payload["predictions"])]
        valid = valid[~valid["family"].astype(str).str.contains("ensemble", case=False)]
        ranked = valid.sort_values("q2_cv_oof", ascending=False)
        key = ranked.iloc[0]["key"]
        yt = np.asarray(payload["y_test"], float)
        score = np.asarray(payload["predictions"][key]["test"], float)
        entry = {"model": key, "model_q2_cv": float(ranked.iloc[0]["q2_cv_oof"]),
                 "runner_up": ranked.iloc[1]["key"], "runner_up_q2_cv": float(ranked.iloc[1]["q2_cv_oof"]),
                 **triage_stats(yt, score, threshold, rng)}
        # the model the first revision used on every partition, for continuity
        ref = "RDKit-desc | ExtraTrees"
        if key != ref and ref in payload["predictions"]:
            entry["round_one_model"] = {"model": ref, **triage_stats(
                yt, np.asarray(payload["predictions"][ref]["test"], float), threshold, rng)}
        results[split] = entry
        print(f"{split:14s} {key:28s} n={entry['n_test']} potent={entry['n_potent']} "
              f"EF5={entry['ef_top5']:.2f} {np.round(entry['ef_top5_ci95'], 2)} "
              f"EF10={entry['ef_top10']:.2f} {np.round(entry['ef_top10_ci95'], 2)} "
              f"EF20={entry['ef_top20']:.2f} {np.round(entry['ef_top20_ci95'], 2)} "
              f"rho={entry['spearman']:.3f} p10={entry['p_hypergeom_top10']:.2g}")
    return results


def repeated(y_all, threshold, rng):
    parts_path = os.path.join(OUT, "c1_partitions.json")
    if not os.path.exists(parts_path):
        return None
    parts = read_json(parts_path)["partitions"]
    out = {}
    for design in DESIGNS:
        folds = []
        pooled_hits = pooled_k = pooled_act = pooled_n = 0
        for k in REP_FOLDS:
            pid = f"{design}_f{k:02d}"
            path = os.path.join(OUT, "grid", f"{pid}.json")
            if not os.path.exists(path):
                continue
            cells = read_json(path)["cells"]
            if any(c not in cells for c in CANDIDATES):
                continue
            key = max(CANDIDATES, key=lambda c: cells[c]["cv_r2_inner_mean"])
            te = np.asarray(parts[pid]["test_idx"])
            yt = y_all[te]
            score = np.asarray(cells[key]["test_predictions"], float)
            row = {"fold": k, "model": key}
            for frac in FRACTIONS:
                ef, hr, kk, hits = enrichment(yt, score, frac, threshold)
                row[f"ef_top{int(frac * 100)}"] = ef
                if frac == 0.10:
                    pooled_hits += hits
                    pooled_k += kk
            row["n_potent"] = int((yt >= threshold).sum())
            row["spearman"] = float(stats.spearmanr(yt, score).statistic)
            pooled_act += row["n_potent"]
            pooled_n += len(yt)
            folds.append(row)
        if len(folds) < len(REP_FOLDS):
            continue
        df = pd.DataFrame(folds)
        summary = {"folds": folds, "n_folds": len(folds)}
        for frac in FRACTIONS:
            col = f"ef_top{int(frac * 100)}"
            v = df[col].dropna().to_numpy()
            summary[col] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                            "min": float(v.min()), "max": float(v.max()),
                            "n_folds_above_one": int((v > 1).sum())}
        summary["spearman"] = {"mean": float(df["spearman"].mean()), "sd": float(df["spearman"].std(ddof=1)),
                               "min": float(df["spearman"].min()), "max": float(df["spearman"].max())}
        # the hold-outs do not overlap, so the top deciles of the folds can be pooled
        summary["pooled_top10"] = {
            "n_selected": int(pooled_k), "n_hits": int(pooled_hits),
            "hit_rate_pct": float(100 * pooled_hits / pooled_k),
            "base_rate_pct": float(100 * pooled_act / pooled_n),
            "enrichment_factor": float((pooled_hits / pooled_k) / (pooled_act / pooled_n)),
        }
        summary["models_used"] = df["model"].value_counts().to_dict()
        out[design] = summary
        print(f"repeated {design:11s} EF10 mean={summary['ef_top10']['mean']:.2f} "
              f"[{summary['ef_top10']['min']:.2f}, {summary['ef_top10']['max']:.2f}] "
              f"pooled EF10={summary['pooled_top10']['enrichment_factor']:.2f} "
              f"rho mean={summary['spearman']['mean']:.3f} models={summary['models_used']}")
    return out or None


def main():
    rng = np.random.default_rng(SEED)
    _, y, _, _ = load_light()
    threshold = float(np.percentile(y, 90))
    report = {
        "potent_definition": f"experimental pIC50 >= {threshold:.3f} (top decile of the full dataset)",
        "threshold": threshold,
        "n_boot": N_BOOT,
        "primary": primary(y, threshold, rng),
    }
    rep = repeated(y, threshold, rng)
    if rep:
        report["repeated"] = rep
    write_json(os.path.join(OUT, "c7_triage.json"), report)
    print("wrote", os.path.join(OUT, "c7_triage.json"))


if __name__ == "__main__":
    main()
