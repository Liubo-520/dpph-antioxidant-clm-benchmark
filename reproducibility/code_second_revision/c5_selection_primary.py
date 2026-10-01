"""Family representatives chosen from training data only (Reviewer 2, point 1).

Table 3 of the first revision reported, for each representation family and
partition, the cell with the highest *held-out* R2. That is model selection on
the test set. Here the representative of each family is the cell with the
highest training-set Q2_CV (the pooled out-of-fold R2 already tabulated for
every cell), and the held-out compounds are used once, to score that cell.

No model is refitted: every candidate cell, its Q2_CV, its inner grid-search
score and its held-out predictions were stored by the first revision
(a5_matrix_<partition>.csv and ..._predictions.json), so this is a re-reading
of existing outputs under a different, training-only, selection rule.

Outputs
  c5_selection_primary.json   everything below, per partition and family
  c5_selection_primary.csv    one row per family x partition
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import FAMILIES, OUT, PRIMARY, R1_OUT, family_key, read_json, write_json

N_BOOT = 2000
SEED = 20260930

# contrasts on which the ranking-inversion statement rests
CONTRASTS = [("Frozen", "Hyb"), ("Frozen", "Fp"), ("Frozen", "Ft"), ("Hyb", "Fp"), ("Frozen", "Desc")]


def boot_indices(n, rng):
    return rng.integers(0, n, size=(N_BOOT, n))


def r2_boot(y, p, idx):
    return np.array([r2_score(y[i], p[i]) for i in idx])


def main():
    rng = np.random.default_rng(SEED)
    report = {"selection_rule": "highest training-set Q2_CV within the family",
              "n_boot": N_BOOT, "partitions": {}}
    rows = []

    for split in PRIMARY:
        table = pd.read_csv(os.path.join(R1_OUT, f"a5_matrix_{split}.csv"))
        payload = read_json(os.path.join(R1_OUT, f"a5_matrix_{split}_predictions.json"))
        y = np.asarray(payload["y_test"], float)
        table["key"] = table["representation"] + " | " + table["learner"]
        table["fam"] = [family_key(f, r) for f, r in zip(table["family"], table["representation"])]
        table = table[table["fam"].notna() & table["key"].isin(payload["predictions"])].copy()
        idx = boot_indices(len(y), rng)

        chosen, part = {}, {"n_test": int(len(y)), "families": {}}
        for fam, label in FAMILIES:
            sub = table[table["fam"] == fam]
            if not len(sub):
                continue
            by_q2 = sub.sort_values("q2_cv_oof", ascending=False).iloc[0]
            by_inner = sub.sort_values("cv_r2_inner_mean", ascending=False).iloc[0]
            by_test = sub.sort_values("test_r2", ascending=False).iloc[0]
            pred = np.asarray(payload["predictions"][by_q2["key"]]["test"], float)
            b = r2_boot(y, pred, idx)
            chosen[fam] = pred
            entry = {
                "label": label,
                "n_candidates": int(len(sub)),
                "candidates": sorted(sub["key"].tolist()),
                "selected": by_q2["key"],
                "selected_by_inner_cv": by_inner["key"],
                "same_under_both_training_criteria": bool(by_q2["key"] == by_inner["key"]),
                "q2_cv": float(by_q2["q2_cv_oof"]),
                "inner_cv_r2": float(by_q2["cv_r2_inner_mean"]),
                "test_r2": float(r2_score(y, pred)),
                "test_r2_ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                "test_rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "test_mae": float(mean_absolute_error(y, pred)),
                "spearman": float(stats.spearmanr(y, pred).statistic),
                # what the first revision printed: the maximum over the hold-out
                "previous_test_selected": by_test["key"],
                "previous_test_selected_r2": float(by_test["test_r2"]),
                "selection_changed": bool(by_test["key"] != by_q2["key"]),
                "optimism_of_test_selection": float(by_test["test_r2"] - r2_score(y, pred)),
            }
            part["families"][fam] = entry

        order = sorted(part["families"], key=lambda k: -part["families"][k]["test_r2"])
        for rank, fam in enumerate(order, 1):
            part["families"][fam]["rank"] = rank
        part["n_families"] = len(order)
        part["ranking"] = order

        part["contrasts"] = {}
        for a, b_ in CONTRASTS:
            if a not in chosen or b_ not in chosen:
                continue
            d = r2_boot(y, chosen[a], idx) - r2_boot(y, chosen[b_], idx)
            w = stats.wilcoxon(np.abs(y - chosen[a]), np.abs(y - chosen[b_]))
            part["contrasts"][f"{a}-{b_}"] = {
                "delta_r2": float(r2_score(y, chosen[a]) - r2_score(y, chosen[b_])),
                "delta_r2_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_bootstrap_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                "p_wilcoxon_abs_error": float(w.pvalue),
            }
        report["partitions"][split] = part

        for fam, e in part["families"].items():
            rows.append({"partition": split, "family": e["label"], "family_key": fam,
                         **{k: v for k, v in e.items() if k not in ("candidates", "label")}})

    # retention relative to the stratified partition, for the selected cells
    base = report["partitions"]["stratified"]["families"]
    for split in PRIMARY[1:]:
        for fam, e in report["partitions"][split]["families"].items():
            e["retention_pct_of_stratified"] = float(100 * e["test_r2"] / base[fam]["test_r2"])

    n_combos = sum(len(p["families"]) for p in report["partitions"].values())
    report["summary"] = {
        "n_family_partition_combinations": n_combos,
        "n_where_selection_changed": int(sum(e["selection_changed"]
                                             for p in report["partitions"].values()
                                             for e in p["families"].values())),
        "n_where_both_training_criteria_agree": int(sum(e["same_under_both_training_criteria"]
                                                        for p in report["partitions"].values()
                                                        for e in p["families"].values())),
        "max_optimism_of_test_selection": float(max(e["optimism_of_test_selection"]
                                                    for p in report["partitions"].values()
                                                    for e in p["families"].values())),
    }

    write_json(os.path.join(OUT, "c5_selection_primary.json"), report)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "c5_selection_primary.csv"), index=False)

    pd.set_option("display.width", 220)
    show = df.pivot(index="family_key", columns="partition", values="test_r2")[list(PRIMARY)]
    print("held-out R2 of the Q2-selected representative\n", show.round(3).to_string())
    print("\nranks\n", df.pivot(index="family_key", columns="partition", values="rank")[list(PRIMARY)].to_string())
    print("\nselected cells\n", df.pivot(index="family_key", columns="partition", values="selected")[list(PRIMARY)].to_string())
    print("\nsummary", report["summary"])
    for split in PRIMARY:
        for name, c in report["partitions"][split]["contrasts"].items():
            print(f"{split:14s} {name:12s} dR2={c['delta_r2']:+.3f} CI=[{c['delta_r2_ci95'][0]:+.3f}, "
                  f"{c['delta_r2_ci95'][1]:+.3f}] p_boot={c['p_bootstrap_two_sided']:.3f} "
                  f"p_wilcoxon={c['p_wilcoxon_abs_error']:.3f}")


if __name__ == "__main__":
    main()
