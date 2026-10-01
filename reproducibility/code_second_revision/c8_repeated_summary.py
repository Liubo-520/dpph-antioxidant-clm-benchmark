"""Stability of the family ranking across repeated partitions (Reviewer 2, point 2).

Reads the per-partition outputs of c2_grid.py, c3_finetune.py and c4_gnn.py and
asks whether the ordering of the representation families, and in particular
its reversal between the chemistry-stratified and the group-disjoint designs,
survives when the assignment of scaffolds, clusters or publications to the
hold-out is resampled.

No model is selected with held-out information. Two training-only rules are
applied and reported side by side:

  nested   in every partition the representative of a grid family is the cell
           with the highest inner cross-validated R2 on that partition's
           training compounds (the score the grid search itself maximised),
           among the 18 candidate cells of passes 1 and 2 of c2_grid.py;
  fixed    the representative of a grid family is the cell that training-set
           Q2_CV selected on the primary partitions, with its hyperparameters
           re-tuned inside each repeated training partition.

The fine-tuned language model and the graph network are single end-to-end
predictors and are the same under both rules. Held-out compounds are scored
once.

Three kinds of evidence are produced for each design:

  per fold   held-out R2 and rank of every family in each of the ten
             partitions, their mean and spread, and how often each family leads;
  paired     for pairs of families, the number of partitions in which one
             beats the other, the mean difference with a t interval over
             partitions, and an exact Wilcoxon signed-rank test;
  pooled     the hold-outs of a design do not overlap, so they are pooled into
             one out-of-fold prediction per held-out compound and family and
             scored together, with a cluster bootstrap over the groups that
             were held out together.

The reversal itself is tested as an interaction: the pooled difference between
two families under a group-disjoint design minus the same difference under the
stratified design, with a bootstrap over Butina clusters.

Five of the ten folds of c1_partitions.py are used (common.REP_FOLDS), so each
design contributes five 9:1 partitions whose hold-outs are disjoint and together
cover half of the compounds.

A second set of partitions repeats the rule of the primary scaffold- and
cluster-disjoint partitions itself (c1b_partitions_same_rule.py): the hold-out
is filled with the smallest groups first, so it consists of structurally
isolated compounds. Those hold-outs overlap with each other, so they are not
pooled and no interval is computed over partitions; instead one resample of
Butina clusters is applied to every partition at once, which resamples a
compound shared by several hold-outs in all of them together, and the mean over
partitions is recomputed on each resample (analyse_same_rule).

The script runs on whatever partitions are complete and says how many it used.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from c2_grid import PASSES
from common import (DESIGNS, FAMILIES, OUT, REP_FOLDS, SAME_RULE_DESIGNS, SAME_RULE_SEEDS, load_light,
                    load_partitions, read_json, write_json)

N_BOOT = 2000
SEED = 20260930
GRID_FAMILIES = ("Fp", "Desc", "Frozen", "Hyb", "HybClm")
FIXED = {"Fp": "ECFP4 | ExtraTrees", "Desc": "RDKit-desc | ExtraTrees", "Frozen": "MoLFormer-XL | SVR",
         "Hyb": "ECFP4+Mordred | ExtraTrees", "HybClm": "ECFP4+ChemBERTa-ZINC | ExtraTrees"}
NESTED_CANDIDATES = [f"{r} | {l}" for r, l in PASSES[1] + PASSES[2]]
LATE_CELLS = [f"{r} | {l}" for r, l in PASSES[3]]
ZINC_FROZEN = ["ChemBERTa-ZINC | Ridge", "ChemBERTa-ZINC | SVR"]
CONTRASTS = [("Frozen", "Hyb"), ("Frozen", "Fp"), ("Frozen", "Ft"), ("Frozen", "Desc"),
             ("Frozen", "Gnn"), ("Hyb", "Fp"), ("Desc", "Fp"), ("Gnn", "Ft")]
FAM_KEYS = [k for k, _ in FAMILIES]


def metrics(y, p):
    return {"r2": float(r2_score(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)), "spearman": float(stats.spearmanr(y, p).statistic)}


def load_partition(pid, mode):
    """Family -> representative of this partition under the given rule, or None if incomplete."""
    gpath = os.path.join(OUT, "grid", f"{pid}.json")
    if not os.path.exists(gpath):
        return None
    cells = read_json(gpath)["cells"]
    needed = list(FIXED.values()) if mode == "fixed" else NESTED_CANDIDATES
    if any(k not in cells for k in needed):
        return None
    out = {}
    for fam in GRID_FAMILIES:
        if mode == "fixed":
            key = FIXED[fam]
            cand = {key: cells[key]}
        else:
            cand = {k: cells[k] for k in NESTED_CANDIDATES if cells[k]["family_key"] == fam}
            key = max(cand, key=lambda k: cand[k]["cv_r2_inner_mean"])
        by_test = max(cand, key=lambda k: cand[k]["test_r2"])
        out[fam] = {"model": key, "inner": cells[key]["cv_r2_inner_mean"],
                    "pred": np.asarray(cells[key]["test_predictions"], float),
                    "n_candidates": len(cand), "test_selected_r2": cand[by_test]["test_r2"]}
        # would one of the four late tree-ensemble cells have been chosen instead?
        late = {k: cells[k] for k in LATE_CELLS if k in cells and cells[k]["family_key"] == fam}
        if mode == "nested" and all(k in cells for k in LATE_CELLS):
            out[fam]["late_cell_would_win"] = bool(
                late and max(c["cv_r2_inner_mean"] for c in late.values()) > out[fam]["inner"])
    for fam, folder in (("Ft", "finetune"), ("Gnn", "gnn")):
        path = os.path.join(OUT, folder, f"{pid}.json")
        if os.path.exists(path):
            d = read_json(path)
            out[fam] = {"model": d["model"], "inner": d["cv_r2_inner_mean"],
                        "pred": np.asarray(d["test_predictions"], float), "n_candidates": 1,
                        "test_selected_r2": d["test_r2"]}
    # the frozen cell of the encoder that is fine-tuned, chosen by its inner score,
    # for the like-for-like question of what fine-tuning that encoder buys
    zinc = {k: cells[k] for k in ZINC_FROZEN if k in cells}
    if len(zinc) == len(ZINC_FROZEN):
        key = max(zinc, key=lambda k: zinc[k]["cv_r2_inner_mean"])
        out["_zinc_frozen"] = {"model": key, "r2": zinc[key]["test_r2"]}
    return out


def paired_over_folds(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    out = {"n_folds": int(n), "n_a_better": int((d > 0).sum()), "mean_delta": float(d.mean()),
           "sd_delta": float(d.std(ddof=1)),
           "ci95": [float(d.mean() - t * se), float(d.mean() + t * se)],
           "p_sign_two_sided": float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue)}
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        out["p_wilcoxon"] = float("nan")
    return out


def cluster_boot(groups, rng):
    """Index arrays that resample whole groups with replacement."""
    uniq, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = []
    for _ in range(N_BOOT):
        draw = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([members[g] for g in draw]))
    return out


def analyse(mode, y, parts, group_of, butina, rng, verbose=True):
    report = {"mode": mode, "designs": {}}
    fold_rows, pooled_pred = [], {}

    for design in DESIGNS:
        folds = {}
        for k in REP_FOLDS:
            got = load_partition(f"{design}_f{k:02d}", mode)
            if got is not None:
                folds[k] = got
        full = {k: v for k, v in folds.items() if all(f in v for f in FAM_KEYS)}
        d = {"n_folds_grid_complete": len(folds), "n_folds_all_families": len(full),
             "folds_used": sorted(full)}
        report["designs"][design] = d
        if len(full) < 3:
            if verbose:
                print(f"[{mode}] {design}: only {len(full)} complete partitions, skipped")
            continue

        r2 = {f: [] for f in FAM_KEYS}
        rmse = {f: [] for f in FAM_KEYS}
        rho = {f: [] for f in FAM_KEYS}
        ranks = {f: [] for f in FAM_KEYS}
        chosen = {f: [] for f in FAM_KEYS}
        optimism = {f: [] for f in FAM_KEYS}
        late_wins = {f: [] for f in GRID_FAMILIES}
        for k in sorted(full):
            pid = f"{design}_f{k:02d}"
            te = np.asarray(parts[pid]["test_idx"])
            m = {f: metrics(y[te], full[k][f]["pred"]) for f in FAM_KEYS}
            order = sorted(FAM_KEYS, key=lambda f: -m[f]["r2"])
            for f in FAM_KEYS:
                r2[f].append(m[f]["r2"])
                rmse[f].append(m[f]["rmse"])
                rho[f].append(m[f]["spearman"])
                ranks[f].append(order.index(f) + 1)
                chosen[f].append(full[k][f]["model"])
                optimism[f].append(full[k][f]["test_selected_r2"] - m[f]["r2"])
                if f in late_wins and "late_cell_would_win" in full[k][f]:
                    late_wins[f].append(full[k][f]["late_cell_would_win"])
                fold_rows.append({"rule": mode, "design": design, "fold": k, "partition": pid,
                                  "family": f, "selected_model": full[k][f]["model"],
                                  "inner_cv_r2": full[k][f]["inner"], "n_test": int(len(te)),
                                  "test_sd_pIC50": float(y[te].std(ddof=1)), **m[f],
                                  "rank": order.index(f) + 1,
                                  "n_candidates": full[k][f]["n_candidates"]})
        fam_stats = {}
        for f in FAM_KEYS:
            v, rk = np.asarray(r2[f]), np.asarray(ranks[f])
            fam_stats[f] = {
                "r2_mean": float(v.mean()), "r2_sd": float(v.std(ddof=1)),
                "r2_median": float(np.median(v)), "r2_min": float(v.min()), "r2_max": float(v.max()),
                "rmse_mean": float(np.mean(rmse[f])), "rmse_sd": float(np.std(rmse[f], ddof=1)),
                "spearman_mean": float(np.mean(rho[f])),
                "rank_mean": float(rk.mean()), "rank_median": float(np.median(rk)),
                "n_first": int((rk == 1).sum()), "n_top3": int((rk <= 3).sum()),
                "n_last": int((rk == len(FAM_KEYS)).sum()),
                "selected_models": pd.Series(chosen[f]).value_counts().to_dict(),
                "mean_optimism_of_test_selection": float(np.mean(optimism[f])),
                "r2_by_fold": [float(x) for x in v], "rank_by_fold": [int(x) for x in rk],
                "rmse_by_fold": [float(x) for x in rmse[f]],
            }
            if f in late_wins and late_wins[f]:
                fam_stats[f]["late_cells_checked_in_folds"] = len(late_wins[f])
                fam_stats[f]["late_cell_would_have_been_selected"] = int(sum(late_wins[f]))
        by_mean = sorted(FAM_KEYS, key=lambda f: -fam_stats[f]["r2_mean"])
        for pos, f in enumerate(by_mean, 1):
            fam_stats[f]["rank_of_mean_r2"] = pos
        by_rank = sorted(FAM_KEYS, key=lambda f: fam_stats[f]["rank_mean"])
        for pos, f in enumerate(by_rank, 1):
            fam_stats[f]["rank_of_mean_rank"] = pos
        d["families"] = fam_stats
        d["ranking_by_mean_r2"] = by_mean
        d["n_negative_r2"] = int(sum(x < 0 for f in FAM_KEYS for x in r2[f]))
        d["n_scored"] = int(len(FAM_KEYS) * len(full))
        d["folds_where_every_family_is_negative"] = int(
            sum(all(r2[f][i] < 0 for f in FAM_KEYS) for i in range(len(full))))
        fr = stats.friedmanchisquare(*[r2[f] for f in FAM_KEYS])
        d["friedman"] = {"statistic": float(fr.statistic), "p": float(fr.pvalue)}
        R = np.array([ranks[f] for f in FAM_KEYS], float)          # families x folds
        n_f, n_k = R.shape
        S = ((R.sum(1) - n_k * (n_f + 1) / 2) ** 2).sum()
        d["kendall_w"] = float(12 * S / (n_k ** 2 * (n_f ** 3 - n_f)))
        d["paired"] = {f"{a}-{b}": paired_over_folds(r2[a], r2[b]) for a, b in CONTRASTS}
        if all("_zinc_frozen" in full[k] for k in full):
            zf = [full[k]["_zinc_frozen"]["r2"] for k in sorted(full)]
            d["same_encoder"] = {
                "frozen_chemberta_zinc_mean_r2": float(np.mean(zf)),
                "fine_tuned_mean_r2": float(np.mean(r2["Ft"])),
                "frozen_cells": pd.Series([full[k]["_zinc_frozen"]["model"] for k in sorted(full)]).value_counts().to_dict(),
                "fine_tuned_minus_frozen": paired_over_folds(r2["Ft"], zf),
            }

        if len(full) == len(REP_FOLDS):
            pooled = {f: np.full(len(y), np.nan) for f in FAM_KEYS}
            for k in full:
                te = np.asarray(parts[f"{design}_f{k:02d}"]["test_idx"])
                for f in FAM_KEYS:
                    pooled[f][te] = full[k][f]["pred"]
            pooled_pred[design] = pooled
            held = np.flatnonzero(np.isfinite(pooled[FAM_KEYS[0]]))      # compounds held out once
            d["n_pooled_compounds"] = int(len(held))
            yh = y[held]
            boots = cluster_boot(np.asarray(group_of[design])[held], rng)
            pool = {}
            for f in FAM_KEYS:
                ph = pooled[f][held]
                bb = np.array([r2_score(yh[i], ph[i]) for i in boots])
                pool[f] = {**metrics(yh, ph),
                           "r2_ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}
            order = sorted(FAM_KEYS, key=lambda f: -pool[f]["r2"])
            for pos, f in enumerate(order, 1):
                pool[f]["rank"] = pos
            d["pooled"] = pool
            d["pooled_ranking"] = order
            d["pooled_paired"] = {}
            for a_, b_ in CONTRASTS:
                pa, pb = pooled[a_][held], pooled[b_][held]
                diff = np.array([r2_score(yh[i], pa[i]) - r2_score(yh[i], pb[i]) for i in boots])
                d["pooled_paired"][f"{a_}-{b_}"] = {
                    "delta_r2": float(pool[a_]["r2"] - pool[b_]["r2"]),
                    "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                    "p_bootstrap_two_sided": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))),
                }

        if verbose:
            print(f"\n=== [{mode}] {design}: {len(full)} partitions with all seven families")
            tab = pd.DataFrame({f: {"mean R2": fam_stats[f]["r2_mean"], "sd": fam_stats[f]["r2_sd"],
                                    "min": fam_stats[f]["r2_min"], "max": fam_stats[f]["r2_max"],
                                    "mean rank": fam_stats[f]["rank_mean"], "first": fam_stats[f]["n_first"],
                                    "pooled R2": d.get("pooled", {}).get(f, {}).get("r2", np.nan)}
                                for f in FAM_KEYS}).T
            print(tab.round(3).to_string())
            print(f"Friedman p={d['friedman']['p']:.4g}  Kendall W={d['kendall_w']:.3f}  "
                  f"negative R2 in {d['n_negative_r2']}/{d['n_scored']}")
            for name, c in d["paired"].items():
                print(f"  {name:12s} A better in {c['n_a_better']}/{c['n_folds']}  mean d={c['mean_delta']:+.3f} "
                      f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  Wilcoxon p={c['p_wilcoxon']:.3f}")
            if mode == "nested":
                print("  selected:", {f: fam_stats[f]["selected_models"] for f in GRID_FAMILIES})
            if "same_encoder" in d:
                c = d["same_encoder"]["fine_tuned_minus_frozen"]
                print(f"  same encoder: fine-tuned {d['same_encoder']['fine_tuned_mean_r2']:.3f} vs frozen ChemBERTa-ZINC "
                      f"{d['same_encoder']['frozen_chemberta_zinc_mean_r2']:.3f}; fine-tuned ahead in "
                      f"{c['n_a_better']}/{c['n_folds']}, mean d={c['mean_delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")

    # ------------------------------------------- the reversal as an interaction
    # The two designs hold out different compounds, so each gap is computed on
    # the compounds its own design held out. One resample of Butina clusters is
    # applied to both, which keeps structurally related compounds together.
    inter = {}
    if "stratified" in pooled_pred:
        boots = cluster_boot(butina, rng)
        everything = np.arange(len(y))
        for design in DESIGNS[1:]:
            if design not in pooled_pred:
                continue
            inter[design] = {}
            for a_, b_ in CONTRASTS:
                def gap(des, idx=everything):
                    pa, pb = pooled_pred[des][a_], pooled_pred[des][b_]
                    idx = idx[np.isfinite(pa[idx])]
                    ea = (y[idx] - pa[idx]) ** 2
                    eb = (y[idx] - pb[idx]) ** 2
                    return float((eb.mean() - ea.mean()) / np.var(y[idx]))   # R2 units, a minus b
                bs = np.array([gap(design, i) - gap("stratified", i) for i in boots])
                inter[design][f"{a_}-{b_}"] = {
                    "gap_under_design": gap(design), "gap_under_stratified": gap("stratified"),
                    "interaction": float(gap(design) - gap("stratified")),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "p_bootstrap_two_sided": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
                }
            if verbose:
                c = inter[design]["Frozen-Hyb"]
                print(f"[{mode}] interaction {design:9s} Frozen-Hyb: stratified {c['gap_under_stratified']:+.3f} -> "
                      f"{c['gap_under_design']:+.3f}, change {c['interaction']:+.3f} "
                      f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] p={c['p_bootstrap_two_sided']:.3f}")
    report["interaction_vs_stratified"] = inter
    return report, fold_rows, pooled_pred


def boot_mean_r2(y, items, fam, W):
    """Mean over partitions of the held-out R2 of one family, for every row of weights W."""
    acc = np.zeros(len(W))
    for te, rep in items:
        w, yt, p = W[:, te], y[te], rep[fam]["pred"]
        mu = (w * yt).sum(1) / w.sum(1)
        res = (w * (yt - p) ** 2).sum(1)
        tot = (w * (yt[None, :] - mu[:, None]) ** 2).sum(1)
        acc += 1 - res / tot
    return acc / len(items)


def two_sided(b):
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))


def analyse_same_rule(mode, y, parts, butina, rng, verbose=True):
    """The primary rule repeated: isolated-compound hold-outs of the scaffold and cluster designs."""
    report = {"mode": mode, "designs": {}}
    rows = []
    uniq, inv = np.unique(np.asarray(butina).astype(str), return_inverse=True)
    W = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
                  for _ in range(N_BOOT)]).astype(float)

    strat = []
    for k in REP_FOLDS:
        got = load_partition(f"stratified_f{k:02d}", mode)
        if got is not None and all(f in got for f in FAM_KEYS):
            strat.append((np.asarray(parts[f"stratified_f{k:02d}"]["test_idx"]), got))

    for design in SAME_RULE_DESIGNS:
        items, ids = [], []
        for seed in SAME_RULE_SEEDS:
            pid = f"{design}_s{seed:02d}"
            got = load_partition(pid, mode)
            if got is not None and all(f in got for f in FAM_KEYS):
                items.append((np.asarray(parts[pid]["test_idx"]), got))
                ids.append(pid)
        d = {"n_partitions": len(items), "partitions_used": ids}
        report["designs"][design] = d
        if len(items) < 3:
            if verbose:
                print(f"[{mode}] same rule, {design}: only {len(items)} complete partitions, skipped")
            continue

        r2 = {f: [] for f in FAM_KEYS}
        ranks = {f: [] for f in FAM_KEYS}
        chosen = {f: [] for f in FAM_KEYS}
        for pid, (te, rep) in zip(ids, items):
            m = {f: metrics(y[te], rep[f]["pred"]) for f in FAM_KEYS}
            order = sorted(FAM_KEYS, key=lambda f: -m[f]["r2"])
            for f in FAM_KEYS:
                r2[f].append(m[f]["r2"])
                ranks[f].append(order.index(f) + 1)
                chosen[f].append(rep[f]["model"])
                rows.append({"rule": mode, "design": design, "partition": pid, "family": f,
                             "selected_model": rep[f]["model"], "inner_cv_r2": rep[f]["inner"],
                             "n_test": int(len(te)), "test_sd_pIC50": float(y[te].std(ddof=1)), **m[f],
                             "rank": order.index(f) + 1, "n_candidates": rep[f]["n_candidates"]})
        fam_stats = {}
        for f in FAM_KEYS:
            v, rk = np.asarray(r2[f]), np.asarray(ranks[f])
            bb = boot_mean_r2(y, items, f, W)
            fam_stats[f] = {
                "r2_mean": float(v.mean()), "r2_sd": float(v.std(ddof=1)),
                "r2_min": float(v.min()), "r2_max": float(v.max()),
                "r2_mean_ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                "rank_mean": float(rk.mean()), "n_first": int((rk == 1).sum()),
                "n_top3": int((rk <= 3).sum()), "n_last": int((rk == len(FAM_KEYS)).sum()),
                "selected_models": pd.Series(chosen[f]).value_counts().to_dict(),
                "r2_by_partition": [float(x) for x in v], "rank_by_partition": [int(x) for x in rk],
            }
        for pos, f in enumerate(sorted(FAM_KEYS, key=lambda f: -fam_stats[f]["r2_mean"]), 1):
            fam_stats[f]["rank_of_mean_r2"] = pos
        for pos, f in enumerate(sorted(FAM_KEYS, key=lambda f: fam_stats[f]["rank_mean"]), 1):
            fam_stats[f]["rank_of_mean_rank"] = pos
        d["families"] = fam_stats
        d["ranking_by_mean_r2"] = sorted(FAM_KEYS, key=lambda f: -fam_stats[f]["r2_mean"])
        R = np.array([ranks[f] for f in FAM_KEYS], float)
        n_f, n_k = R.shape
        d["kendall_w"] = float(12 * ((R.sum(1) - n_k * (n_f + 1) / 2) ** 2).sum() / (n_k ** 2 * (n_f ** 3 - n_f)))

        d["paired"], d["interaction_vs_stratified"] = {}, {}
        for a_, b_ in CONTRASTS:
            da = np.asarray(r2[a_]) - np.asarray(r2[b_])
            bs = boot_mean_r2(y, items, a_, W) - boot_mean_r2(y, items, b_, W)
            d["paired"][f"{a_}-{b_}"] = {
                "n_partitions": int(len(da)), "n_a_better": int((da > 0).sum()),
                "mean_delta": float(da.mean()), "min_delta": float(da.min()), "max_delta": float(da.max()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "p_bootstrap_two_sided": two_sided(bs),
            }
            if len(strat) == len(REP_FOLDS):
                ref = np.mean([r2_score(y[te], rep[a_]["pred"]) - r2_score(y[te], rep[b_]["pred"])
                               for te, rep in strat])
                bref = boot_mean_r2(y, strat, a_, W) - boot_mean_r2(y, strat, b_, W)
                d["interaction_vs_stratified"][f"{a_}-{b_}"] = {
                    "gap_under_design": float(da.mean()), "gap_under_stratified": float(ref),
                    "interaction": float(da.mean() - ref),
                    "ci95": [float(np.percentile(bs - bref, 2.5)), float(np.percentile(bs - bref, 97.5))],
                    "p_bootstrap_two_sided": two_sided(bs - bref),
                }

        if verbose:
            print(f"\n=== [{mode}] same rule, {design}: {len(items)} partitions of isolated compounds")
            tab = pd.DataFrame({f: {"mean R2": fam_stats[f]["r2_mean"], "sd": fam_stats[f]["r2_sd"],
                                    "min": fam_stats[f]["r2_min"], "max": fam_stats[f]["r2_max"],
                                    "mean rank": fam_stats[f]["rank_mean"], "first": fam_stats[f]["n_first"]}
                                for f in FAM_KEYS}).T
            print(tab.round(3).to_string())
            print(f"Kendall W={d['kendall_w']:.3f}")
            for name, c in d["paired"].items():
                i = d["interaction_vs_stratified"].get(name)
                extra = (f"   vs stratified {i['gap_under_stratified']:+.3f}: change {i['interaction']:+.3f} "
                         f"[{i['ci95'][0]:+.3f}, {i['ci95'][1]:+.3f}] p={i['p_bootstrap_two_sided']:.3f}") if i else ""
                print(f"  {name:12s} A better in {c['n_a_better']}/{c['n_partitions']}  mean d={c['mean_delta']:+.3f} "
                      f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] p={c['p_bootstrap_two_sided']:.3f}{extra}")
            if mode == "nested":
                print("  selected:", {f: fam_stats[f]["selected_models"] for f in GRID_FAMILIES})
    return report, rows


def main():
    smiles, y, curated, _ = load_light()
    payload = load_partitions()
    parts = payload["partitions"]
    butina = np.asarray(payload["butina_cluster_of_compound"])
    group_of = {
        "stratified": np.arange(len(y)),
        "scaffold": curated["bemis_murcko_scaffold"].fillna("").astype(str).to_numpy(),
        "cluster": butina,
        "source": curated["source_doi"].fillna("NA").astype(str).to_numpy(),
    }

    out = {"n_boot": N_BOOT, "seed": SEED, "folds_used": list(REP_FOLDS),
           "fixed_representatives": FIXED,
           "nested_candidates": NESTED_CANDIDATES, "late_cells": LATE_CELLS}
    all_rows = []
    for mode in ("nested", "fixed"):
        rng = np.random.default_rng(SEED)
        report, rows, pooled = analyse(mode, y, parts, group_of, butina, rng)
        out[mode] = report
        all_rows += rows
        if pooled:
            cols = {"compound_index": np.arange(len(y)), "pIC50": y}
            for design, p in pooled.items():
                for f in FAM_KEYS:
                    cols[f"{design}__{f}"] = p[f]
            pd.DataFrame(cols).to_csv(os.path.join(OUT, f"c8_repeated_pooled_predictions_{mode}.csv"), index=False)

    # how often does nested re-selection differ from the fixed representative?
    df = pd.DataFrame(all_rows)
    if len(df):
        df.to_csv(os.path.join(OUT, "c8_repeated_by_fold.csv"), index=False)
        both = df[df["family"].isin(GRID_FAMILIES)].pivot_table(
            index=["partition", "family"], columns="rule", values="selected_model", aggfunc="first").dropna()
        if {"nested", "fixed"} <= set(both.columns):
            same = both["nested"] == both["fixed"]
            out["nested_vs_fixed"] = {
                "n_family_partition_combinations": int(len(both)),
                "n_same_cell": int(same.sum()),
                "by_family": {f: {"n": int(len(g)), "n_same": int((g["nested"] == g["fixed"]).sum())}
                              for f, g in both.groupby(level="family")},
            }
            print("\nnested vs fixed:", out["nested_vs_fixed"])

    # the rule of the primary partitions, repeated
    out["same_rule"] = {"seeds": list(SAME_RULE_SEEDS), "designs": list(SAME_RULE_DESIGNS),
                        "overlap": payload.get("same_rule", {}).get("overlap", {})}
    sr_rows = []
    for mode in ("nested", "fixed"):
        rng = np.random.default_rng(SEED)
        report, rows = analyse_same_rule(mode, y, parts, butina, rng)
        out["same_rule"][mode] = report
        sr_rows += rows
    if sr_rows:
        sr = pd.DataFrame(sr_rows)
        sr.to_csv(os.path.join(OUT, "c8_same_rule_by_partition.csv"), index=False)
        both = sr[sr["family"].isin(GRID_FAMILIES)].pivot_table(
            index=["partition", "family"], columns="rule", values="selected_model", aggfunc="first").dropna()
        if {"nested", "fixed"} <= set(both.columns):
            out["same_rule"]["nested_vs_fixed"] = {
                "n_family_partition_combinations": int(len(both)),
                "n_same_cell": int((both["nested"] == both["fixed"]).sum()),
            }
            print("same rule, nested vs fixed:", out["same_rule"]["nested_vs_fixed"])

    write_json(os.path.join(OUT, "c8_repeated_summary.json"), out)
    print("\nwrote", os.path.join(OUT, "c8_repeated_summary.json"))


if __name__ == "__main__":
    main()
