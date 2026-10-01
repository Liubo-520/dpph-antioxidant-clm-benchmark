"""Check the directional statements in the manuscript and response letter.

Numbers reach the documents through generated macros, so they cannot be wrong.
The sentences around them can: "the language model retains more of its
accuracy", "the stack is above every individual cell", "the gap widens under
scaffold-disjoint evaluation" are claims about the direction of a comparison,
and they were written before the corresponding runs finished. This script
re-derives each one from the analysis outputs and prints HOLDS or FAILS, with
the numbers, so that any sentence contradicted by the data can be rewritten
before the package is built.

    python b5_verify_claims.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")

SPLITS = ["stratified", "scaffold", "cluster", "dissimilarity", "source"]
FAM_FP = "Fingerprint baseline"
FAM_DESC = "Descriptor baseline"
FAM_FT = "Fine-tuned chemical language model"

results = []


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")) if name.endswith(".json") else pd.read_csv(p)


def claim(where, text, ok, detail):
    results.append((where, text, ok, detail))


def best(frame, family):
    if frame is None:
        return np.nan
    sub = frame[frame["family"] == family]["test_r2"]
    return float(sub.max()) if len(sub) else np.nan


def main():
    frames = {s: load(f"a5_matrix_{s}.csv") for s in SPLITS}
    stats = load("a7_stats_ad.json")
    summary = load("a7_model_summary.csv")
    attr = load("a6_attribution.json")
    util = load("a8_chemspace_utility.json")

    # ---------------------------------------------------------------- stacking
    if summary is not None and stats is not None:
        stack = summary[summary["model"] == "Stacked model over representation families"]
        others = summary[~summary["model"].str.startswith("Stacked model")]
        if len(stack) and len(others):
            s_r2 = float(stack.iloc[0]["r2"])
            o_r2 = float(others["r2"].max())
            claim(
                "manuscript Sec. 3.4",
                "stacking does not improve on the best individual cell",
                s_r2 <= o_r2,
                f"stack R2 = {s_r2:.4f}, best individual = {o_r2:.4f} "
                f"({others.iloc[others['r2'].to_numpy().argmax()]['model']})",
            )
        abl = (stats or {}).get("stack_ablation", {})
        if abl.get("delta_r2") is not None:
            d = abl["delta_r2"]
            ci = abl.get("paired_bootstrap", {}).get("delta_r2_ci95", [np.nan, np.nan])
            claim(
                "manuscript Sec. 3.4 / abstract",
                "the language-model ablation is null (interval spans zero)",
                ci[0] <= 0 <= ci[1],
                f"delta R2 = {d:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], "
                f"Wilcoxon p = {abl.get('wilcoxon_p', float('nan')):.3g}",
            )

    # ------------------------------------------------------ retained accuracy
    FAM_FROZEN = "Frozen chemical language model"
    FAM_GNN = "Graph neural network"
    FAM_HYB = "Hybrid"
    ALL_FAMS = [FAM_FP, FAM_DESC, FAM_FROZEN, FAM_FT, FAM_HYB, FAM_GNN]
    if frames.get("stratified") is not None:
        base = {f: best(frames["stratified"], f) for f in ALL_FAMS}
        for split in ("scaffold", "cluster"):
            if frames.get(split) is None:
                continue
            ret = {}
            for f in ALL_FAMS:
                b, v = base[f], best(frames[split], f)
                ret[f] = 100 * v / b if np.isfinite(b) and np.isfinite(v) and b > 0 else np.nan
            finite = {k: v for k, v in ret.items() if np.isfinite(v)}
            order = ", ".join(f"{k.split()[0].lower()} {v:.0f}%"
                              for k, v in sorted(finite.items(), key=lambda kv: -kv[1]))
            claim(
                f"manuscript Sec. 3.5 ({split})",
                "the frozen language-model representation retains the largest share "
                "of its stratified accuracy",
                np.isfinite(ret[FAM_FROZEN])
                and ret[FAM_FROZEN] >= max(v for k, v in finite.items() if k != FAM_FROZEN),
                f"retention, best first: {order}",
            )
            others = [v for k, v in finite.items() if k != FAM_FT]
            claim(
                f"manuscript Sec. 3.5 ({split})",
                "the fine-tuned encoder retains less than the frozen one and less "
                "than the descriptor baseline",
                np.isfinite(ret[FAM_FT])
                and ret[FAM_FT] < ret.get(FAM_FROZEN, np.inf)
                and ret[FAM_FT] < ret.get(FAM_DESC, np.inf),
                f"fine-tuned {ret[FAM_FT]:.0f}%, frozen {ret.get(FAM_FROZEN, float('nan')):.0f}%, "
                f"descriptor {ret.get(FAM_DESC, float('nan')):.0f}%",
            )
        # "the most accurate model on the two partitions with the least overlap"
        for split in ("cluster", "dissimilarity"):
            f = frames.get(split)
            if f is None:
                continue
            top = f.sort_values("test_r2", ascending=False).iloc[0]
            claim(
                "manuscript Sec. 3.5 / response R2-Q1, R4-Q13",
                f"a frozen language-model cell is the best model on the {split} partition",
                top["family"] == FAM_FROZEN,
                f"best is {top['representation']} | {top['learner']} "
                f"({top['family']}) at R2 = {top['test_r2']:.4f}",
            )
        # fine-tuning helps on the easy partition, for the same encoder
        s = frames["stratified"]
        froz = s[(s["family"] == FAM_FROZEN) & (s["representation"] == "ChemBERTa-ZINC")]["test_r2"]
        ft = s[s["representation"].astype(str).str.startswith("ChemBERTa-ZINC (fine-tuned)")]["test_r2"]
        if len(froz) and len(ft):
            claim(
                "manuscript Sec. 3.5",
                "for the same encoder, fine-tuning raises the stratified score",
                float(ft.max()) > float(froz.max()),
                f"ChemBERTa-ZINC frozen {float(froz.max()):.4f} -> fine-tuned {float(ft.max()):.4f}",
            )

    # The response now says the ranking reverses between partitions rather than
    # that the fine-tuned model is always near the top.
    s, c = frames.get("stratified"), frames.get("cluster")
    if s is not None and c is not None:
        fp_s, fp_c = best(s, FAM_FP), best(c, FAM_FP)
        clm_s = max(best(s, FAM_FROZEN), best(s, FAM_FT))
        clm_c = max(best(c, FAM_FROZEN), best(c, FAM_FT))
        claim(
            "response R2-Q1 / R4-Q13",
            "the fingerprint baseline leads on the stratified partition and "
            "a language-model cell leads on the cluster-disjoint one",
            fp_s > clm_s and clm_c > fp_c,
            f"stratified: fingerprint {fp_s:.4f} vs best language model {clm_s:.4f}; "
            f"cluster: fingerprint {fp_c:.4f} vs best language model {clm_c:.4f}",
        )

    # ------------------------------------------------------------- attribution
    if attr is not None:
        rows = pd.DataFrame(attr["enrichment"])
        n_sig = int((rows["p_bh_adjusted"] < 0.05).sum())
        claim(
            "manuscript abstract / Sec. 3.8",
            "attribution is significantly enriched on chemically relevant atoms",
            n_sig > len(rows) / 2,
            f"{n_sig} of {len(rows)} method-group tests significant after BH correction",
        )
        phen = rows[rows["atom_group"].str.startswith("phenolic")]
        if len(phen):
            claim(
                "manuscript Sec. 3.8",
                "phenolic atoms carry more attribution than the permutation null",
                float(phen["mean_enrichment"].min()) > 1.0,
                f"phenolic enrichment {phen['mean_enrichment'].min():.2f}-"
                f"{phen['mean_enrichment'].max():.2f}x",
            )
        arom = rows[rows["atom_group"] == "aromatic"]
        if len(arom) and len(phen):
            claim(
                "manuscript abstract / Sec. 3.8 / response R4-Q11",
                "aromatic atoms as a class are not enriched, unlike phenolic ones",
                float(arom["mean_enrichment"].max()) < float(phen["mean_enrichment"].min()),
                f"aromatic {arom['mean_enrichment'].min():.2f}-"
                f"{arom['mean_enrichment'].max():.2f}x against phenolic "
                f"{phen['mean_enrichment'].min():.2f}-{phen['mean_enrichment'].max():.2f}x",
            )
        faith = pd.DataFrame(attr["faithfulness"])
        claim(
            "manuscript abstract / Sec. 3.8",
            "masking the top-attributed atoms changes predictions more than random masking",
            float(faith["ratio"].min()) > 1.0,
            f"ratio {faith['ratio'].min():.2f}-{faith['ratio'].max():.2f}x across "
            f"{len(faith)} methods",
        )
        cs = attr.get("cross_seed_spearman", {})
        if cs:
            med = float(np.median([v["median"] for v in cs.values()]))
            claim(
                "manuscript Sec. 3.8",
                "attributions are stable across model seeds",
                med > 0.3,
                f"median cross-seed Spearman rho = {med:.3f}",
            )
        cm = attr.get("cross_method_spearman", {})
        if cm:
            meds = [v["median"] for v in cm.values()]
            claim(
                "manuscript Sec. 3.8",
                "attributions agree between explanation methods",
                min(meds) > 0.2,
                f"pairwise median Spearman rho {min(meds):.2f}-{max(meds):.2f}",
            )

    # ------------------------------------------------------------ triage / MMP
    if util is not None:
        tri = util.get("prospective_triage", {}).get("results", {})
        for split in ("source", "scaffold", "cluster"):
            if split not in tri:
                continue
            ef = tri[split].get("enrichment_factor_top10pct", np.nan)
            claim(
                f"manuscript Sec. 3.9 / response R2 ({split})",
                "ranking by prediction enriches potent compounds over random selection",
                np.isfinite(ef) and ef > 1.0,
                f"enrichment factor at top 10% = {ef:.2f}",
            )
        mmp = util.get("matched_molecular_pairs") or {}
        if mmp.get("spearman_observed_vs_predicted") is not None:
            claim(
                "manuscript Sec. 3.9",
                "the model reproduces the measured effect of adding one phenolic hydroxyl",
                mmp["spearman_observed_vs_predicted"] > 0,
                f"n = {mmp['n_matched_pairs_H_to_OH']}, rho = "
                f"{mmp['spearman_observed_vs_predicted']:+.3f}, p = {mmp['spearman_p']:.3g}, "
                f"sign agreement {mmp['sign_agreement_pct']:.0f}%",
            )
        scans = util.get("umap_sensitivity") or []
        if scans:
            pres = [s["neighbourhood_preservation_k15"] for s in scans]
            rho = max(abs(s["spearman_activity_vs_umap1"]) for s in scans)
            sil = max(s["category_silhouette_2d"] for s in scans)
            claim(
                "manuscript Sec. 3.9 / response R5-Q7",
                "the UMAP projection carries no category or activity structure, "
                "whatever it does for local neighbourhoods",
                sil < 0.1 and rho < 0.4,
                f"category silhouette at most {sil:+.3f}, max |rho| with activity "
                f"{rho:.3f}, neighbourhood preservation {min(pres):.3f}-{max(pres):.3f}",
            )
        land = util.get("activity_landscape") or {}
        near = land.get("mean_abs_activity_gap_5_nearest_neighbours")
        rand = land.get("mean_abs_activity_gap_random_pairs")
        if near is not None and rand is not None:
            claim(
                "manuscript Sec. 3.9",
                "the activity landscape is locally smooth",
                near < rand,
                f"5-NN gap {near:.3f} against random {rand:.3f} "
                f"(Wilcoxon p = {land.get('wilcoxon_p', float('nan')):.3g})",
            )

    # ------------------------------------------------ the graph-network claim
    f = frames.get("stratified")
    if f is not None:
        gnn = best(f, "Graph neural network")
        ft = best(f, FAM_FT)
        if np.isfinite(gnn) and np.isfinite(ft):
            claim(
                "manuscript Sec. 3.2 / response R4-Q13",
                "the graph network performs comparably to the fine-tuned language model",
                abs(gnn - ft) <= 0.06,
                f"AttentiveFP {gnn:.4f}, fine-tuned CLM {ft:.4f}, difference {gnn - ft:+.4f}",
            )

    # ------------------- the source-disjoint partition, stated as a failure
    import pickle

    from scipy import stats as st

    splits_path = os.path.join(OUT, "a4_splits.json")
    curated_path = os.path.join(OUT, "curated_dataset.csv")
    if os.path.exists(splits_path) and os.path.exists(curated_path):
        sp = json.load(open(splits_path, encoding="utf-8"))["splits"].get("source")
        yv = pd.read_csv(curated_path)["pIC50"].to_numpy(float)
        preds = load("a5_matrix_source_predictions.json")
        table = frames.get("source")
        if sp and preds is not None and table is not None:
            tr = np.asarray(sp["train_idx"], int)
            yt = np.asarray(preds["y_test"], float)
            null_rmse = float(np.sqrt(np.mean((yt - yv[tr].mean()) ** 2)))
            ft = table[table["family"] == FAM_FT]
            ft = ft[ft["representation"].astype(str).str.cat(
                ft["learner"].astype(str), sep=" | ").isin(preds["predictions"])]
            if len(ft):
                row = ft.sort_values("test_r2", ascending=False).iloc[0]
                p = np.asarray(preds["predictions"][
                    f"{row['representation']} | {row['learner']}"]["test"], float)
                rmse = float(np.sqrt(np.mean((yt - p) ** 2)))
                rho = float(st.spearmanr(yt, p).statistic)
                claim(
                    "manuscript Sec. 3.5 / abstract / response R2",
                    "on the source-disjoint partition the model is worse than "
                    "predicting the training mean",
                    rmse > null_rmse,
                    f"RMSE {rmse:.3f} against {null_rmse:.3f} for the training-mean predictor",
                )
                claim(
                    "manuscript Sec. 3.5 / response R2",
                    "rank information survives on the source-disjoint partition",
                    rho > 0,
                    f"Spearman rho = {rho:+.3f}",
                )
            n_neg = int((table["test_r2"] < 0).sum())
            n_worse = int((table["test_rmse"] > null_rmse).sum())
            claim(
                "manuscript abstract / Sec. 3.5 / response R2",
                "every model on the source-disjoint partition has a negative R2, "
                "and most predict less accurately than the training mean",
                n_neg == len(table) and n_worse > len(table) / 2,
                f"{n_neg} of {len(table)} negative R2, {n_worse} of {len(table)} above "
                f"the training-mean RMSE of {null_rmse:.3f}; best R2 "
                f"{float(table['test_r2'].max()):+.3f}",
            )
            ks = st.ks_2samp(yv[tr], yt)
            claim(
                "manuscript Sec. 3.5",
                "the source-disjoint hold-out is shifted and narrower than the training set",
                ks.pvalue < 0.05 and yt.std(ddof=1) < yv[tr].std(ddof=1),
                f"mean {yt.mean():.3f} vs {yv[tr].mean():.3f}, s.d. {yt.std(ddof=1):.3f} "
                f"vs {yv[tr].std(ddof=1):.3f}, KS p = {ks.pvalue:.2g}",
            )

    # ----------------------------------------- the ChemBERTa-MTR explanation
    f = frames.get("stratified")
    if f is not None:
        froz = f[f["family"] == "Frozen chemical language model"]
        mtr = froz[froz["representation"] == "ChemBERTa-MTR"]
        ridge = mtr[mtr["learner"] == "Ridge"]["test_r2"]
        svr = mtr[mtr["learner"] == "SVR"]["test_r2"]
        if len(ridge) and len(svr):
            claim(
                "response R5-Q9",
                "the weak frozen ChemBERTa-MTR cell is a learner effect, not a "
                "property of the embedding",
                float(svr.iloc[0]) - float(ridge.iloc[0]) > 0.1,
                f"same frozen features: Ridge {float(ridge.iloc[0]):.4f}, "
                f"SVR {float(svr.iloc[0]):.4f}",
            )
        desc = best(f, FAM_DESC)
        if np.isfinite(desc):
            claim(
                "response R5-Q6",
                "tuning improves on the untuned Mordred Extra Trees of the original "
                "submission (0.7589)",
                desc > 0.7589,
                f"best tuned descriptor model {desc:.4f}",
            )

    # ------------------------------------------------- what demonstrates the leak
    # The text no longer calls 0.8759-against-0.7380 a like-for-like measurement,
    # because the leak-free number comes from a different fitting protocol. What
    # it does claim is the controlled contrast: one archived matrix, one hold-out,
    # split only by whether the encoder had already seen the compound.
    legacy = load("a9_legacy_audit.json")
    f = frames.get("stratified")
    if legacy is not None:
        ct = legacy["contamination_test"]
        arch = ct["archived_features"]
        controls = legacy["contamination_test"]["label_free_controls"]
        claim(
            "manuscript Introduction / abstract / conclusions / response / cover letter",
            "the leak is demonstrated by the seen-against-unseen contrast inside one "
            "archived hold-out, and label-free controls do not show it",
            arch["rmse_seen"] < arch["rmse_unseen"]
            and arch["rmse_ratio_unseen_over_seen"]
            > max(c["rmse_ratio_unseen_over_seen"] for c in controls.values()),
            f"archived RMSE {arch['rmse_seen']:.3f} on "
            f"{ct['n_test_seen_by_supervised_encoders']} seen against "
            f"{arch['rmse_unseen']:.3f} on {ct['n_test_never_seen']} unseen "
            f"(ratio {arch['rmse_ratio_unseen_over_seen']:.2f}; best label-free control "
            f"{max(c['rmse_ratio_unseen_over_seen'] for c in controls.values()):.2f})",
        )
    if legacy is not None and f is not None:
        leaked = legacy["ridge_probe_on_stratified_split"][
            "archived feature matrix used for the submitted result"]["test_r2"]
        clean = best(f, FAM_FT)
        if np.isfinite(clean):
            claim(
                "manuscript Introduction / abstract / conclusions",
                "the submitted result is higher than what survives the corrected "
                "protocol, which is what the text says the difference measures",
                leaked > clean,
                f"submitted {leaked:.4f} against corrected {clean:.4f}, "
                f"difference {leaked - clean:+.3f} between two workflows",
            )

    # ------------------------------------------------------ the benchmark grid
    if f is not None:
        grid = f[~f["family"].isin(["Descriptor ensemble member", "Matched descriptor ensemble",
                                    "Fine-tuned chemical language model",
                                    "Graph neural network"])]
        inputs = grid["representation"].nunique()
        per = grid.groupby("learner")["representation"].nunique()
        claim(
            "manuscript Sec. 3.2 / response / cover letter",
            "the grid is not a full product, so its size is not inputs times learners",
            len(grid) != inputs * per.size and len(grid) == int(per.sum()),
            f"{inputs} inputs, coverage {dict(per)}, {len(grid)} cells "
            f"(a full product would be {inputs * per.size})",
        )

    # ---------------------------------- the hybrid that leads the stratified table
    if f is not None:
        hyb = f[f["family"] == "Hybrid"].sort_values("test_r2", ascending=False)
        if len(hyb):
            top = str(hyb.iloc[0]["representation"])
            claim(
                "manuscript Table 3",
                "the leading hybrid on the stratified partition contains no language "
                "model, so the row cannot be labelled after one",
                not any(t in top for t in ("ChemBERTa", "MoLFormer")),
                f"best hybrid is {top} at {hyb.iloc[0]['test_r2']:.4f}",
            )

    # ------------------------------------- the inversion the abstract now leads on
    if len(frames) >= 3:
        allsplits = pd.concat(
            [f.assign(split=name) for name, f in frames.items() if f is not None],
            ignore_index=True)
        CLM = ("ChemBERTa", "MoLFormer")

        def family_of(row):
            if row["family"] == "Hybrid":
                return ("Hybrid with CLM"
                        if any(t in str(row["representation"]) for t in CLM)
                        else "Hybrid fp+desc")
            return row["family"]

        allsplits["fam2"] = allsplits.apply(family_of, axis=1)
        # the same seven representation families the manuscript's family table
        # ranks; the reproduced reference ensemble and its members are models of
        # a published workflow, not representation families, and counting them
        # here would make this line disagree with the paper
        allsplits = allsplits[~allsplits["fam2"].isin(
            ["Descriptor ensemble member", "Matched descriptor ensemble"])]
        best_at = lambda s: (allsplits[allsplits["split"] == s]
                             .groupby("fam2")["test_r2"].max().sort_values(ascending=False))
        FROZEN = "Frozen chemical language model"
        HYB = "Hybrid fp+desc"
        for split_name in ("cluster", "dissimilarity"):
            if split_name not in frames:
                continue
            strat, hard = best_at("stratified"), best_at(split_name)
            needed = (FROZEN, HYB)
            if any(fam not in strat.index or fam not in hard.index for fam in needed):
                print(f"  (skipping the {split_name} inversion check: the family table is "
                      "incomplete, so run the benchmark merge first)")
                continue
            rank = lambda s, fam: int(list(s.index).index(fam)) + 1
            claim(
                f"manuscript abstract / Sec. 3.5, {split_name}-disjoint",
                "the frozen language model is last or near last on the stratified "
                f"partition and first on the {split_name} one, while the hybrid that "
                "leads the stratified partition drops",
                rank(hard, FROZEN) == 1 and rank(strat, HYB) == 1
                and rank(hard, HYB) > rank(strat, HYB),
                f"frozen {rank(strat, FROZEN)} of {len(strat)} stratified -> "
                f"{rank(hard, FROZEN)} on {split_name}; hybrid "
                f"{rank(strat, HYB)} -> {rank(hard, HYB)} "
                f"({strat[HYB]:.3f} -> {hard[HYB]:.3f})",
            )

    # ------------------------------------------- the endpoint is nominally 30 min
    sens = load("a10_assay_sensitivity.json")
    if sens:
        worst = max(abs(c["delta_r2"]) for c in sens["cells"]) if sens["cells"] else None
        if worst is not None:
            claim(
                "manuscript Sec. 2.1 / response",
                "excluding the time-window records does not change the conclusion",
                worst < 0.05,
                f"{sens['n_window_wording']} of {sens['n_records']} records "
                f"({sens['pct_window_wording']:.1f}%), largest R2 shift {worst:+.4f}",
            )

    # --------------------------------------------------------- matched ensemble
    if frames.get("stratified") is not None:
        f = frames["stratified"]
        me = f[f["family"] == "Matched descriptor ensemble"]["test_r2"]
        if len(me):
            claim(
                "manuscript Sec. 3.3",
                "the re-implemented reference workflow reproduces the published R2 = 0.78",
                abs(float(me.max()) - 0.78) < 0.06,
                f"matched ensemble R2 = {float(me.max()):.4f} against 0.78 published",
            )

    # ------------------------------------------------------------------ report
    if not results:
        print("No outputs available yet; nothing to verify.")
        return 0
    width = max(len(c[1]) for c in results)
    failed = 0
    for where, text, ok, detail in results:
        tag = "HOLDS" if ok else "FAILS"
        failed += 0 if ok else 1
        print(f"[{tag}] {text:<{width}}  {detail}")
        print(f"         {where}")
    print(f"\n{len(results) - failed} of {len(results)} claims hold.")
    if failed:
        print("Rewrite the sentences listed as FAILS before building the package.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
