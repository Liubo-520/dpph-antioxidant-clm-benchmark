"""Check the directional statements of the second revision against the outputs.

Numbers reach the documents through generated macros, so they cannot be wrong.
The sentences around them can: "the reversal does not recur", "the interval
includes zero", "the fine-tuned model loses most" are claims about the direction
or the resolution of a comparison. This script re-derives each one from the
analysis outputs and prints HOLDS or FAILS with the numbers, so that any
sentence the data contradict is rewritten before the package is built.

    python e6_verify_claims.py          exit status 1 if any claim fails
"""

import os
import sys

import numpy as np
import pandas as pd

from common import FAMILIES, OUT, PRIMARY, read_json

FAM_KEYS = [k for k, _ in FAMILIES]
results = []


def claim(where, text, ok, detail):
    results.append((where, text, bool(ok), detail))


def spans_zero(ci):
    return ci[0] <= 0 <= ci[1]


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    return read_json(p) if name.endswith(".json") else pd.read_csv(p)


def main():
    sel, rep = load("c5_selection_primary.json"), load("c8_repeated_summary.json")
    mmp, tri, sa = load("c6_mmp.json"), load("c7_triage.json"), load("a7_stats_ad.json")
    pairs = load("a7_pairwise_tests.csv")

    # ----------------------------------------------------- training-only selection
    if sel:
        P = sel["partitions"]
        S = sel["summary"]
        claim("Sec. 2.5 / response R2-1", "both training-only criteria select the same model everywhere",
              S["n_where_both_training_criteria_agree"] == S["n_family_partition_combinations"],
              f"{S['n_where_both_training_criteria_agree']} of {S['n_family_partition_combinations']}")
        fr = {s: P[s]["families"]["Frozen"]["rank"] for s in PRIMARY}
        claim("Sec. 3.2, 3.5", "frozen CLM is the weakest family on the primary stratified hold-out",
              fr["stratified"] == 7, f"rank {fr['stratified']} of 7")
        claim("Sec. 3.5", "frozen CLM has the highest R2 on the primary scaffold, cluster and dissimilarity hold-outs",
              fr["scaffold"] == fr["cluster"] == fr["dissimilarity"] == 1, str(fr))
        c = P["stratified"]["contrasts"]["Frozen-Hyb"]
        claim("Sec. 3.2, 3.5", "primary stratified: frozen is below the hybrid, interval excludes zero",
              c["delta_r2"] < 0 and not spans_zero(c["delta_r2_ci95"]), f"{c['delta_r2']:+.3f} {c['delta_r2_ci95']}")
        c = P["stratified"]["contrasts"]["Frozen-Ft"]
        claim("Sec. 3.2", "primary stratified: frozen and fine-tuned are not distinguishable",
              spans_zero(c["delta_r2_ci95"]), f"{c['delta_r2']:+.3f} {c['delta_r2_ci95']}")
        c = P["cluster"]["contrasts"]["Frozen-Hyb"]
        claim("Sec. 3.5", "primary cluster: the frozen lead over the hybrid has an interval that includes zero",
              c["delta_r2"] > 0 and spans_zero(c["delta_r2_ci95"]), f"{c['delta_r2']:+.3f} {c['delta_r2_ci95']}")
        ok, det = True, []
        for s in ("scaffold", "dissimilarity"):
            for name in ("Frozen-Hyb", "Frozen-Fp", "Frozen-Desc"):
                c = P[s]["contrasts"][name]
                ok &= spans_zero(c["delta_r2_ci95"])
                det.append(f"{s[:4]} {name}: {c['delta_r2']:+.3f}")
        claim("Sec. 3.5", "primary scaffold and dissimilarity: frozen leads over hybrid, fingerprint and "
                          "descriptor are all unresolved", ok, "; ".join(det))
        ft = {s: P[s]["families"]["Ft"]["rank"] for s in PRIMARY}
        claim("Sec. 3.5", "fine-tuned CLM is the least accurate family on the primary cluster and dissimilarity "
                          "hold-outs", ft["cluster"] == 7 and ft["dissimilarity"] == 7, str(ft))
        ok = all(not spans_zero(P[s]["contrasts"]["Frozen-Ft"]["delta_r2_ci95"])
                 and P[s]["contrasts"]["Frozen-Ft"]["delta_r2"] > 0 for s in ("cluster", "dissimilarity"))
        claim("Sec. 3.5", "there the frozen representative is ahead of the fine-tuned one, intervals exclude zero",
              ok, "; ".join(f"{s}: {P[s]['contrasts']['Frozen-Ft']['delta_r2']:+.3f} "
                            f"{P[s]['contrasts']['Frozen-Ft']['delta_r2_ci95']}" for s in ("cluster", "dissimilarity")))
        vals = [e["test_r2"] for e in P["source"]["families"].values()]
        claim("Sec. 3.5", "every family representative is negative on the primary source-disjoint hold-out",
              max(vals) < 0, f"{min(vals):.3f} to {max(vals):.3f}")

    # ------------------------------------------------------------ reported model
    if sa and pairs is not None:
        summ = load("a7_model_summary.csv")
        claim("Sec. 3.4", "the reported model has the highest Q2_CV of all scored models",
              summ.sort_values("q2_cv_oof", ascending=False).iloc[0]["model"] == sa["best_model"],
              sa["best_model"])
        by_role = {r["role"]: r for _, r in pairs.iterrows()}
        ref = by_role["reproduced reference workflow"]
        ci = eval(ref["delta_r2_ci95"]) if isinstance(ref["delta_r2_ci95"], str) else ref["delta_r2_ci95"]
        claim("Abstract / Sec. 3.3, 3.4", "the reported model exceeds the reproduced reference; interval "
                                         "excludes zero and the corrected Wilcoxon test is significant",
              ref["delta_r2"] > 0 and not spans_zero(ci) and ref["wilcoxon_p_bh"] < 0.05,
              f"{ref['delta_r2']:+.3f} {ci}, p_BH = {ref['wilcoxon_p_bh']:.4f}")

        def ci_of(role_part):
            r = next(v for k, v in by_role.items() if role_part in k)
            c = eval(r["delta_r2_ci95"]) if isinstance(r["delta_r2_ci95"], str) else r["delta_r2_ci95"]
            return r, c

        ok, det = True, []
        for part in ("ECFP4", "Mordred / RDKit", "Hybrid, ECFP4", "stack ablation"):
            r, c = ci_of(part)
            ok &= spans_zero(c)
            det.append(f"{part}: {r['delta_r2']:+.3f}")
        claim("Sec. 3.4, 3.7", "reported model is not distinguishable from the fingerprint, descriptor and "
                              "fp-descriptor hybrid representatives nor from the ablated stack", ok, "; ".join(det))
        ok, det = True, []
        for part in ("reproduced reference", "Frozen CLM", "Fine-tuned CLM", "Hybrid with a CLM"):
            r, c = ci_of(part)
            ok &= (r["wilcoxon_p_bh"] < 0.05 and r["delta_r2"] > 0 and not spans_zero(c))
            det.append(f"{part}: {r['delta_r2']:+.3f}, p_BH {r['wilcoxon_p_bh']:.4f}")
        claim("Sec. 3.7", "it is ahead of the reference, both language-model representatives and the hybrid "
                          "with a CLM, after BH correction", ok, "; ".join(det))
        r, c = ci_of("Graph network")
        claim("Sec. 3.7", "graph network: Wilcoxon significant after correction but the R2 interval spans zero",
              r["wilcoxon_p_bh"] < 0.05 and spans_zero(c), f"{r['delta_r2']:+.3f} {c}, p_BH {r['wilcoxon_p_bh']:.4f}")
        claim("Sec. 3.7", "number of significant comparisons as stated",
              int((pairs["wilcoxon_p_bh"] < 0.05).sum()) == 5 and len(pairs) == 9,
              f"{int((pairs['wilcoxon_p_bh'] < 0.05).sum())} of {len(pairs)}")
        ad = sa["applicability_domain_williams"]
        claim("Sec. 3.8", "domain filtering improves apparent accuracy",
              ad["metrics_inside_domain"]["r2"] > ad["metrics_full_test_set"]["r2"],
              f"{ad['metrics_full_test_set']['r2']:.4f} -> {ad['metrics_inside_domain']['r2']:.4f}")
        sim = sa["applicability_domain_similarity"]
        nn, err = np.asarray(sim["nn_tanimoto"]), np.asarray(sim["abs_error"])
        hi = nn >= 0.6
        claim("Sec. 3.8", "error is larger below a nearest-neighbour similarity of 0.6",
              np.sqrt(np.mean(err[~hi] ** 2)) > np.sqrt(np.mean(err[hi] ** 2)) and sim["spearman_similarity_vs_abs_error"] < 0,
              f"RMSE {np.sqrt(np.mean(err[hi] ** 2)):.3f} (n={hi.sum()}) vs {np.sqrt(np.mean(err[~hi] ** 2)):.3f} "
              f"(n={(~hi).sum()}); rho {sim['spearman_similarity_vs_abs_error']:.3f}")
        cw = {r["category"]: r for r in sa["classwise"]}
        big = sorted((r for r in sa["classwise"] if r.get("rmse") is not None), key=lambda r: -r["n"])[:2]
        rest = [r for r in sa["classwise"] if r.get("rmse") is not None and r not in big]
        claim("Sec. 3.8", "the two largest classes are predicted best",
              max(r["rmse"] for r in big) < min(r["rmse"] for r in rest),
              f"{[(r['category'], round(r['rmse'], 3)) for r in big]} vs min of rest {min(r['rmse'] for r in rest):.3f}")
        small = [cw[k] for k in ("Flavonoid", "Phenolic acid", "Terpenoid")]
        claim("Sec. 3.8", "within-class R2 is low in the flavonoid, phenolic-acid and terpenoid classes (8-16 compounds)",
              all(r["r2"] < 0.3 and 8 <= r["n"] <= 16 for r in small),
              str([(r["category"], r["n"], round(r["r2"], 3)) for r in small]))

    # ------------------------------------------------------------- matched pairs
    if mmp:
        main_key = "RDKit-desc | ExtraTrees"
        pr = mmp["protocols"][main_key]
        c = pr["component_aware"]
        claim("Sec. 3.10 / response R2-3", "pair-aware: positive correlation, interval excludes zero",
              c["spearman"] > 0 and c["spearman_ci95"][0] > 0, f"rho {c['spearman']:.3f} {c['spearman_ci95']}")
        claim("Sec. 3.10", "pair-aware: sign agreement above chance, interval excludes 50%",
              c["sign_agreement_ci95"][0] > 50, f"{c['sign_agreement_pct']:.1f}% {c['sign_agreement_ci95']}")
        claim("Sec. 3.10 / response R2-3", "pair-aware is not weaker than molecule-level for the main model",
              c["spearman"] >= pr["molecule_level"]["spearman"],
              f"{pr['molecule_level']['spearman']:.3f} -> {c['spearman']:.3f}")
        ok, det = True, []
        for key in ("ECFP4 | ExtraTrees", "MoLFormer-XL | SVR"):
            a, b = mmp["protocols"][key]["molecule_level"]["spearman"], mmp["protocols"][key]["component_aware"]["spearman"]
            ok &= abs(a) < 0.1 and 0.25 < b < 0.40
            det.append(f"{key}: {a:+.3f} -> {b:+.3f}")
        claim("Sec. 3.10 / response R2-3", "fingerprint and frozen-CLM models: near zero under molecule-level "
                                          "prediction, close to 0.3 under pair-aware prediction", ok, "; ".join(det))
        if "scaffold_aware" in pr:
            s = pr["scaffold_aware"]
            claim("Sec. 3.10", "scaffold-aware: the result stays in place (interval excludes zero)",
                  s["spearman_ci95"][0] > 0, f"n={s['n_pairs']}, rho {s['spearman']:.3f} {s['spearman_ci95']}")
        claim("Sec. 3.10", "both members of every pair share a Bemis-Murcko scaffold",
              mmp["pairs_sharing_bemis_murcko_scaffold"] == mmp["n_pairs"],
              f"{mmp['pairs_sharing_bemis_murcko_scaffold']} of {mmp['n_pairs']}")

    # -------------------------------------------------------------------- triage
    if tri:
        T = tri["primary"]
        claim("Sec. 3.10 / response R2-4", "stratified and scaffold: top-10% enrichment interval excludes chance",
              T["stratified"]["ef_top10_ci95"][0] > 1 and T["scaffold"]["ef_top10_ci95"][0] > 1,
              f"{T['stratified']['ef_top10_ci95']}, {T['scaffold']['ef_top10_ci95']}")
        ok = all(T[s]["ef_top10"] > 1 and T[s]["ef_top10_ci95"][0] <= 1 for s in ("cluster", "dissimilarity", "source"))
        claim("Sec. 3.10 / response R2-4", "cluster, dissimilarity and source: point estimate above 1, interval reaches chance",
              ok, "; ".join(f"{s}: {T[s]['ef_top10']:.2f} {np.round(T[s]['ef_top10_ci95'], 2).tolist()}"
                            for s in ("cluster", "dissimilarity", "source")))
        pot = [T[s]["n_potent"] for s in PRIMARY]
        claim("Sec. 3.10", "each hold-out contains between 13 and 26 potent compounds; top 10% is 19 compounds",
              min(pot) == 13 and max(pot) == 26 and all(T[s]["n_selected_top10"] == 19 for s in PRIMARY), str(pot))
        models = {s: T[s]["model"] for s in PRIMARY}
        claim("Table 8 caption", "ranking model is RDKit-desc ET on four partitions and ECFP4+Mordred ET on cluster",
              models["cluster"] == "ECFP4+Mordred | ExtraTrees"
              and all(models[s] == "RDKit-desc | ExtraTrees" for s in PRIMARY if s != "cluster"), str(models))
        claim("Sec. 3.5", "rank correlation on the primary source hold-out is positive, interval excludes zero",
              T["source"]["spearman_ci95"][0] > 0, f"{T['source']['spearman']:.3f} {T['source']['spearman_ci95']}")

    # ------------------------------------------------------- repeated partitions
    if rep and "families" in rep["nested"]["designs"].get("stratified", {}):
        N = rep["nested"]["designs"]
        designs = [d for d in ("stratified", "scaffold", "cluster", "source") if "families" in N[d]]
        means = {d: [N[d]["families"][f]["r2_mean"] for f in FAM_KEYS] for d in designs}
        claim("Sec. 3.6", "every family: mean R2 falls stratified > scaffold > cluster > source",
              all(N["stratified"]["families"][f]["r2_mean"] > N["scaffold"]["families"][f]["r2_mean"]
                  > N["cluster"]["families"][f]["r2_mean"] > N["source"]["families"][f]["r2_mean"] for f in FAM_KEYS),
              "; ".join(f"{d}: {min(means[d]):.3f} to {max(means[d]):.3f}" for d in designs))
        for d in ("stratified", "scaffold", "cluster", "source"):
            c = N[d]["paired"]["Frozen-Hyb"]
            claim("Sec. 3.6", f"{d}: frozen minus hybrid over partitions has an interval that includes zero",
                  spans_zero(c["ci95"]), f"ahead {c['n_a_better']}/{c['n_folds']}, mean {c['mean_delta']:+.3f} "
                                         f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
        rk = {d: N[d]["families"]["Frozen"]["rank_mean"] for d in designs}
        claim("Sec. 3.6", "frozen CLM is neither last on the repeated stratified partitions nor first on the "
                          "repeated cluster-disjoint ones (by mean rank)",
              N["stratified"]["families"]["Frozen"]["rank_of_mean_rank"] != 7
              and N["cluster"]["families"]["Frozen"]["rank_of_mean_rank"] != 1,
              f"mean rank {rk}")
        inter = rep["nested"]["interaction_vs_stratified"]
        for d in ("scaffold", "cluster"):
            if d in inter:
                c = inter[d]["Frozen-Hyb"]
                claim("Sec. 3.6", f"interaction stratified -> {d} for frozen minus hybrid includes zero",
                      spans_zero(c["ci95"]), f"{c['interaction']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
        for d in ("scaffold", "cluster"):
            c = N[d]["paired"]["Frozen-Ft"]
            claim("Sec. 3.6", f"{d}: frozen is ahead of fine-tuned in most partitions, mean difference positive",
                  c["n_a_better"] >= 4 and c["mean_delta"] > 0,
                  f"ahead {c['n_a_better']}/{c['n_folds']}, mean {c['mean_delta']:+.3f} "
                  f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
            last = max(FAM_KEYS, key=lambda f: N[d]["families"][f]["rank_mean"])
            claim("Sec. 3.6", f"{d}: the fine-tuned CLM has the worst mean rank", last == "Ft",
                  f"worst mean rank: {last} ({N[d]['families'][last]['rank_mean']:.1f})")
        c = N["stratified"]["paired"]["Frozen-Ft"]
        last = max(FAM_KEYS, key=lambda f: N["stratified"]["families"][f]["rank_mean"])
        claim("Sec. 3.6", "stratified: the fine-tuned CLM has the worst mean rank and is behind the frozen "
                          "representative in most partitions", last == "Ft" and c["n_a_better"] >= 3,
              f"worst mean rank: {last}; frozen ahead {c['n_a_better']}/{c['n_folds']}")
        sd_max = max(N["cluster"]["families"][f]["r2_sd"] for f in FAM_KEYS)
        spread = max(means["cluster"]) - min(means["cluster"])
        claim("Sec. 3.6", "cluster: the s.d. of a family across partitions exceeds the spread between family means",
              sd_max > spread, f"largest s.d. {sd_max:.3f}, spread {spread:.3f}")
        hyb = N["stratified"]["families"]["Hyb"]
        fro = N["stratified"]["families"]["Frozen"]
        prim = sel["partitions"]["stratified"]["families"]
        claim("Sec. 3.6", "primary stratified: the hybrid is above every repeated stratified partition; the "
                          "frozen value lies within their range",
              prim["Hyb"]["test_r2"] > hyb["r2_max"] and fro["r2_min"] <= prim["Frozen"]["test_r2"] <= fro["r2_max"],
              f"hybrid {prim['Hyb']['test_r2']:.3f} vs {hyb['r2_min']:.3f}-{hyb['r2_max']:.3f}; "
              f"frozen {prim['Frozen']['test_r2']:.3f} vs {fro['r2_min']:.3f}-{fro['r2_max']:.3f}")
        best = {d_: sorted(FAM_KEYS, key=lambda f: N[d_]["families"][f]["rank_mean"]) for d_ in designs}
        rm = {d_: {f: N[d_]["families"][f]["rank_mean"] for f in FAM_KEYS} for d_ in designs}
        claim("Sec. 3.6", "whole groups: best mean rank is the hybrid (scaffold), the descriptors (cluster), and "
                          "is shared by descriptors and the hybrid with a CLM (source)",
              best["scaffold"][0] == "Hyb" and rm["scaffold"]["Hyb"] < min(v for k, v in rm["scaffold"].items() if k != "Hyb")
              and best["cluster"][0] == "Desc" and rm["cluster"]["Desc"] < min(v for k, v in rm["cluster"].items() if k != "Desc")
              and rm["source"]["Desc"] == rm["source"]["HybClm"] == min(rm["source"].values())
              and sum(v == min(rm["source"].values()) for v in rm["source"].values()) == 2,
              str({d_: {k: round(v, 1) for k, v in rm[d_].items()} for d_ in ("scaffold", "cluster", "source")}))
        ws = {d_: N[d_]["kendall_w"] for d_ in designs}
        claim("Sec. 3.6", "Kendall's W is far from 1 in every design", max(ws.values()) < 0.6,
              str({k: round(v, 2) for k, v in ws.items()}))
        rho = [N["source"]["families"][f]["spearman_mean"] for f in FAM_KEYS]
        claim("Sec. 3.6", "source-disjoint repeats: mean rank correlation is positive for every family",
              min(rho) > 0, f"{min(rho):.2f} to {max(rho):.2f}")
        chars = pd.read_csv(os.path.join(OUT, "c1_partition_characterisation.csv"))
        chars = chars[(chars["design"] == "source") & (chars["fold"] < 5)]
        claim("Sec. 3.6", "repeated source-disjoint hold-outs are more remote than the primary one (0.480)",
              chars["nn_tanimoto_median"].mean() < 0.480, f"mean of medians {chars['nn_tanimoto_median'].mean():.3f}")
        s = N["source"]
        claim("Sec. 3.6", "source-disjoint repeats: family means are close to zero (between -0.2 and 0.2)",
              all(-0.2 < v < 0.2 for v in means["source"]),
              f"{min(means['source']):.3f} to {max(means['source']):.3f}; negative in {s['n_negative_r2']} of {s['n_scored']}")
        nf = rep.get("nested_vs_fixed")
        if nf:
            claim("Sec. 3.6", "nested and fixed selection mostly choose the same cell",
                  nf["n_same_cell"] / nf["n_family_partition_combinations"] > 0.5,
                  f"{nf['n_same_cell']} of {nf['n_family_partition_combinations']}")
        F = rep["fixed"]["designs"]
        ok, det = True, []
        for d in ("stratified", "cluster"):
            c = F[d]["paired"]["Frozen-Hyb"]
            ok &= spans_zero(c["ci95"])
            det.append(f"{d}: {c['mean_delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
        claim("Sec. 3.6", "the same conclusion under fixed representatives (frozen minus hybrid includes zero)",
              ok, "; ".join(det))

    # ------------------------------- the restored statements about the primary partitions
    if sel:
        P = sel["partitions"]
        ret = {s: {f: P[s]["families"][f]["retention_pct_of_stratified"] for f in FAM_KEYS}
               for s in ("scaffold", "cluster")}
        claim("Sec. 3.5", "frozen CLM retains the largest share of its stratified accuracy on the scaffold- and "
                          "cluster-disjoint partitions",
              all(max(ret[s], key=ret[s].get) == "Frozen" for s in ret),
              "; ".join(f"{s}: Frozen {ret[s]['Frozen']:.0f}%, next {sorted(ret[s].values())[-2]:.0f}%" for s in ret))
        claim("Sec. 3.5", "fine-tuned CLM retains less than the frozen representative on both structural partitions",
              all(ret[s]["Ft"] < ret[s]["Frozen"] for s in ret),
              "; ".join(f"{s}: Ft {ret[s]['Ft']:.0f}% vs Frozen {ret[s]['Frozen']:.0f}%" for s in ret))
        claim("Sec. 3.5", "the fingerprint-descriptor hybrid leads the primary stratified hold-out",
              P["stratified"]["families"]["Hyb"]["rank"] == 1, f"rank {P['stratified']['families']['Hyb']['rank']}")
        c = P["cluster"]["contrasts"]["Frozen-Fp"]
        claim("Sec. 3.5", "primary cluster: the frozen lead over the fingerprint representative excludes zero",
              c["delta_r2"] > 0 and not spans_zero(c["delta_r2_ci95"]), f"{c['delta_r2']:+.3f} {c['delta_r2_ci95']}")

    # ----------------------------------------- replicates of the primary rule (c1b, c8)
    SR = rep.get("same_rule", {}).get("nested", {}).get("designs", {}) if rep else {}
    if all("families" in SR.get(d, {}) for d in ("scaffold", "cluster")):
        for d in ("scaffold", "cluster"):
            D = SR[d]
            for pair, need in (("Frozen-Fp", 4), ("Frozen-Hyb", 4)):
                c = D["paired"][pair]
                claim("Sec. 3.6 / Abstract", f"{d} replicates: {pair.replace('-', ' ahead of ')} in most replicates, "
                                             f"mean difference positive",
                      c["n_a_better"] >= need and c["mean_delta"] > 0,
                      f"ahead {c['n_a_better']}/{c['n_partitions']}, mean {c['mean_delta']:+.3f} "
                      f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p {c['p_bootstrap_two_sided']:.3f}")
            fr = D["families"]["Frozen"]
            claim("Sec. 3.6", f"{d} replicates: the frozen CLM is among the two best families by mean rank",
                  fr["rank_of_mean_rank"] <= 2,
                  f"mean rank {fr['rank_mean']:.1f} (position {fr['rank_of_mean_rank']}); first in {fr['n_first']}, "
                  f"top three in {fr['n_top3']} of {D['n_partitions']}; all mean ranks "
                  f"{ {f: round(D['families'][f]['rank_mean'], 1) for f in FAM_KEYS} }")
            c = D["paired"]["Frozen-Ft"]
            claim("Sec. 3.6", f"{d} replicates: frozen is ahead of the fine-tuned CLM, mean difference positive",
                  c["n_a_better"] >= 4 and c["mean_delta"] > 0,
                  f"ahead {c['n_a_better']}/{c['n_partitions']}, mean {c['mean_delta']:+.3f} "
                  f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
            i = D["interaction_vs_stratified"]["Frozen-Hyb"]
            claim("Sec. 3.6", f"{d} replicates: the frozen-minus-hybrid margin is larger than on the stratified "
                              f"partitions (change positive)",
                  i["interaction"] > 0,
                  f"{i['gap_under_stratified']:+.3f} -> {i['gap_under_design']:+.3f}, change {i['interaction']:+.3f} "
                  f"[{i['ci95'][0]:+.3f}, {i['ci95'][1]:+.3f}], p {i['p_bootstrap_two_sided']:.3f}")
            i = D["interaction_vs_stratified"]["Frozen-Fp"]
            claim("Sec. 3.6", f"{d} replicates: change of the frozen-minus-fingerprint margin from the stratified "
                              f"partitions (reported, direction positive)",
                  i["interaction"] > 0,
                  f"{i['gap_under_stratified']:+.3f} -> {i['gap_under_design']:+.3f}, change {i['interaction']:+.3f} "
                  f"[{i['ci95'][0]:+.3f}, {i['ci95'][1]:+.3f}], p {i['p_bootstrap_two_sided']:.3f}")
        claim("Sec. 3.6", "replicates: the frozen CLM has the best mean rank of any family on both designs",
              all(SR[d]["families"]["Frozen"]["rank_of_mean_rank"] == 1 for d in ("scaffold", "cluster")),
              str({d: SR[d]["families"]["Frozen"]["rank_mean"] for d in ("scaffold", "cluster")}))
        ranks = [r for d in ("scaffold", "cluster") for r in SR[d]["families"]["Frozen"]["rank_by_partition"]]
        claim("Sec. 3.6", "replicates: the frozen CLM is first or second in nine of the ten replicates",
              sum(r <= 2 for r in ranks) == 9 and len(ranks) == 10, str(ranks))
        g, fz = SR["scaffold"]["families"]["Gnn"], SR["scaffold"]["families"]["Frozen"]
        claim("Sec. 3.6", "scaffold replicates: the graph network has a slightly higher mean R2 than the frozen CLM "
                          "and a worse mean rank", g["r2_mean"] > fz["r2_mean"] and g["rank_mean"] > fz["rank_mean"],
              f"Gnn {g['r2_mean']:.3f} (rank {g['rank_mean']:.1f}) vs Frozen {fz['r2_mean']:.3f} (rank {fz['rank_mean']:.1f})")
        four = {(d, p): SR[d]["paired"][p]["ci95"] for d in ("scaffold", "cluster") for p in ("Frozen-Hyb", "Frozen-Fp")}
        excl = [k for k, v in four.items() if not spans_zero(v)]
        claim("Sec. 3.6", "of the four margins over hybrid and fingerprint, only cluster Frozen-Fp excludes zero",
              excl == [("cluster", "Frozen-Fp")], str(excl))
        claim("Sec. 3.6", "change of the frozen-minus-hybrid margin from stratified: cluster excludes zero, "
                          "scaffold does not",
              not spans_zero(SR["cluster"]["interaction_vs_stratified"]["Frozen-Hyb"]["ci95"])
              and spans_zero(SR["scaffold"]["interaction_vs_stratified"]["Frozen-Hyb"]["ci95"]),
              f"scaffold {SR['scaffold']['interaction_vs_stratified']['Frozen-Hyb']['ci95']}, "
              f"cluster {SR['cluster']['interaction_vs_stratified']['Frozen-Hyb']['ci95']}")
        ft_rank = {d: SR[d]["families"]["Ft"]["rank_mean"] for d in ("scaffold", "cluster")}
        claim("Sec. 3.6", "replicates: the fine-tuned CLM is in the lower half by mean rank on both designs",
              all(v > 4 for v in ft_rank.values()), str(ft_rank))
        ov = rep["same_rule"]["overlap"]
        claim("Sec. 2.3, 3.6", "replicates share about half (scaffold) and two thirds (cluster) of their hold-out "
                               "with the primary partition; the source rule returns nearly the same hold-out",
              45 < ov["scaffold"]["mean_pct_shared_with_primary"] < 60
              and 60 < ov["cluster"]["mean_pct_shared_with_primary"] < 72
              and ov["source"]["mean_pct_shared_with_primary"] > 85,
              str({k: round(v["mean_pct_shared_with_primary"]) for k, v in ov.items()}))
    if rep and "same_encoder" in rep["nested"]["designs"].get("stratified", {}):
        c = rep["nested"]["designs"]["stratified"]["same_encoder"]["fine_tuned_minus_frozen"]
        claim("Sec. 3.6", "repeated stratified: fine-tuning does not improve on its own frozen encoder "
                          "(interval includes zero)", spans_zero(c["ci95"]),
              f"ahead {c['n_a_better']}/{c['n_folds']}, mean {c['mean_delta']:+.3f} "
              f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    r0 = load("c0_reproduce_primary.json")
    if r0 and r0.get("grid"):
        worst = max(r["abs_difference"] for r in r0["grid"])
        claim("Methods / deposit", "the second-revision grid code reproduces first-revision values on primary partitions",
              worst < 1e-6, f"largest difference over {len(r0['grid'])} cells: {worst:.1e}")
    if r0 and r0.get("gnn"):
        worst = max(r["abs_difference"] for r in r0["gnn"])
        claim("SI, seeds paragraph", "the graph-network code reproduces first-revision values to within 0.02 R2 units",
              worst < 0.02, "; ".join(f"{r['partition']}: {r['first_revision']:.3f} -> {r['second_revision_code']:.3f}"
                                      for r in r0["gnn"]))

    width = max(len(t) for _, t, _, _ in results)
    failed = 0
    for where, text, ok, detail in results:
        print(f"{'HOLDS' if ok else 'FAILS'}  {text:<{width}}  [{where}]\n       {detail}")
        failed += not ok
    print(f"\n{len(results) - failed} of {len(results)} claims hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
