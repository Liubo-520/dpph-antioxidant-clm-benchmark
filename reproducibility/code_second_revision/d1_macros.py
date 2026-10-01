"""LaTeX macros and table bodies for the second revision.

The first revision wrote every reported number to generated/results_macros.tex.
Most of those numbers are untouched by this round, so that file is taken over
as it stands and only two things are done to it here:

  * every value that used to depend on choosing a model by its held-out score
    is replaced by the value obtained under training-only selection
    (c5_selection_primary.py, c9_stats_ad.py), and the matched-pair and triage
    values by those of the pair-aware and interval analyses (c6, c7);
  * the numbers of the new analyses are added (c8 repeated partitions, etc.).

The result is one macro file with one definition per name, plus the table
bodies that changed or are new. A log of every replaced value, old against new,
is written next to it so that each sentence quoting one can be checked.

    python d1_macros.py
"""

import os
import re

import numpy as np
import pandas as pd

from common import (FAMILIES, FAMILY_LABEL, N_FOLDS, OUT, PRIMARY, R1_OUT, R2, REP_FOLDS, REVISION, read_json,
                    unbreakable_minus)

PKG = os.path.join(R2, "Revision_R2")
GEN = os.path.join(PKG, "generated")
R1_GEN = os.path.join(REVISION, "一审", "Revision_R1_new", "generated")

STAG = {"stratified": "Strat", "scaffold": "Scaf", "cluster": "Clus", "dissimilarity": "Diss", "source": "Src"}
PRETTY = {"stratified": "Chemistry-stratified", "scaffold": "Scaffold-disjoint", "cluster": "Cluster-disjoint",
          "dissimilarity": "Maximum-dissimilarity", "source": "Source-disjoint"}
FAM_KEYS = [k for k, _ in FAMILIES]
ROLE = {"Fp": "ECFP4 family", "Desc": "Descriptor family", "Frozen": "Frozen CLM family",
        "Ft": "Fine-tuned CLM family", "Hyb": "Hybrid, ECFP4 + Mordred", "HybClm": "Hybrid with a CLM",
        "Gnn": "Graph network"}

# family names as they read inside a sentence ("the ... family")
PROSE = {"Fp": "fingerprint", "Desc": "descriptor", "Frozen": "frozen language-model",
         "Ft": "fine-tuned language-model", "Hyb": "fingerprint--descriptor hybrid",
         "HybClm": "language-model hybrid", "Gnn": "graph-network"}

MACROS, NEW = {}, {}


# ------------------------------------------------------------------- helpers
def f(x, n=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def ci(pair, n=3):
    return f"[{pair[0]:.{n}f}, {pair[1]:.{n}f}]"


def pval(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "$<$ 0.0001"
    return "= " + f"{p:.4f}".rstrip("0").rstrip(".")


def pval_cell(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}".rstrip("0").rstrip(".")


BOOT_FLOOR = 0.001      # two-sided probability from 2000 resamples: nothing smaller can be resolved


def pboot(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "$<$ 0.001" if p < BOOT_FLOOR else f"= {p:.3f}"


def pboot_cell(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "$<$0.001" if p < BOOT_FLOOR else f"{p:.3f}"


def tex(s):
    s = str(s)
    for k, v in {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}.items():
        s = s.replace(k, v)
    return s.replace(" | ", " + ")


def mac(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    NEW[name] = str(value)


def write(name, lines):
    with open(os.path.join(GEN, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    return read_json(path) if name.endswith(".json") else pd.read_csv(path)


# -------------------------------------------- the first revision's macro file
DEFINE = re.compile(r"\\newcommand\*\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
for line in open(os.path.join(R1_GEN, "results_macros.tex"), encoding="utf-8"):
    m = DEFINE.match(line.rstrip("\n"))
    if m:
        MACROS[m.group(1)] = m.group(2)
print(f"{len(MACROS)} macros taken over from the first revision")


# =========================================================================
# 1. family representatives on the five primary partitions (c5)
# =========================================================================
sel = load("c5_selection_primary.json")
if sel:
    P = sel["partitions"]
    rows = []
    for fam, label in FAMILIES:
        cells = [f(P[s]["families"][fam]["test_r2"], 3) if fam in P[s]["families"] else "--" for s in PRIMARY]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    write("tab_generalisation.tex", rows)

    # response letter: what each cell of that table was under selection on the
    # hold-out and what it is under training-only selection
    rows = []
    for fam, label in FAMILIES:
        cells = []
        for s in PRIMARY:
            e = P[s]["families"][fam]
            old, new = e["previous_test_selected_r2"], e["test_r2"]
            cells.append(f"{old:.3f} $\\rightarrow$ \\textbf{{{new:.3f}}}" if e["selection_changed"]
                         else f"{new:.3f}")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    write("tab_generalisation_change.tex", rows)

    for s in PRIMARY:
        mac(f"NFamilies{STAG[s]}", P[s]["n_families"])
        for fam in FAM_KEYS:
            e = P[s]["families"][fam]
            mac(f"Rank{STAG[s]}{fam}", e["rank"])
            mac(f"Rtwo{fam}{STAG[s]}", f(e["test_r2"], 3))
            mac(f"RtwoCI{fam}{STAG[s]}", ci(e["test_r2_ci95"]))
            if s != "stratified":
                mac(f"Ret{fam}{STAG[s]}", f(e["retention_pct_of_stratified"], 0))
        for name, c in P[s]["contrasts"].items():
            tag = "Con" + name.replace("-", "") + STAG[s]
            mac(tag, f(c["delta_r2"], 3))
            mac(tag + "CI", ci(c["delta_r2_ci95"]))
            mac(tag + "P", pboot(c["p_bootstrap_two_sided"]))

    for s in PRIMARY:
        vals = [P[s]["families"][fam]["test_r2"] for fam in FAM_KEYS]
        mac(f"PrimMin{STAG[s]}", f(min(vals), 3))
        mac(f"PrimMax{STAG[s]}", f(max(vals), 3))
        mac(f"PrimNNeg{STAG[s]}", sum(v < 0 for v in vals))

    S = sel["summary"]
    mac("NSelCombos", S["n_family_partition_combinations"])
    mac("NSelChanged", S["n_where_selection_changed"])
    mac("NSelAgree", S["n_where_both_training_criteria_agree"])
    mac("SelMaxOptimism", f(S["max_optimism_of_test_selection"], 3))
    worst = max(((s, fam, e) for s in PRIMARY for fam, e in P[s]["families"].items()),
                key=lambda t: t[2]["optimism_of_test_selection"])
    mac("SelMaxOptimismFamily", FAMILY_LABEL[worst[1]])
    mac("SelMaxOptimismPartition", PRETTY[worst[0]].lower())
    mac("SelMaxOptimismOld", f(worst[2]["previous_test_selected_r2"], 3))
    mac("SelMaxOptimismNew", f(worst[2]["test_r2"], 3))
    opt = [e["optimism_of_test_selection"] for p in P.values() for e in p["families"].values()]
    mac("SelMeanOptimism", f(float(np.mean(opt)), 3))
    mac("SelMedianOptimism", f(float(np.median(opt)), 3))
    ncand = sorted({e["n_candidates"] for e in P["stratified"]["families"].values()})
    mac("NCandMinStrat", ncand[0])
    mac("NCandMaxStrat", ncand[-1])
    ncand_h = sorted({e["n_candidates"] for s in PRIMARY[1:] for e in P[s]["families"].values()})
    mac("NCandMinHard", ncand_h[0])
    mac("NCandMaxHard", ncand_h[-1])

    # the "best <family>" macros of round one, now the training-selected cell
    strat = P["stratified"]["families"]
    for fam, old in (("Fp", "Fingerprint"), ("Desc", "Descriptor"), ("Frozen", "Frozen"),
                     ("Ft", "FineTuned"), ("Hyb", "Hybrid"), ("Gnn", "Gnn"), ("HybClm", "HybridClm")):
        e = strat[fam]
        mac(f"Best{old}Name", tex(e["selected"]))
        mac(f"Best{old}Rtwo", f(e["test_r2"]))
        mac(f"Best{old}RMSE", f(e["test_rmse"]))
        mac(f"Best{old}Qtwo", f(e["q2_cv"]))
    r2s = [e["test_r2"] for e in strat.values()]
    mac("FamilySpread", f(max(r2s) - min(r2s)))
    mac("FamilyLeaderStrat", FAMILY_LABEL[P["stratified"]["ranking"][0]])
    for s in PRIMARY:
        mac(f"FamilyLeader{STAG[s]}", FAMILY_LABEL[P[s]["ranking"][0]])
        mac(f"FamilyLastName{STAG[s]}", FAMILY_LABEL[P[s]["ranking"][-1]])

    # Supplementary table: every family x partition selection, with what it replaces
    rows = []
    for s in PRIMARY:
        for i, (fam, label) in enumerate(FAMILIES):
            e = P[s]["families"][fam]
            changed = "yes" if e["selection_changed"] else "no"
            rows.append(
                f"{PRETTY[s] if i == 0 else ''} & {label} & {e['n_candidates']} & {tex(e['selected'])} & "
                f"{f(e['q2_cv'], 3)} & {f(e['test_r2'], 3)} {ci(e['test_r2_ci95'])} & {e['rank']} & "
                f"{f(e['previous_test_selected_r2'], 3)} & {changed} \\\\")
        rows.append(r"\addlinespace")
    write("tab_family_selection.tex", rows)

    rows = []
    for name in ("Frozen-Hyb", "Frozen-Fp", "Frozen-Ft", "Frozen-Desc", "Hyb-Fp"):
        a, b = name.split("-")
        for i, s in enumerate(PRIMARY):
            c = P[s]["contrasts"][name]
            head = f"{FAMILY_LABEL[a]} minus {FAMILY_LABEL[b]}" if i == 0 else ""
            rows.append(f"{head} & {PRETTY[s]} & {c['delta_r2']:+.3f} & "
                        f"[{c['delta_r2_ci95'][0]:+.3f}, {c['delta_r2_ci95'][1]:+.3f}] & "
                        f"{pboot_cell(c['p_bootstrap_two_sided'])} \\\\")
        rows.append(r"\addlinespace")
    write("tab_family_contrasts.tex", rows)

# complete matrices, now listed in the order of the training-set criterion
mat = pd.read_csv(os.path.join(R1_OUT, "a5_matrix_stratified.csv"))
write("tab_matrix_full.tex", [
    f"{tex(r['representation'])} & {tex(r['learner'])} & {int(r['n_features'])} & {f(r['q2_cv_oof'])} & "
    f"{f(r['test_r2'])} & {f(r['test_rmse'])} & {f(r['test_mae'])} \\\\"
    for _, r in mat.sort_values("q2_cv_oof", ascending=False).iterrows()])
rows = []
for s in PRIMARY[1:]:
    t = pd.read_csv(os.path.join(R1_OUT, f"a5_matrix_{s}.csv")).sort_values("q2_cv_oof", ascending=False)
    for i, (_, r) in enumerate(t.iterrows()):
        rows.append(f"{PRETTY[s] if i == 0 else ''} & {tex(r['representation'])} & {tex(r['learner'])} & "
                    f"{f(r['q2_cv_oof'])} & {f(r['test_r2'])} & {f(r['test_rmse'])} & {f(r['test_mae'])} \\\\")
    rows.append(r"\addlinespace")
write("tab_aux_matrix.tex", rows)


# =========================================================================
# 2. the reported model, chosen by training Q2 (c9)
# =========================================================================
sa = load("a7_stats_ad.json")
summary = load("a7_model_summary.csv")
pairs = load("a7_pairwise_tests.csv")
if sa and summary is not None:
    def lst(v):
        return eval(v) if isinstance(v, str) else v

    by_model = summary.set_index("model")
    best = sa["best_model"]
    top = by_model.loc[best]
    mac("BestModelName", tex(best))
    mac("BestTestRtwo", f(top["r2"]))
    mac("BestTestRMSE", f(top["rmse"]))
    mac("BestTestMAE", f(top["mae"]))
    mac("BestQtwo", f(top["q2_cv_oof"]))
    mac("BestTestRtwoCI", ci(lst(top["r2_ci95"])))
    mac("BestTestRMSECI", ci(lst(top["rmse_ci95"])))
    mac("TopHeldOutName", tex(sa["model_with_highest_held_out_r2"]))
    hi = by_model.loc[sa["model_with_highest_held_out_r2"]]
    mac("TopHeldOutRtwo", f(hi["r2"]))
    mac("TopHeldOutQtwo", f(hi["q2_cv_oof"]))

    # highest-Q2 model that is not a stack
    single = summary[~summary["model"].str.startswith("Stacked model")].sort_values("q2_cv_oof", ascending=False)
    mac("BestSingleName", tex(single.iloc[0]["model"]))
    mac("BestSingleRtwo", f(single.iloc[0]["r2"]))
    mac("BestSingleQtwo", f(single.iloc[0]["q2_cv_oof"]))
    top_single = summary[~summary["model"].str.startswith("Stacked model")].sort_values("r2", ascending=False).iloc[0]
    mac("TopHeldOutSingleName", tex(top_single["model"]))
    mac("TopHeldOutSingleRtwo", f(top_single["r2"]))

    # Table 2: the reported model, its ablation, one representative per family, the reference
    reps = sa["family_representatives"]
    listed = [(best, "Reported model"), ("Stacked model without language-model members", "Stack ablation")]
    listed += [(reps[k], ROLE[k]) for k in FAM_KEYS if k in reps]
    listed += [("Mordred | Ensemble mean (reproduced reference workflow)", "Reference workflow")]
    listed = sorted(listed, key=lambda t: -by_model.loc[t[0], "q2_cv_oof"])
    write("tab_main_benchmark.tex", [
        f"{role} & {tex(m)} & {f(by_model.loc[m, 'q2_cv_oof'])} & {f(by_model.loc[m, 'r2'])} & "
        f"{ci(lst(by_model.loc[m, 'r2_ci95']))} & {f(by_model.loc[m, 'rmse'])} & {f(by_model.loc[m, 'mae'])} \\\\"
        for m, role in listed])
    mac("NMainRows", len(listed))

    ad = sa["applicability_domain_williams"]
    mac("ADnTest", ad["n_test"])
    mac("ADhStar", f(ad["h_star"], 3))
    mac("ADnHighLeverage", ad["n_high_leverage"])
    mac("ADnHighResidual", ad["n_high_standardised_residual"])
    mac("ADnInside", ad["n_inside_domain"])
    mac("ADPctInside", f(ad["pct_inside_domain"], 1))
    mac("ADRtwoInside", f(ad["metrics_inside_domain"]["r2"]))
    mac("ADRMSEInside", f(ad["metrics_inside_domain"]["rmse"]))
    mac("ADVarExplained", f(100 * ad["variance_explained"], 0))
    sim = sa["applicability_domain_similarity"]
    mac("SimSpearman", f(sim["spearman_similarity_vs_abs_error"], 3))
    mac("SimSpearmanP", pval(sim["spearman_p"]))
    # error above and below a nearest-neighbour similarity of 0.6: the bins of
    # the supplementary table are not strictly monotonic, so the text states the
    # contrast the data do support
    _nn = np.asarray(sim["nn_tanimoto"], float)
    _err = np.asarray(sim["abs_error"], float)
    _hi = _nn >= 0.6
    mac("SimRMSEHigh", f(float(np.sqrt(np.mean(_err[_hi] ** 2))), 3))
    mac("SimRMSELow", f(float(np.sqrt(np.mean(_err[~_hi] ** 2))), 3))
    mac("SimNHigh", int(_hi.sum()))
    mac("SimNLow", int((~_hi).sum()))
    write("tab_similarity_bins.tex", [
        f"{{{tex(b['nn_tanimoto_bin'])}}} & {b['n']} & {f(b['rmse'])} & {f(b['mae'])} & {f(b.get('r2'))} \\\\"
        for b in sim["bins"]])
    conf = sa["conformal_prediction"]
    write("tab_conformal.tex", [
        f"{tex(k)} & {f(d.get('half_width', d.get('mean_half_width')), 3)} & "
        f"{100 * d['target_coverage']:.0f} & {100 * d['empirical_coverage']:.1f} \\\\" for k, d in conf.items()])
    mac("ConfHalfWidth", f(conf["90%"]["half_width"], 3))
    mac("ConfCoverage", f(100 * conf["90%"]["empirical_coverage"], 1))
    write("tab_classwise.tex", [
        f"{tex(r['category'])} & {r['n']} & {f(r['mean_pIC50'], 2)} & {f(r.get('rmse'))} & "
        f"{f(r.get('mae'))} & {f(r.get('r2'))} \\\\" for r in sa["classwise"]])
    cw = [r for r in sa["classwise"] if r.get("rmse") is not None]
    mac("ClasswiseRMSEMin", f(min(r["rmse"] for r in cw)))
    mac("ClasswiseRMSEMax", f(max(r["rmse"] for r in cw)))


    # largest held-out errors of the reported model
    rp = load("c9_reported_model_predictions.json")
    cur = pd.read_csv(os.path.join(R1_OUT, "curated_dataset.csv"))
    te = np.asarray(rp["test_idx"], int)
    yt, pt = np.asarray(rp["y_test"], float), np.asarray(rp["test"], float)
    nn = np.asarray(sim["nn_tanimoto"], float)
    err = np.abs(yt - pt)
    rows = []
    for i in np.argsort(err)[::-1][:12]:
        smi = cur["canonical_smiles"].iloc[te[i]]
        smi = smi if len(smi) <= 46 else smi[:44] + "..."
        rows.append(f"\\texttt{{\\seqsplit{{{tex(smi)}}}}} & {yt[i]:.2f} & {pt[i]:.2f} & {err[i]:.2f} & "
                    f"{nn[i]:.2f} & {tex(cur['chemistry_category'].iloc[te[i]])} \\\\")
    write("tab_outliers.tex", rows)
    mac("MaxAbsError", f(float(err.max()), 2))
    mac("NErrorAboveOne", int((err > 1.0).sum()))

if pairs is not None and len(pairs):
    def lst(v):
        return eval(v) if isinstance(v, str) else v

    ref = pairs[pairs["role"] == "reproduced reference workflow"].iloc[0]
    mac("MatchedDeltaRtwo", f(ref["delta_r2"], 3))
    mac("MatchedDeltaCI", ci(lst(ref["delta_r2_ci95"])))
    mac("MatchedDeltaP", pval(ref["wilcoxon_p_bh"]))
    mac("MatchedDeltaModel", tex(ref["model_a"]))
    ft = pairs[pairs["role"].str.contains("Fine-tuned")].iloc[0]
    mac("BestVsFtDelta", f(ft["delta_r2"], 3))
    mac("BestVsFtCI", ci(lst(ft["delta_r2_ci95"])))
    mac("BestVsFtP", pval(ft["wilcoxon_p_bh"]))
    hy = pairs[pairs["role"].str.contains("Hybrid, ECFP4")].iloc[0]
    mac("BestVsHybDelta", f(hy["delta_r2"], 3))
    mac("BestVsHybCI", ci(lst(hy["delta_r2_ci95"])))
    sg = pairs[pairs["model_b"] == single.iloc[0]["model"]]
    if len(sg):
        mac("BestVsSingleDelta", f(sg.iloc[0]["delta_r2"], 3))
        mac("BestVsSingleCI", ci(lst(sg.iloc[0]["delta_r2_ci95"])))
    mac("NPairs", len(pairs))
    mac("NPairsSignificant", int((pairs["wilcoxon_p_bh"] < 0.05).sum()))
    rows = []
    for _, r in pairs.iterrows():
        d = lst(r["delta_r2_ci95"])
        label = r["role"].replace("family representative: ", "")
        label = {"reproduced reference workflow": "Reference workflow",
                 "stack ablation": "Stack without language-model members"}.get(label, label)
        rows.append(f"{label} & {tex(r['model_b'])} & {r['delta_r2']:+.3f} & [{d[0]:+.3f}, {d[1]:+.3f}] & "
                    f"{pval_cell(r['wilcoxon_abs_error_p'])} & {pval_cell(r['wilcoxon_p_bh'])} \\\\")
    write("tab_pairwise.tex", rows)

if "BestTestRtwo" in NEW:
    mac("FingerprintDeltaRtwo", f(float(NEW["BestTestRtwo"]) - float(NEW["BestFingerprintRtwo"]), 3))


# =========================================================================
# 3. matched molecular pairs under pair-aware splitting (c6)
# =========================================================================
mmp = load("c6_mmp.json")
if mmp:
    mac("NMMP", mmp["n_pairs"])
    mac("NMMPMolecules", mmp["n_molecules_in_pairs"])
    mac("NMMPComponents", mmp["n_connected_components"])
    mac("NMMPLargestComponent", mmp["largest_component_molecules"])
    mac("NMMPPhenolic", mmp["hydroxyl_type_counts"].get("phenolic", 0))
    mac("NMMPOther", mmp["n_pairs"] - mmp["hydroxyl_type_counts"].get("phenolic", 0))
    mac("NMMPSameScaffold", mmp["pairs_sharing_bemis_murcko_scaffold"])
    main_key = "RDKit-desc | ExtraTrees"
    pr = mmp["protocols"][main_key]
    mac("MMPModel", tex(main_key))
    for proto, tag in (("molecule_level", ""), ("component_aware", "Comp"), ("scaffold_aware", "Scaf"),
                       ("component_aware_same_pairs", "CompSub"), ("component_aware_phenolic_only", "CompPh")):
        if proto not in pr:
            continue
        s = pr[proto]
        mac(f"MMP{tag}ObsDelta", f(s["mean_observed_delta"], 3))
        mac(f"MMP{tag}PredDelta", f(s["mean_predicted_delta"], 3))
        mac(f"MMP{tag}PredDeltaCI", ci(s["mean_predicted_delta_ci95"]))
        mac(f"MMP{tag}Rho", f(s["spearman"], 3))
        mac(f"MMP{tag}RhoCI", ci(s["spearman_ci95"]))
        mac(f"MMP{tag}RhoP", pval(s["spearman_p"]))
        mac(f"MMP{tag}SignAgreement", f(s["sign_agreement_pct"], 0))
        mac(f"MMP{tag}SignCI", f"[{s['sign_agreement_ci95'][0]:.0f}, {s['sign_agreement_ci95'][1]:.0f}]")
        mac(f"MMP{tag}SignP", pval(s["sign_agreement_binomial_p"]))
        mac(f"MMP{tag}N", s["n_pairs"])
    comp = pr["component_aware"]
    mac("MMPCompMagnitudePct", f(100 * comp["mean_predicted_delta"] / comp["mean_observed_delta"], 0))
    ex = pr["molecule_level"]["exposure"]
    mac("MMPExposedPairsPct", f(ex["pct_pairs_with_any_partner_exposure"], 0))
    mac("MMPExposedPredPct", f(ex["pct_predictions_made_by_a_model_trained_on_the_partner"], 0))
    mac("MMPExposedBoth", ex["pairs_partner_seen_for_both_members"])
    mac("MMPExposedOne", ex["pairs_partner_seen_for_one_member"])
    mac("MMPExposedNone", ex["pairs_partner_seen_for_neither"])
    label = {"molecule_level": "Molecule-level (previous version)", "component_aware": "Pair-aware (connected components)",
             "scaffold_aware": "Scaffold-aware, pairs in the held-out scaffolds",
             "component_aware_same_pairs": "Pair-aware, the same pairs",
             "component_aware_phenolic_only": "Pair-aware, phenolic pairs only"}
    rows = []
    for key, protos in mmp["protocols"].items():
        for i, (proto, s) in enumerate(protos.items()):
            rows.append(
                f"{tex(key) if i == 0 else ''} & {label[proto]} & {s['n_pairs']} & {s['mean_predicted_delta']:+.3f} & "
                f"{s['spearman']:.3f} [{s['spearman_ci95'][0]:.3f}, {s['spearman_ci95'][1]:.3f}] & "
                f"{pval_cell(s['spearman_p'])} & {s['sign_agreement_pct']:.0f} "
                f"[{s['sign_agreement_ci95'][0]:.0f}, {s['sign_agreement_ci95'][1]:.0f}] & "
                f"{pval_cell(s['sign_agreement_binomial_p'])} \\\\")
        rows.append(r"\addlinespace")
    write("tab_mmp.tex", rows)


# =========================================================================
# 4. retrospective triage with intervals (c7)
# =========================================================================
tri = load("c7_triage.json")
if tri:
    mac("TriageThreshold", f(tri["threshold"], 2))
    rows, full = [], []
    for s in PRIMARY:
        t = tri["primary"][s]
        tag = STAG[s]
        mac(f"EFFive{tag}", f(t["ef_top5"], 2))
        mac(f"EFFive{tag}CI", ci(t["ef_top5_ci95"], 2))
        mac(f"EFTen{tag}", f(t["ef_top10"], 2))
        mac(f"EFTen{tag}CI", ci(t["ef_top10_ci95"], 2))
        mac(f"EFTwenty{tag}", f(t["ef_top20"], 2))
        mac(f"EFTwenty{tag}CI", ci(t["ef_top20_ci95"], 2))
        mac(f"HitTen{tag}", f(t["hit_rate_top10_pct"], 1))
        mac(f"BaseHit{tag}", f(t["base_rate_pct"], 1))
        mac(f"TriageRho{tag}", f(t["spearman"], 3))
        mac(f"TriageRho{tag}CI", ci(t["spearman_ci95"]))
        mac(f"TriageModel{tag}", tex(t["model"]))
        mac(f"TriageP{tag}", pval(t["p_hypergeom_top10"]))
        mac(f"TriageHits{tag}", t["n_hits_top10"])
        mac(f"TriageSelected{tag}", t["n_selected_top10"])
        rows.append(f"{PRETTY[s]} & {t['n_test']} & {t['n_potent']} & {t['base_rate_pct']:.1f} & "
                    f"{t['hit_rate_top10_pct']:.1f} & {t['ef_top5']:.2f} & {t['ef_top10']:.2f} & "
                    f"{t['ef_top20']:.2f} & {t['spearman']:.3f} \\\\")
        rows.append("& & & & & {\\scriptsize " + ci(t["ef_top5_ci95"], 2) + "} & {\\scriptsize "
                    + ci(t["ef_top10_ci95"], 2) + "} & {\\scriptsize " + ci(t["ef_top20_ci95"], 2)
                    + "} & {\\scriptsize " + ci(t["spearman_ci95"], 2) + "} \\\\")
        full.append(
            f"{PRETTY[s]} & {tex(t['model'])} & {t['n_hits_top5']}/{t['n_selected_top5']} & "
            f"{pval_cell(t['p_hypergeom_top5'])} & {t['n_hits_top10']}/{t['n_selected_top10']} & "
            f"{pval_cell(t['p_hypergeom_top10'])} & {t['n_hits_top20']}/{t['n_selected_top20']} & "
            f"{pval_cell(t['p_hypergeom_top20'])} & {t['ef_top10_max_possible']:.2f} \\\\")
    write("tab_triage.tex", rows)
    write("tab_triage_counts.tex", full)
    lo = [tri["primary"][s]["ef_top10_ci95"][0] for s in PRIMARY]
    mac("NTriageTenAboveOne", sum(v > 1 for v in lo))
    mac("NTriagePartitions", len(PRIMARY))
    models = sorted({tri["primary"][s]["model"] for s in PRIMARY})
    mac("NTriageModels", len(models))
    if "repeated" in tri:
        rows = []
        for d, tag in (("stratified", "Strat"), ("scaffold", "Scaf"), ("cluster", "Clus"), ("source", "Src")):
            if d not in tri["repeated"]:
                continue
            r = tri["repeated"][d]
            for frac, nm in (("ef_top5", "Five"), ("ef_top10", "Ten"), ("ef_top20", "Twenty")):
                mac(f"RepEF{nm}{tag}Mean", f(r[frac]["mean"], 2))
                mac(f"RepEF{nm}{tag}Min", f(r[frac]["min"], 2))
                mac(f"RepEF{nm}{tag}Max", f(r[frac]["max"], 2))
                mac(f"RepEF{nm}{tag}Above", r[frac]["n_folds_above_one"])
            mac(f"RepEFPooled{tag}", f(r["pooled_top10"]["enrichment_factor"], 2))
            mac(f"RepTriageRho{tag}", f(r["spearman"]["mean"], 3))
            p = r["pooled_top10"]
            rows.append(
                f"{PRETTY[d]} & {r['ef_top5']['mean']:.2f} ({r['ef_top5']['min']:.2f}--{r['ef_top5']['max']:.2f}) & "
                f"{r['ef_top10']['mean']:.2f} ({r['ef_top10']['min']:.2f}--{r['ef_top10']['max']:.2f}) & "
                f"{r['ef_top20']['mean']:.2f} ({r['ef_top20']['min']:.2f}--{r['ef_top20']['max']:.2f}) & "
                f"{r['ef_top10']['n_folds_above_one']}/{r['n_folds']} & "
                f"{p['n_hits']}/{p['n_selected']} & {p['base_rate_pct']:.1f} & {p['enrichment_factor']:.2f} & "
                f"{r['spearman']['mean']:.3f} \\\\")
        write("tab_triage_repeated.tex", rows)


# =========================================================================
# 5. repeated partitions (c8) and their characterisation (c1)
# =========================================================================
RD = (("stratified", "Strat"), ("scaffold", "Scaf"), ("cluster", "Clus"), ("source", "Src"))
chars = load("c1_partition_characterisation.csv")
if chars is not None:
    # five of the ten folds serve as repeated hold-outs (common.REP_FOLDS)
    chars = chars[chars["fold"].isin(REP_FOLDS)]
    rows = []
    for d, tag in RD:
        g = chars[chars["design"] == d]
        mac(f"Rep{tag}NNSim", f(g["nn_tanimoto_median"].mean(), 3))
        mac(f"Rep{tag}NNSimMin", f(g["nn_tanimoto_median"].min(), 3))
        mac(f"Rep{tag}NNSimMax", f(g["nn_tanimoto_median"].max(), 3))
        mac(f"Rep{tag}HighSim", f(g["pct_nn_ge_0.7"].mean(), 1))
        mac(f"Rep{tag}ScafSeen", f(g["pct_scaffold_seen"].mean(), 1))
        mac(f"Rep{tag}SDMin", f(g["test_sd_pIC50"].min(), 2))
        mac(f"Rep{tag}SDMax", f(g["test_sd_pIC50"].max(), 2))
        rows.append(
            f"{PRETTY[d]} & {int(g['n_test'].min())}--{int(g['n_test'].max())} & "
            f"{g['nn_tanimoto_median'].mean():.3f} ({g['nn_tanimoto_median'].min():.3f}--{g['nn_tanimoto_median'].max():.3f}) & "
            f"{g['pct_nn_ge_0.7'].mean():.1f} ({g['pct_nn_ge_0.7'].min():.1f}--{g['pct_nn_ge_0.7'].max():.1f}) & "
            f"{g['pct_scaffold_seen'].mean():.1f} ({g['pct_scaffold_seen'].min():.1f}--{g['pct_scaffold_seen'].max():.1f}) & "
            f"{g['test_mean_pIC50'].min():.2f}--{g['test_mean_pIC50'].max():.2f} & "
            f"{g['test_sd_pIC50'].min():.2f}--{g['test_sd_pIC50'].max():.2f} \\\\")
    write("tab_rep_partitions.tex", rows)
    mac("NRepFolds", len(REP_FOLDS))
    mac("NRepFoldsTotal", N_FOLDS)
    mac("NRepPartitions", len(chars))
    mac("NRepDesigns", chars["design"].nunique())
    mac("NRepGroupPartitions", int((chars["design"] != "stratified").sum()))

parts = load("c1_partitions.json")
if parts:
    mac("RepSeed", parts["seed"])
    mac("NButinaClusters", parts["n_groups"]["cluster"])

rep = load("c8_repeated_summary.json")
if rep:
    # negative values are written with a plain hyphen, as everywhere in the
    # macro file of the first revision, so that an unchanged number is not
    # reported as changed by the marked-up comparison
    def signed(v, n=3):
        return f"{v:.{n}f}"

    def sci(pair, n=3):
        return ci(pair, n)

    mac("NRepCandidates", len(rep["nested_candidates"]))
    mac("NRepLateCells", len(rep["late_cells"]))
    for mode, prefix in (("nested", "Rep"), ("fixed", "RepFx")):
        R = rep[mode]
        for d, tag in RD:
            D = R["designs"].get(d, {})
            if "families" not in D:
                continue
            mac(f"{prefix}{tag}NFolds", D["n_folds_all_families"])
            mac(f"{prefix}{tag}FriedmanP", pval(D["friedman"]["p"]))
            mac(f"{prefix}{tag}KendallW", f(D["kendall_w"], 2))
            mac(f"{prefix}{tag}NNeg", D["n_negative_r2"])
            mac(f"{prefix}{tag}NScored", D["n_scored"])
            mac(f"{prefix}{tag}AllNegFolds", D["folds_where_every_family_is_negative"])
            mac(f"{prefix}{tag}Leader", FAMILY_LABEL[D["ranking_by_mean_r2"][0]])
            mac(f"{prefix}{tag}Last", FAMILY_LABEL[D["ranking_by_mean_r2"][-1]])
            means = [D["families"][fam]["r2_mean"] for fam in FAM_KEYS]
            sds = [D["families"][fam]["r2_sd"] for fam in FAM_KEYS]
            rks = [D["families"][fam]["rank_mean"] for fam in FAM_KEYS]
            rhos = [D["families"][fam]["spearman_mean"] for fam in FAM_KEYS]
            allv = [v for fam in FAM_KEYS for v in D["families"][fam]["r2_by_fold"]]
            mac(f"{prefix}{tag}MeanMin", signed(min(means)))
            mac(f"{prefix}{tag}MeanMax", signed(max(means)))
            mac(f"{prefix}{tag}MeanSpread", f(max(means) - min(means), 3))
            mac(f"{prefix}{tag}SDMinFam", f(min(sds), 3))
            mac(f"{prefix}{tag}SDMaxFam", f(max(sds), 3))
            mac(f"{prefix}{tag}RankMin", f(min(rks), 1))
            mac(f"{prefix}{tag}RankMax", f(max(rks), 1))
            mac(f"{prefix}{tag}RhoMin", f(min(rhos), 2))
            mac(f"{prefix}{tag}RhoMax", f(max(rhos), 2))
            mac(f"{prefix}{tag}AllMin", signed(min(allv)))
            mac(f"{prefix}{tag}AllMax", signed(max(allv)))
            # a tie in mean rank names every family that shares it
            best_rank = min(D["families"][k]["rank_mean"] for k in FAM_KEYS)
            worst_rank = max(D["families"][k]["rank_mean"] for k in FAM_KEYS)
            mac(f"{prefix}{tag}BestRankFamily",
                " and ".join(PROSE[k] for k in FAM_KEYS if D["families"][k]["rank_mean"] == best_rank))
            mac(f"{prefix}{tag}WorstRankFamily",
                " and ".join(PROSE[k] for k in FAM_KEYS if D["families"][k]["rank_mean"] == worst_rank))
            mac(f"{prefix}{tag}BestRank", f(best_rank, 1))
            if "n_pooled_compounds" in D:
                mac(f"{prefix}{tag}NPooled", f"{D['n_pooled_compounds']:,}")
            for fam in FAM_KEYS:
                s = D["families"][fam]
                mac(f"{prefix}{tag}{fam}Mean", signed(s["r2_mean"]))
                mac(f"{prefix}{tag}{fam}SD", f(s["r2_sd"], 3))
                mac(f"{prefix}{tag}{fam}Min", signed(s["r2_min"]))
                mac(f"{prefix}{tag}{fam}Max", signed(s["r2_max"]))
                mac(f"{prefix}{tag}{fam}RMSE", f(s["rmse_mean"], 3))
                mac(f"{prefix}{tag}{fam}Rank", f(s["rank_mean"], 1))
                mac(f"{prefix}{tag}{fam}First", s["n_first"])
                mac(f"{prefix}{tag}{fam}Last", s["n_last"])
                mac(f"{prefix}{tag}{fam}RankOfMean", s["rank_of_mean_r2"])
                mac(f"{prefix}{tag}{fam}Rho", f(s["spearman_mean"], 2))
                if "pooled" in D:
                    mac(f"{prefix}{tag}{fam}Pooled", signed(D["pooled"][fam]["r2"]))
                    mac(f"{prefix}{tag}{fam}PooledCI", sci(D["pooled"][fam]["r2_ci95"]))
                    mac(f"{prefix}{tag}{fam}PoolRank", D["pooled"][fam]["rank"])
                    mac(f"{prefix}{tag}{fam}PoolRMSE", f(D["pooled"][fam]["rmse"], 3))
            if "same_encoder" in D:
                se = D["same_encoder"]
                c = se["fine_tuned_minus_frozen"]
                mac(f"{prefix}{tag}ZincFrozenMean", signed(se["frozen_chemberta_zinc_mean_r2"]))
                mac(f"{prefix}{tag}FtVsZincWins", c["n_a_better"])
                mac(f"{prefix}{tag}FtVsZincDelta", signed(c["mean_delta"]))
                mac(f"{prefix}{tag}FtVsZincCI", sci(c["ci95"]))
            for name, c in D["paired"].items():
                key = name.replace("-", "")
                mac(f"{prefix}{tag}{key}Wins", c["n_a_better"])
                mac(f"{prefix}{tag}{key}Delta", signed(c["mean_delta"]))
                mac(f"{prefix}{tag}{key}CI", sci(c["ci95"]))
                mac(f"{prefix}{tag}{key}P", pval(c["p_wilcoxon"]))
            for name, c in D.get("pooled_paired", {}).items():
                key = name.replace("-", "")
                mac(f"{prefix}{tag}{key}PoolDelta", signed(c["delta_r2"]))
                mac(f"{prefix}{tag}{key}PoolCI", sci(c["ci95"]))
                mac(f"{prefix}{tag}{key}PoolP", pboot(c["p_bootstrap_two_sided"]))
        for d, tag in RD[1:]:
            for name, c in R["interaction_vs_stratified"].get(d, {}).items():
                key = name.replace("-", "")
                mac(f"{prefix}Int{tag}{key}", signed(c["interaction"]))
                mac(f"{prefix}Int{tag}{key}CI", sci(c["ci95"]))
                mac(f"{prefix}Int{tag}{key}P", pboot(c["p_bootstrap_two_sided"]))

    N = rep["nested"]["designs"]
    done = [d for d, _ in RD if "families" in N.get(d, {})]
    if done:
        # main-text table, block (b): whole groups of any size held out
        rows = []
        for fam, label in FAMILIES:
            cells = []
            for d, _ in RD[1:]:
                if d in done:
                    s = N[d]["families"][fam]
                    cells += [f"{s['r2_mean']:.3f} $\\pm$ {s['r2_sd']:.3f}", f"{s['rank_mean']:.1f}"]
                else:
                    cells += ["--", "--"]
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        write("tab_repeated_b.tex", rows)

        # SI: every fold
        rows = []
        for d in done:
            D = N[d]
            for i, k in enumerate(D["folds_used"]):
                vals = [D["families"][fam]["r2_by_fold"][i] for fam in FAM_KEYS]
                best_i = int(np.argmax(vals))
                cells = [(f"\\textbf{{{v:.3f}}}" if j == best_i else f"{v:.3f}") for j, v in enumerate(vals)]
                rows.append(f"{PRETTY[d] if i == 0 else ''} & {k + 1} & " + " & ".join(cells) + r" \\")
            rows.append("\\cmidrule(lr){2-9}")
            rows.append(" & mean & " + " & ".join(f"{D['families'][fam]['r2_mean']:.3f}" for fam in FAM_KEYS) + r" \\")
            rows.append(" & s.d. & " + " & ".join(f"{D['families'][fam]['r2_sd']:.3f}" for fam in FAM_KEYS) + r" \\")
            rows.append(" & mean rank & " + " & ".join(f"{D['families'][fam]['rank_mean']:.1f}" for fam in FAM_KEYS) + r" \\")
            rows.append(" & ranked first & " + " & ".join(str(D["families"][fam]["n_first"]) for fam in FAM_KEYS) + r" \\")
            if "pooled" in D:
                rows.append(" & pooled \\rsq{} & " + " & ".join(f"{D['pooled'][fam]['r2']:.3f}" for fam in FAM_KEYS) + r" \\")
            rows.append(r"\addlinespace")
        write("tab_rep_folds.tex", rows)

        # SI: paired contrasts across partitions and pooled
        rows = []
        for name in ("Frozen-Hyb", "Frozen-Fp", "Frozen-Ft", "Frozen-Desc", "Frozen-Gnn", "Hyb-Fp"):
            a, b = name.split("-")
            for i, d in enumerate(done):
                c = N[d]["paired"][name]
                pp = N[d].get("pooled_paired", {}).get(name)
                pooled = (f"{pp['delta_r2']:+.3f} [{pp['ci95'][0]:+.3f}, {pp['ci95'][1]:+.3f}]" if pp else "--")
                rows.append(
                    f"{FAMILY_LABEL[a] + ' minus ' + FAMILY_LABEL[b] if i == 0 else ''} & {PRETTY[d]} & "
                    f"{c['n_a_better']}/{c['n_folds']} & {c['mean_delta']:+.3f} "
                    f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] & {pval_cell(c['p_wilcoxon'])} & {pooled} \\\\")
            rows.append(r"\addlinespace")
        write("tab_rep_contrasts.tex", rows)

        # SI: interaction with the stratified design
        rows = []
        inter = rep["nested"]["interaction_vs_stratified"]
        for name in ("Frozen-Hyb", "Frozen-Fp", "Frozen-Ft", "Frozen-Desc", "Hyb-Fp"):
            a, b = name.split("-")
            first = True
            for d in done[1:]:
                if d not in inter:
                    continue
                c = inter[d][name]
                rows.append(
                    f"{FAMILY_LABEL[a] + ' minus ' + FAMILY_LABEL[b] if first else ''} & {PRETTY[d]} & "
                    f"{c['gap_under_stratified']:+.3f} & {c['gap_under_design']:+.3f} & "
                    f"{c['interaction']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] & "
                    f"{pboot_cell(c['p_bootstrap_two_sided'])} \\\\")
                first = False
            rows.append(r"\addlinespace")
        write("tab_rep_interaction.tex", rows)

        # SI: which cell nested selection chose, and the fixed-representative results
        rows = []
        F = rep["fixed"]["designs"]
        for fam, label in FAMILIES:
            for i, d in enumerate(done):
                s = N[d]["families"][fam]
                chosen = "; ".join(f"{tex(k)} ({v})" for k, v in s["selected_models"].items())
                fx = F.get(d, {}).get("families", {}).get(fam)
                fixed = f"{fx['r2_mean']:.3f} $\\pm$ {fx['r2_sd']:.3f}" if fx else "--"
                rows.append(f"{label if i == 0 else ''} & {PRETTY[d]} & {chosen} & "
                            f"{s['r2_mean']:.3f} $\\pm$ {s['r2_sd']:.3f} & {fixed} \\\\")
            rows.append(r"\addlinespace")
        write("tab_rep_selection.tex", rows)

    nf = rep.get("nested_vs_fixed")
    if nf:
        mac("RepSameCell", nf["n_same_cell"])
        mac("RepSameCellOf", nf["n_family_partition_combinations"])
        mac("RepSameCellPct", f(100 * nf["n_same_cell"] / nf["n_family_partition_combinations"], 0))
    late_n = late_win = 0
    for d in done:
        for fam in ("Fp", "Desc", "Frozen", "Hyb", "HybClm"):
            s = N[d]["families"][fam]
            late_n += s.get("late_cells_checked_in_folds", 0)
            late_win += s.get("late_cell_would_have_been_selected", 0)
    mac("RepLateChecked", late_n)
    mac("RepLateWouldWin", late_win)


# ------------- do the second-revision workers reproduce first-revision values? (c0)
repro = load("c0_reproduce_primary.json")
if repro:
    if repro.get("grid"):
        mac("ReproGridCells", len(repro["grid"]))
        worst = max(r["abs_difference"] for r in repro["grid"])
        assert worst < 1e-9, worst
    if repro.get("gnn"):
        mac("ReproGnnMaxDiff", f(max(r["abs_difference"] for r in repro["gnn"]), 3))
        mac("ReproGnnPartitions", len(repro["gnn"]))


# ------------------- the rule of the primary partitions, repeated (c1b, c8)
SR = [("scaffold", "Scaf"), ("cluster", "Clus")]
sr_chars = load("c1b_same_rule_characterisation.csv")
if sr_chars is not None:
    rows = []
    for d, tag in SR + [("source", "Src")]:
        g = sr_chars[sr_chars["design"] == d]
        mac(f"Sr{tag}SharedPrimary", f(g["pct_shared_with_primary"].mean(), 0))
        mac(f"Sr{tag}SharedPrimaryMin", f(g["pct_shared_with_primary"].min(), 0))
        mac(f"Sr{tag}SharedPrimaryMax", f(g["pct_shared_with_primary"].max(), 0))
        if d == "source":
            continue
        mac(f"Sr{tag}NNSim", f(g["nn_tanimoto_median"].mean(), 3))
        mac(f"Sr{tag}NNSimMin", f(g["nn_tanimoto_median"].min(), 3))
        mac(f"Sr{tag}NNSimMax", f(g["nn_tanimoto_median"].max(), 3))
        mac(f"Sr{tag}ScafSeen", f(g["pct_scaffold_seen"].mean(), 1))
        mac(f"Sr{tag}LargestGroup", int(g["largest_test_group"].max()))
        mac(f"Sr{tag}NTest", int(g["n_test"].max()))
        rows.append(
            f"{PRETTY[d]} & {int(g['n_test'].min())} & {int(g['largest_test_group'].max())} & "
            f"{g['nn_tanimoto_median'].mean():.3f} ({g['nn_tanimoto_median'].min():.3f}--{g['nn_tanimoto_median'].max():.3f}) & "
            f"{g['pct_scaffold_seen'].mean():.1f} ({g['pct_scaffold_seen'].min():.1f}--{g['pct_scaffold_seen'].max():.1f}) & "
            f"{g['test_sd_pIC50'].min():.2f}--{g['test_sd_pIC50'].max():.2f} & "
            f"{g['pct_shared_with_primary'].mean():.0f} ({g['pct_shared_with_primary'].min():.0f}--{g['pct_shared_with_primary'].max():.0f}) \\\\")
    write("tab_sr_partitions.tex", rows)

if parts and "same_rule" in parts:
    ov = parts["same_rule"]["overlap"]
    sizes = pd.Series(parts["butina_cluster_of_compound"]).value_counts()
    mac("NSingletonClusters", int((sizes == 1).sum()))
    mac("NSrSeeds", len(parts["same_rule"]["seeds"]))
    mac("NSrPartitions", len(parts["same_rule"]["seeds"]) * len(parts["same_rule"]["designs_replicated"]))
    for d, tag in SR + [("source", "Src")]:
        mac(f"Sr{tag}SharedRepeats", f(ov[d]["mean_pct_shared_between_two_repeats"], 0))
        mac(f"Sr{tag}EverHeldOut", ov[d]["n_compounds_ever_held_out"])

if rep and "same_rule" in rep:
    sel_primary = load("c5_selection_primary.json")
    for mode, prefix in (("nested", "Sr"), ("fixed", "SrFx")):
        R = rep["same_rule"].get(mode, {}).get("designs", {})
        for d, tag in SR:
            D = R.get(d, {})
            if "families" not in D:
                continue
            mac(f"{prefix}{tag}NPartitions", D["n_partitions"])
            mac(f"{prefix}{tag}KendallW", f(D["kendall_w"], 2))
            mac(f"{prefix}{tag}Leader", FAMILY_LABEL[D["ranking_by_mean_r2"][0]])
            means = [D["families"][k]["r2_mean"] for k in FAM_KEYS]
            mac(f"{prefix}{tag}MeanMin", f"{min(means):.3f}")
            mac(f"{prefix}{tag}MeanMax", f"{max(means):.3f}")
            best_rank = min(D["families"][k]["rank_mean"] for k in FAM_KEYS)
            worst_rank = max(D["families"][k]["rank_mean"] for k in FAM_KEYS)
            mac(f"{prefix}{tag}BestRankFamily",
                " and ".join(PROSE[k] for k in FAM_KEYS if D["families"][k]["rank_mean"] == best_rank))
            mac(f"{prefix}{tag}WorstRankFamily",
                " and ".join(PROSE[k] for k in FAM_KEYS if D["families"][k]["rank_mean"] == worst_rank))
            for fam in FAM_KEYS:
                st = D["families"][fam]
                mac(f"{prefix}{tag}{fam}Mean", f"{st['r2_mean']:.3f}")
                mac(f"{prefix}{tag}{fam}SD", f(st["r2_sd"], 3))
                mac(f"{prefix}{tag}{fam}Min", f"{st['r2_min']:.3f}")
                mac(f"{prefix}{tag}{fam}Max", f"{st['r2_max']:.3f}")
                mac(f"{prefix}{tag}{fam}Rank", f(st["rank_mean"], 1))
                mac(f"{prefix}{tag}{fam}First", st["n_first"])
                mac(f"{prefix}{tag}{fam}TopThree", st["n_top3"])
                mac(f"{prefix}{tag}{fam}Last", st["n_last"])
                mac(f"{prefix}{tag}{fam}RankOfMean", st["rank_of_mean_r2"])
            for name, c in D["paired"].items():
                key = name.replace("-", "")
                mac(f"{prefix}{tag}{key}Wins", c["n_a_better"])
                mac(f"{prefix}{tag}{key}Delta", f"{c['mean_delta']:.3f}")
                mac(f"{prefix}{tag}{key}DeltaMin", f"{c['min_delta']:.3f}")
                mac(f"{prefix}{tag}{key}DeltaMax", f"{c['max_delta']:.3f}")
                mac(f"{prefix}{tag}{key}CI", ci(c["ci95"]))
                mac(f"{prefix}{tag}{key}P", pboot(c["p_bootstrap_two_sided"]))
            for name, c in D.get("interaction_vs_stratified", {}).items():
                key = name.replace("-", "")
                mac(f"{prefix}Int{tag}{key}", f"{c['interaction']:.3f}")
                mac(f"{prefix}Int{tag}{key}CI", ci(c["ci95"]))
                mac(f"{prefix}Int{tag}{key}P", pboot(c["p_bootstrap_two_sided"]))
                mac(f"{prefix}Int{tag}{key}StratGap", f"{c['gap_under_stratified']:.3f}")

    nf = rep["same_rule"].get("nested_vs_fixed")
    if nf:
        mac("SrSameCell", nf["n_same_cell"])
        mac("SrSameCellOf", nf["n_family_partition_combinations"])

    # totals over the replicates of both designs, for the sentences that summarise them
    R = rep["same_rule"].get("nested", {}).get("designs", {})
    if all("families" in R.get(d, {}) for d, _ in SR):
        n_all = sum(R[d]["n_partitions"] for d, _ in SR)
        mac("NSrDone", n_all)
        mac("NRepAllPartitions", n_all + sum(D.get("n_folds_all_families", 0)
                                             for D in rep["nested"]["designs"].values()))
        for name in ("Frozen-Fp", "Frozen-Hyb", "Frozen-Desc", "Frozen-Ft", "Frozen-Gnn"):
            mac(f"Sr{name.replace('-', '')}WinsTotal", sum(R[d]["paired"][name]["n_a_better"] for d, _ in SR))
        for d, tag in SR:
            ranks = R[d]["families"]["Frozen"]["rank_by_partition"]
            mac(f"Sr{tag}FrozenTopTwo", sum(r <= 2 for r in ranks))
            mac(f"Sr{tag}FrozenWorstRank", max(ranks))
            mac(f"Sr{tag}FrozenRankOfRank", R[d]["families"]["Frozen"]["rank_of_mean_rank"])
        mac("SrFrozenTopTwoTotal", sum(r <= 2 for d, _ in SR for r in R[d]["families"]["Frozen"]["rank_by_partition"]))
        mac("SrFrozenFirstTotal", sum(R[d]["families"]["Frozen"]["n_first"] for d, _ in SR))

    S = rep["same_rule"].get("nested", {}).get("designs", {})
    sr_done = [d for d, _ in SR if "families" in S.get(d, {})]
    if sr_done and sel_primary:
        P = sel_primary["partitions"]
        # main-text table, block (a): the rule of each primary partition repeated
        strat = rep["nested"]["designs"].get("stratified", {}).get("families")
        rows = []
        for fam, label in FAMILIES:
            cells = []
            for src in [strat] + [S[d]["families"] if d in sr_done else None for d, _ in SR]:
                if src:
                    st = src[fam]
                    cells += [f"{st['r2_mean']:.3f} $\\pm$ {st['r2_sd']:.3f}", f"{st['rank_mean']:.1f}"]
                else:
                    cells += ["--", "--"]
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        write("tab_repeated_a.tex", rows)

        # SI: every replicate
        rows = []
        for d in sr_done:
            D = S[d]
            prim = [P[d]["families"][fam]["test_r2"] for fam in FAM_KEYS]
            bi = int(np.argmax(prim))
            rows.append(f"{PRETTY[d]} & primary & " + " & ".join(
                (f"\\textbf{{{v:.3f}}}" if j == bi else f"{v:.3f}") for j, v in enumerate(prim)) + r" \\")
            for i, pid in enumerate(D["partitions_used"]):
                vals = [D["families"][fam]["r2_by_partition"][i] for fam in FAM_KEYS]
                bi = int(np.argmax(vals))
                rows.append(f" & {int(pid[-2:])} & " + " & ".join(
                    (f"\\textbf{{{v:.3f}}}" if j == bi else f"{v:.3f}") for j, v in enumerate(vals)) + r" \\")
            rows.append("\\cmidrule(lr){2-9}")
            rows.append(" & mean & " + " & ".join(f"{D['families'][fam]['r2_mean']:.3f}" for fam in FAM_KEYS) + r" \\")
            rows.append(" & s.d. & " + " & ".join(f"{D['families'][fam]['r2_sd']:.3f}" for fam in FAM_KEYS) + r" \\")
            rows.append(" & mean rank & " + " & ".join(f"{D['families'][fam]['rank_mean']:.1f}" for fam in FAM_KEYS) + r" \\")
            rows.append(" & ranked first & " + " & ".join(str(D["families"][fam]["n_first"]) for fam in FAM_KEYS) + r" \\")
            rows.append(r"\addlinespace")
        write("tab_sr_folds.tex", rows)

        # SI: contrasts on the replicates, and their change from the stratified design
        rows = []
        for name in ("Frozen-Hyb", "Frozen-Fp", "Frozen-Desc", "Frozen-Gnn", "Frozen-Ft", "Hyb-Fp"):
            a, b = name.split("-")
            for i, d in enumerate(sr_done):
                c = S[d]["paired"][name]
                it = S[d].get("interaction_vs_stratified", {}).get(name)
                change = (f"{it['interaction']:+.3f} [{it['ci95'][0]:+.3f}, {it['ci95'][1]:+.3f}]" if it else "--")
                prim = P[d]["families"][a]["test_r2"] - P[d]["families"][b]["test_r2"]
                rows.append(
                    f"{FAMILY_LABEL[a] + ' minus ' + FAMILY_LABEL[b] if i == 0 else ''} & {PRETTY[d]} & "
                    f"{prim:+.3f} & {c['n_a_better']}/{c['n_partitions']} & "
                    f"{c['mean_delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] & "
                    f"{pboot_cell(c['p_bootstrap_two_sided'])} & {change} \\\\")
            rows.append(r"\addlinespace")
        write("tab_sr_contrasts.tex", rows)
    else:
        # the replicates are not complete yet: keep the documents compilable
        print("NOTE: replicates of the primary rule are incomplete; placeholder table bodies written")
        write("tab_repeated_a.tex", [f"{label} & -- & -- & -- & -- & -- & -- \\\\" for _, label in FAMILIES])
        write("tab_sr_folds.tex", ["pending & & & & & & & & \\\\"])
        write("tab_sr_contrasts.tex", ["pending & & & & & & \\\\"])


# ------------------------------------------- protect row-leading brackets
for name in sorted(os.listdir(GEN)):
    if not (name.startswith("tab_") and name.endswith(".tex")):
        continue
    path = os.path.join(GEN, name)
    lines = open(path, encoding="utf-8").read().split("\n")
    fixed = []
    for line in lines:
        if line.startswith("["):
            head, sep, rest = line.partition("&")
            fixed.append("{" + head.strip() + "} " + sep + rest)
        else:
            fixed.append(line)
    if fixed != lines:
        open(path, "w", encoding="utf-8", newline="\n").write("\n".join(fixed))

# ---------------------------------------------------------------- write, log
changed = {k: (MACROS[k], v) for k, v in NEW.items() if k in MACROS and MACROS[k] != v}
added = {k: v for k, v in NEW.items() if k not in MACROS}
MACROS.update(NEW)

used = set()
for name in os.listdir(PKG):
    if name.endswith(".tex") and name != "manuscript_marked.tex":
        used |= set(re.findall(r"\\([A-Z][A-Za-z]{3,})\{\}", open(os.path.join(PKG, name), encoding="utf-8").read()))
elsewhere = set()
for rel in ("si_numbers.tex", "repository.tex", "generated/crossrefs.tex", "generated/released_model.tex",
            "preamble.tex", "author_metadata.tex", "paper_title.tex"):
    p = os.path.join(PKG, rel.replace("/", os.sep))
    if os.path.exists(p):
        elsewhere |= set(re.findall(r"\\(?:re|provide)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?", open(p, encoding="utf-8").read()))
        elsewhere |= set(re.findall(r"\\providecommand\*?\s*\{?\\([A-Za-z]+)\}?", open(p, encoding="utf-8").read()))
missing = sorted(used - set(MACROS) - elsewhere - {"TeX", "LaTeX"})

with open(os.path.join(GEN, "results_macros.tex"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("% Auto-generated by d1_macros.py (second revision) -- do not edit by hand.\n")
    for k in sorted(MACROS):
        # the sign of a negative number is boxed so that it cannot be left at
        # the end of a line; the marked-up comparison does the same to the
        # values of the previous version, so an unchanged number stays unmarked
        fh.write(f"\\newcommand*{{\\{k}}}{{{unbreakable_minus(MACROS[k])}}}\n")
    if missing:
        fh.write("\n% Referenced in the text but not yet produced by the analysis.\n")
        for k in missing:
            fh.write(f"\\newcommand*{{\\{k}}}{{\\textbf{{??{k}??}}}}\n")

with open(os.path.join(OUT, "d1_macro_changes.txt"), "w", encoding="utf-8") as fh:
    fh.write("Macros whose value changed relative to the first revision (old -> new)\n")
    for k in sorted(changed):
        fh.write(f"  {k:28s} {changed[k][0]}  ->  {changed[k][1]}\n")
    fh.write(f"\n{len(added)} new macros\n")
    for k in sorted(added):
        fh.write(f"  {k:36s} {added[k]}\n")

print(f"wrote {len(MACROS)} macros: {len(changed)} changed, {len(added)} new")
for k in sorted(changed):
    print(f"  {k:28s} {changed[k][0]}  ->  {changed[k][1]}")
if missing:
    print(f"\n{len(missing)} macros referenced in the documents are still missing:")
    for k in missing:
        print("  ??", k)
else:
    print("\nevery macro referenced in the documents is available")
