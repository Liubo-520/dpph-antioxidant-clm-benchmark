"""Turn the analysis outputs into LaTeX macros and table bodies.

Every number that appears in the revised manuscript, the supplementary
information or the response letter is emitted from here, so the documents cannot
drift away from the computed results.
"""

import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
PKG = os.path.join(ROOT, "revision", "Revision_R1_new")
GEN = os.path.join(PKG, "generated")
os.makedirs(GEN, exist_ok=True)

MACROS = {}


def mac(name, value):
    MACROS[name] = value


def f(x, n=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def ci(pair, n=3):
    return f"[{pair[0]:.{n}f}, {pair[1]:.{n}f}]"


def pval(p):
    """A p-value carrying its own relational operator.

    A value below the reporting resolution is an inequality, so the operator
    belongs to the macro rather than to the sentence: written as "$p$ = \\FaithP{}"
    behind a bare "=", such a value typesets as "p = <0.0001". Every call site
    writes "$p$ \\FaithP{}" and the macro supplies "= 0.0031" or "$<$ 0.0001".
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "$<$ 0.0001"
    return "= " + f"{p:.4f}".rstrip("0").rstrip(".")


def pval_cell(p):
    """The same value for a table cell, where the column header carries the name.

    "$p$ = 0.0066" belongs in a sentence; a column of "= 0.0066" does not.
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}".rstrip("0").rstrip(".")


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    if name.endswith(".json"):
        return json.load(open(path, encoding="utf-8"))
    return pd.read_csv(path)


LATEX_ESCAPE = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}


def tex(s):
    s = str(s)
    for k, v in LATEX_ESCAPE.items():
        s = s.replace(k, v)
    # model keys are "<representation> | <learner>"; a bare pipe is not text in
    # every font encoding, and reads better as a join anyway
    return s.replace(" | ", " + ")


PRETTY = {
    "grad_x_input": "gradient $\\times$ input",
    "integrated_gradients": "integrated gradients",
    "occlusion": "occlusion",
    "attention_rollout": "attention rollout",
    "phenolic_OH_oxygen": "phenolic O--H oxygen",
    "phenolic_OH_and_ipso_carbon": "phenolic O--H + ipso carbon",
    "catechol_motif": "catechol motif",
    "aromatic": "aromatic atoms",
    "non_aromatic": "non-aromatic atoms",
    "carbonyl": "carbonyl atoms",
    "conjugated_CC": "conjugated C=C--C=C",
    "aliphatic_OH_oxygen": "aliphatic O--H oxygen",
    "stratified": "Chemistry-stratified",
    "scaffold": "Scaffold-disjoint",
    "cluster": "Cluster-disjoint",
    "dissimilarity": "Maximum-dissimilarity",
    "source": "Source-disjoint",
}


def pretty(k):
    return PRETTY.get(k, tex(k))


# ------------------------------------------------------------------ a1 dataset
audit = load("a1_dataset_audit.json")
if audit:
    p, e = audit["provenance"], audit["endpoint"]
    mac("NCompounds", f"{p['n_records']:,}")
    mac("NSourcePublications", str(p["n_unique_source_publications"]))
    mac("NAssayDescriptions", str(p["n_unique_assay_descriptions"]))
    mac("PctLargestSource", f"{p['share_from_largest_source_pct']:.1f}")
    mac("PctTopTenSources", f"{p['share_from_top10_sources_pct']:.1f}")
    mac("MedianRecordsPerSource", f"{p['records_per_source_median']:.0f}")
    mac("MaxRecordsPerSource", str(p["records_per_source_max"]))
    mac("PicMean", f(e["pIC50_mean"], 3))
    mac("PicSD", f(e["pIC50_sd"], 3))
    mac("PicMedian", f(e["pIC50_median"], 3))
    mac("PicMin", f(e["pIC50_min"], 2))
    mac("PicMax", f(e["pIC50_max"], 2))
    mac("SkewIC", f(e["skew_ic50"], 1))
    mac("SkewPIC", f(e["skew_pic50"], 2))
    mac("IcMedianMicromolar", f(e["ic50_micromolar_median"], 1))
    mac("IcOrdersOfMagnitude", f(e["ic50_log_range_orders"], 1))
    mac("MwMedian", f(e["mw_median"], 0))
    c = audit["category_label_check"]
    mac("CategoryAnovaF", f(c["anova_F"], 1))
    mac("CategoryAnovaP", pval(c["anova_p"]))
    mac("CategoryEtaSq", f(c["eta_squared"], 3))
    mac("CategoryEtaSqPct", f(100 * c["eta_squared"], 1))
    s = audit["split_endpoint_distribution"]
    mac("NTrain", f"{s['n_train']:,}")
    mac("NTest", str(s["n_test"]))
    mac("TrainMeanPic", f(s["train_mean"], 3))
    mac("TestMeanPic", f(s["test_mean"], 3))
    mac("TrainSDPic", f(s["train_sd"], 3))
    mac("TestSDPic", f(s["test_sd"], 3))
    mac("SplitKSP", pval(s["ks_p"]))
    mac("SplitKSStat", f(s["ks_statistic"], 3))
    mac("SplitLeveneP", pval(s["levene_p"]))
    sc = audit["scaffold_inventory"]
    mac("NScaffolds", str(sc["n_unique_bemis_murcko_scaffolds"]))
    mac("NSingletonScaffolds", str(sc["n_singleton_scaffolds"]))
    mac("LargestScaffoldSize", str(sc["largest_scaffold_size"]))
    mac("NDuplicateGroups", "0")

    # Table S1: provenance summary
    rows = [
        ("Curated compounds", f"{p['n_records']:,}"),
        ("Unique ChEMBL compound identifiers", f"{p['n_unique_chembl_ids']:,}"),
        ("Unique primary publications (DOI)", str(p["n_unique_source_publications"])),
        ("Median (IQR) records per publication",
         f"{p['records_per_source_median']:.0f} ({p['records_per_source_iqr'][0]:.0f}--{p['records_per_source_iqr'][1]:.0f})"),
        ("Largest single publication", f"{p['records_per_source_max']} compounds ({p['share_from_largest_source_pct']:.1f}\\%)"),
        ("Ten largest publications combined", f"{p['share_from_top10_sources_pct']:.1f}\\%"),
        ("Publications contributing one compound", str(p["n_sources_with_single_record"])),
        ("Distinct free-text assay descriptions", str(p["n_unique_assay_descriptions"])),
        ("Assay descriptions mentioning DPPH", f"{100 * p['fraction_assay_text_mentions_DPPH']:.0f}\\%"),
        # "specifying 30 min" reads as a single timepoint and contradicts the
        # window wordings listed two tables later; every description carries a
        # 30-min readout, but a minority states it as a range
        ("Assay descriptions assigned to the nominal 30-min endpoint",
         f"{100 * p['fraction_assay_text_mentions_30_min']:.0f}\\%"),
        ("Median IC$_{50}$", f"{e['ic50_micromolar_median']:.1f} $\\mu$M"),
        ("IC$_{50}$ range", f"{e['ic50_log_range_orders']:.1f} orders of magnitude"),
        ("Median molecular weight", f"{e['mw_median']:.0f} Da"),
        ("Skewness, IC$_{50}$ / \\pic{}", f"{e['skew_ic50']:.1f} / {e['skew_pic50']:.2f}"),
        ("Unique Bemis--Murcko scaffolds", str(sc["n_unique_bemis_murcko_scaffolds"])),
        ("Singleton scaffolds", str(sc["n_singleton_scaffolds"])),
    ]
    with open(os.path.join(GEN, "tab_provenance.tex"), "w", encoding="utf-8") as fh:
        for k, v in rows:
            fh.write(f"{k} & {v} \\\\\n")

    # Table S2: duplicate audit
    with open(os.path.join(GEN, "tab_duplicates.tex"), "w", encoding="utf-8") as fh:
        for key, d in audit["duplicates"].items():
            fh.write(
                f"{tex(d['level'])} & {d['n_unique']:,} & {d['n_duplicate_groups']} & "
                f"{d['n_records_in_duplicate_groups']} & {d['n_groups_spread_gt_1_log']} \\\\\n"
            )

    # Table S4: category SMARTS definitions
    with open(os.path.join(GEN, "tab_category_rules.tex"), "w", encoding="utf-8") as fh:
        for i, entry in enumerate(audit["category_definitions"], start=1):
            pats = " \\newline ".join(f"\\texttt{{{tex(s)}}}" for s in entry["smarts"])
            fh.write(f"{i} & {tex(entry['category'])} & {pats} \\\\\n")

    # Table S5: category statistics
    with open(os.path.join(GEN, "tab_categories.tex"), "w", encoding="utf-8") as fh:
        for r in sorted(audit["categories"], key=lambda z: -z["n_total"]):
            fh.write(
                f"{tex(r['category'])} & {r['n_total']} & {r['pct_total']:.1f} & {r['n_train']} & "
                f"{r['n_test']} & {r['n_source_publications']} & {r['mean_pIC50']:.2f} & "
                f"{r['sd_pIC50']:.2f} \\\\\n"
            )

    by_cat = {r["category"]: r for r in audit["categories"]}
    if "Other" in by_cat:
        o = by_cat["Other"]
        mac("NOther", str(o["n_total"]))
        mac("NOtherPct", f(o["pct_total"], 0))
        mac("NOtherSources", str(o["n_source_publications"]))
        mac("OtherSD", f(o["sd_pIC50"], 2))
    if "Aromatic amine" in by_cat:
        mac("NAromaticAminePct", f(by_cat["Aromatic amine"]["pct_total"], 0))

    # Table S3: most frequent assay descriptions
    counts = audit["provenance"]["assay_description_counts"]
    with open(os.path.join(GEN, "tab_assay_text.tex"), "w", encoding="utf-8") as fh:
        for k, v in list(counts.items())[:12]:
            fh.write(f"{tex(k)} & {v} \\\\\n")

# -------------------------------------------------------------------- a4 splits
splits = load("a4_split_characterisation.csv")
if splits is not None:
    splits = splits.set_index("split")
    with open(os.path.join(GEN, "tab_splits.tex"), "w", encoding="utf-8") as fh:
        for name in ("stratified", "scaffold", "cluster", "dissimilarity", "source"):
            if name not in splits.index:
                continue
            r = splits.loc[name]
            # mean and the >= 0.4 share are printed as well: the response letter
            # tells the reviewers this table reports them, and they were computed
            # all along but did not reach the table
            fh.write(
                f"{pretty(name)} & {int(r['n_train']):,} & {int(r['n_test'])} & "
                f"{r['test_nn_tanimoto_mean']:.3f} & "
                f"{r['test_nn_tanimoto_median']:.3f} & {r['test_nn_tanimoto_p10']:.3f} & "
                f"{r['pct_test_with_nn_sim_ge_0.4']:.1f} & "
                f"{r['pct_test_with_nn_sim_ge_0.7']:.1f} & "
                f"{r['pct_test_scaffold_seen_in_training']:.1f} & "
                f"{int(r['n_test_unique_scaffolds'])} \\\\\n"
            )
    for name, tag in (("stratified", "Strat"), ("scaffold", "Scaf"), ("cluster", "Clus"),
                      ("dissimilarity", "Diss"), ("source", "Src")):
        if name in splits.index:
            r = splits.loc[name]
            mac(f"NNSim{tag}", f(r["test_nn_tanimoto_median"], 3))
            mac(f"ScafSeen{tag}", f(r["pct_test_scaffold_seen_in_training"], 1))
            mac(f"HighSim{tag}", f(r["pct_test_with_nn_sim_ge_0.7"], 1))

# ------------------------------------------------------------ a2 representations
reps = load("a2_representations.json")
if reps:
    dims = reps["dimensions"]
    base_reps = [k for k in dims if "+" not in k]
    mac("NRepresentations", str(len(base_reps)))
    mac("NMordredFeatures", f"{dims['Mordred'][1]:,}")
    mac("MaxTokenLength", str(reps["extraction_protocol"]["max_sequence_length"]))
    tl = reps["meta"]["ChemBERTa-ZINC_token_length"]
    mac("MedianTokenLength", f"{tl['median']:.0f}")
    mac("MaxObservedTokenLength", str(tl["max"]))
    mac("NTruncated", str(tl["n_truncated_at_max_length"]))
    n_mol = reps["n_molecules"]

    # cost table: seconds measured on CPU for the whole dataset
    cost = reps["wall_clock_seconds"]
    dim_lookup = {
        "ECFP4 (r=2, 2048 bit)": "ECFP4",
        "ECFP6 (r=3, 2048 bit)": "ECFP6",
        "ECFP4 counts": "ECFP4-count",
        "MACCS (167 keys)": "MACCS",
        "RDKit descriptors": "RDKit-desc",
        "ChemBERTa-ZINC embeddings": "ChemBERTa-ZINC",
        "ChemBERTa-MTR embeddings": "ChemBERTa-MTR",
        "ChemBERTa-MLM embeddings": "ChemBERTa-MLM",
        "MoLFormer-XL embeddings": "MoLFormer-XL",
    }
    with open(os.path.join(GEN, "tab_cost.tex"), "w", encoding="utf-8") as fh:
        for name, secs in cost.items():
            key = dim_lookup.get(name)
            nfeat = f"{dims[key][1]:,}" if key in dims else "--"
            fh.write(
                f"{tex(name)} & {nfeat} & {secs:.1f} & "
                f"{1000 * secs / n_mol:.2f} & CPU \\\\\n"
            )
        fh.write(
            f"Mordred descriptors & {dims['Mordred'][1]:,} & \\multicolumn{{2}}{{c}}"
            "{precomputed for the source benchmark} & CPU \\\\\n"
        )
    mac("CostECFP", f(1000 * cost["ECFP4 (r=2, 2048 bit)"] / n_mol, 2))
    mac("CostZinc", f(1000 * cost["ChemBERTa-ZINC embeddings"] / n_mol, 1))
    mac("CostRatio", f(cost["ChemBERTa-ZINC embeddings"] / cost["ECFP4 (r=2, 2048 bit)"], 0))

    with open(os.path.join(GEN, "tab_checkpoints.tex"), "w", encoding="utf-8") as fh:
        for name, spec in reps["checkpoints"].items():
            fh.write(f"{tex(name)} & \\texttt{{{tex(spec)}}} \\\\\n")

pool = load("a2b_pooling.json")
if pool:
    with open(os.path.join(GEN, "tab_pooling.tex"), "w", encoding="utf-8") as fh:
        for r in pool["strategies"]:
            fh.write(
                f"{tex(r['strategy'])} & {r['n_features']} & {f(r['q2_cv_oof'])} & "
                f"{f(r['test_r2'])} & {f(r['test_rmse'])} \\\\\n"
            )
    best = max(pool["strategies"], key=lambda r: r["test_r2"])
    worst = min(pool["strategies"], key=lambda r: r["test_r2"])
    mac("PoolBestName", tex(best["strategy"]))
    mac("PoolBestRtwo", f(best["test_r2"]))
    mac("PoolWorstRtwo", f(worst["test_r2"]))

# ------------------------------------------------------------------- a3 finetune
ft = load("a3_finetune_chemberta_zinc.json")
if ft:
    per = ft["per_seed"]
    r2s = [v["test_ensemble"]["test_r2"] for v in per.values()]
    q2s = [v["q2_cv_pooled_oof"] for v in per.values()]
    mac("NFtSeeds", str(len(per)))
    mac("FtSeedMeanRtwo", f(float(np.mean(r2s))))
    mac("FtSeedSDRtwo", f(float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0))
    mac("FtSeedMinRtwo", f(float(np.min(r2s))))
    mac("FtSeedMaxRtwo", f(float(np.max(r2s))))
    mac("FtSeedMeanQtwo", f(float(np.mean(q2s))))
    mac("FtFolds", str(ft["n_folds"]))
    with open(os.path.join(GEN, "tab_seeds.tex"), "w", encoding="utf-8") as fh:
        for seed, v in per.items():
            t = v["test_ensemble"]
            fh.write(
                f"{seed} & {v['cv_r2_mean']:.4f} $\\pm$ {v['cv_r2_sd']:.4f} & "
                f"{v['q2_cv_pooled_oof']:.4f} & {t['test_r2']:.4f} & {t['test_rmse']:.4f} & "
                f"{t['test_mae']:.4f} & {v['minutes']:.1f} \\\\\n"
            )

# --------------------------------------------------------------- a5 full matrix
matrix = load("a5_matrix_stratified.csv")
if matrix is not None:
    matrix["key"] = matrix["representation"] + " | " + matrix["learner"]
    mac("NBenchmarkCells", str(len(matrix)))
    # the grid proper: representation x learner cells, excluding the end-to-end
    # predictors and the members of the reproduced reference ensemble
    grid_only = matrix[~matrix["family"].isin(
        ["Descriptor ensemble member", "Matched descriptor ensemble",
         "Fine-tuned chemical language model", "Graph neural network"])]
    mac("NGridCells", str(len(grid_only)))

    # The grid is not a full product and cannot be described as one: the two
    # linear/kernel learners are run on every input, the two tree ensembles only
    # on the subset for which they are affordable. Writing "11 representations
    # times four learners" gives 44, which is neither the number of inputs nor
    # the number of cells, so every count the sentence needs is derived here.
    inputs = sorted(grid_only["representation"].unique())
    mac("NGridInputs", str(len(inputs)))
    mac("NConcatenations", str(sum("+" in r for r in inputs)))
    per_learner = grid_only.groupby("learner")["representation"].nunique()
    wide = sorted(l for l in per_learner.index if per_learner[l] == len(inputs))
    narrow = sorted(l for l in per_learner.index if per_learner[l] < len(inputs))
    if wide and narrow:
        mac("GridWideLearners", " and ".join(wide))
        mac("GridNarrowLearners", " and ".join(narrow))
        mac("NGridNarrowInputs", str(int(per_learner[narrow[0]])))
    with open(os.path.join(GEN, "tab_matrix_full.tex"), "w", encoding="utf-8") as fh:
        for _, r in matrix.sort_values("test_r2", ascending=False).iterrows():
            fh.write(
                f"{tex(r['representation'])} & {tex(r['learner'])} & {int(r['n_features'])} & "
                f"{f(r['q2_cv_oof'])} & {f(r['test_r2'])} & {f(r['test_rmse'])} & "
                f"{f(r['test_mae'])} \\\\\n"
            )
    with open(os.path.join(GEN, "tab_grid_choices.tex"), "w", encoding="utf-8") as fh:
        for _, r in matrix.sort_values(["representation", "learner"]).iterrows():
            bp = str(r["best_params"]).replace("{", "").replace("}", "").replace("'", "")
            bp = bp.replace("mdl__", "")
            fh.write(f"{tex(r['representation'])} & {tex(r['learner'])} & {tex(bp) or '--'} \\\\\n")

    ens = matrix[matrix["family"] == "Matched descriptor ensemble"]
    if len(ens):
        mac("MatchedEnsembleRtwo", f(ens.iloc[0]["test_r2"]))
        mac("MatchedEnsembleRMSE", f(ens.iloc[0]["test_rmse"]))
        mac("MatchedEnsembleQtwo", f(ens.iloc[0]["q2_cv_oof"]))

    def best(mask, label):
        sub = matrix[mask]
        if not len(sub):
            return None
        r = sub.sort_values("test_r2", ascending=False).iloc[0]
        mac(f"Best{label}Name", f"{tex(r['representation'])} + {tex(r['learner'])}")
        mac(f"Best{label}Rtwo", f(r["test_r2"]))
        mac(f"Best{label}RMSE", f(r["test_rmse"]))
        mac(f"Best{label}Qtwo", f(r["q2_cv_oof"]))
        return r

    best(matrix["family"] == "Fingerprint baseline", "Fingerprint")
    best(matrix["family"] == "Descriptor baseline", "Descriptor")
    best(matrix["family"] == "Frozen chemical language model", "Frozen")
    best(matrix["family"] == "Fine-tuned chemical language model", "FineTuned")
    best(matrix["family"] == "Hybrid", "Hybrid")
    best(matrix["family"] == "Graph neural network", "Gnn")

    # How much of the apparent representation effect is really a learner effect?
    tuned = matrix[matrix["family"] != "Matched descriptor ensemble"]
    spread = (
        tuned.groupby("representation")["test_r2"].agg(["min", "max", "count"]).query("count >= 3")
    )
    spread["range"] = spread["max"] - spread["min"]
    if len(spread):
        widest = spread.sort_values("range", ascending=False).iloc[0]
        mac("WidestLearnerRep", tex(str(spread.sort_values("range", ascending=False).index[0])))
        mac("WidestLearnerRange", f(widest["range"]))
        mac("WidestLearnerMin", f(widest["min"]))
        mac("WidestLearnerMax", f(widest["max"]))
        mac("MedianLearnerRange", f(float(spread["range"].median())))
        for rep, tag in (("Mordred", "Mordred"), ("MACCS", "Maccs"), ("ECFP4", "Ecfp")):
            sub = tuned[tuned["representation"] == rep]
            if len(sub) >= 2:
                mac(f"{tag}WorstRtwo", f(float(sub["test_r2"].min())))
                mac(f"{tag}BestRtwo", f(float(sub["test_r2"].max())))
                mac(
                    f"{tag}WorstLearner",
                    tex(str(sub.sort_values("test_r2").iloc[0]["learner"])),
                )
                mac(
                    f"{tag}BestLearner",
                    tex(str(sub.sort_values("test_r2").iloc[-1]["learner"])),
                )
        fam_best = tuned.groupby("family")["test_r2"].max()
        if len(fam_best) >= 2:
            mac("FamilySpread", f(float(fam_best.max() - fam_best.min())))

    # Reviewer 5 asked why the ChemBERTa-MTR Ridge cell of the original
    # submission was so weak; answering that needs the individual cells.
    for rep, tag in (("ChemBERTa-MTR", "Mtr"), ("ChemBERTa-ZINC", "Zinc")):
        frozen = matrix[
            (matrix["representation"] == rep)
            & (matrix["family"] == "Frozen chemical language model")
        ]
        for learner in ("Ridge", "SVR", "ExtraTrees", "CatBoost"):
            cell = frozen[frozen["learner"] == learner]
            if len(cell):
                mac(f"{tag}Frozen{learner}Rtwo", f(float(cell.iloc[0]["test_r2"])))
        ft = matrix[
            matrix["representation"].str.startswith(rep, na=False)
            & (matrix["family"] == "Fine-tuned chemical language model")
        ]
        if len(ft):
            mac(f"{tag}FineTunedRtwo", f(float(ft["test_r2"].max())))

# ---------------------------------------------------- a5 auxiliary split matrix
aux_rows = []
for name in ("stratified", "scaffold", "cluster", "dissimilarity", "source"):
    t = load(f"a5_matrix_{name}.csv")
    if t is None:
        continue
    t["split"] = name
    aux_rows.append(t)
if aux_rows:
    allsplits = pd.concat(aux_rows, ignore_index=True)
    # The Hybrid family holds both fingerprint--descriptor and language-model
    # concatenations, and the row that leads this table is ECFP4+Mordred, which
    # contains no language model at all. Labelling the whole family after the
    # language model would attribute the leading result to it, so the family is
    # split on whether the concatenation actually includes a language model.
    CLM_TOKENS = ("ChemBERTa", "MoLFormer")

    def is_clm_hybrid(rep):
        return any(t in str(rep) for t in CLM_TOKENS)

    fams = [
        ("ECFP4", lambda d: d["family"] == "Fingerprint baseline"),
        ("Mordred / RDKit", lambda d: d["family"] == "Descriptor baseline"),
        ("Frozen CLM", lambda d: d["family"] == "Frozen chemical language model"),
        ("Fine-tuned CLM", lambda d: d["family"] == "Fine-tuned chemical language model"),
        ("Hybrid, fingerprint + descriptor",
         lambda d: (d["family"] == "Hybrid") & ~d["representation"].map(is_clm_hybrid)),
        ("Hybrid, language model + fingerprint or descriptor",
         lambda d: (d["family"] == "Hybrid") & d["representation"].map(is_clm_hybrid)),
        ("Graph network", lambda d: d["family"] == "Graph neural network"),
    ]
    with open(os.path.join(GEN, "tab_generalisation.tex"), "w", encoding="utf-8") as fh:
        for label, select in fams:
            cells = []
            for name in ("stratified", "scaffold", "cluster", "dissimilarity", "source"):
                sub = allsplits[(allsplits["split"] == name) & select(allsplits)]
                cells.append(f"{sub['test_r2'].max():.3f}" if len(sub) else "--")
            fh.write(f"{label} & " + " & ".join(cells) + " \\\\\n")

    with open(os.path.join(GEN, "tab_aux_matrix.tex"), "w", encoding="utf-8") as fh:
        for name in ("scaffold", "cluster", "dissimilarity", "source"):
            sub = allsplits[allsplits["split"] == name].sort_values("test_r2", ascending=False)
            for i, (_, r) in enumerate(sub.iterrows()):
                label = pretty(name) if i == 0 else ""
                fh.write(
                    f"{label} & {tex(r['representation'])} & {tex(r['learner'])} & "
                    f"{f(r['q2_cv_oof'])} & {f(r['test_r2'])} & {f(r['test_rmse'])} & "
                    f"{f(r['test_mae'])} \\\\\n"
                )
            fh.write("\\addlinespace\n")
    # retention of accuracy relative to the stratified partition
    # "Hyb" means the fingerprint--descriptor hybrid, which is what the text calls
    # the hybrid that leads the stratified benchmark. Keeping the whole Hybrid
    # family under one tag would divide a language-model hybrid's score on a hard
    # partition by a fingerprint--descriptor score on the easy one and call the
    # ratio a retention.
    selectors = [
        ("Fp", lambda d: d["family"] == "Fingerprint baseline"),
        ("Desc", lambda d: d["family"] == "Descriptor baseline"),
        ("Ft", lambda d: d["family"] == "Fine-tuned chemical language model"),
        ("Frozen", lambda d: d["family"] == "Frozen chemical language model"),
        ("Gnn", lambda d: d["family"] == "Graph neural network"),
        ("Hyb", lambda d: (d["family"] == "Hybrid")
         & ~d["representation"].map(is_clm_hybrid)),
        ("HybClm", lambda d: (d["family"] == "Hybrid")
         & d["representation"].map(is_clm_hybrid)),
    ]
    # The rank each family holds on the easiest and on a hard partition. The
    # abstract leads on the inversion between the two, so the numbers behind it
    # are computed rather than counted off a printed table by eye.
    for stag, sname in (("Strat", "stratified"), ("Clus", "cluster"),
                        ("Diss", "dissimilarity")):
        best_by_family = {}
        for tag, select in selectors:
            sub = allsplits[select(allsplits) & (allsplits["split"] == sname)]["test_r2"]
            if len(sub) and np.isfinite(sub.max()):
                best_by_family[tag] = float(sub.max())
        order = sorted(best_by_family, key=lambda k: -best_by_family[k])
        mac(f"NFamilies{stag}", str(len(order)))
        for position, tag in enumerate(order, 1):
            mac(f"Rank{stag}{tag}", str(position))

    for tag, select in selectors:
        chosen = allsplits[select(allsplits)]
        base = chosen[chosen["split"] == "stratified"]["test_r2"].max()
        for name, stag in (("scaffold", "Scaf"), ("cluster", "Clus"),
                           ("dissimilarity", "Diss"), ("source", "Src")):
            sub = chosen[chosen["split"] == name]["test_r2"]
            if len(sub) and np.isfinite(base):
                mac(f"Rtwo{tag}{stag}", f(float(sub.max()), 3))
                mac(f"Ret{tag}{stag}", f(100 * float(sub.max()) / base, 0))

# ------------------------- distribution shift and rank signal per partition
# R2 is measured against the variance of the hold-out itself, so a partition
# whose activities are shifted or narrower than the training set can give a
# negative R2 at an unremarkable RMSE. These macros let the text separate the
# two cases instead of reporting a bare number.
SPLIT_TAGS = (("stratified", "Strat"), ("scaffold", "Scaf"), ("cluster", "Clus"),
              ("dissimilarity", "Diss"), ("source", "Src"))
curated_df = load("curated_dataset.csv")
splits_payload = load("a4_splits.json")
if curated_df is not None and splits_payload is not None:
    from scipy import stats as _st

    yv = curated_df["pIC50"].to_numpy(float)
    for name, stag in SPLIT_TAGS:
        s = splits_payload["splits"].get(name)
        if not s:
            continue
        tr = np.asarray(s["train_idx"], int)
        te = np.asarray(s["test_idx"], int)
        mac(f"TrainMean{stag}", f(float(yv[tr].mean()), 3))
        mac(f"TrainSD{stag}", f(float(yv[tr].std(ddof=1)), 3))
        mac(f"TestMean{stag}", f(float(yv[te].mean()), 3))
        mac(f"TestSD{stag}", f(float(yv[te].std(ddof=1)), 3))
        mac(f"ShiftKSP{stag}", pval(float(_st.ks_2samp(yv[tr], yv[te]).pvalue)))
        # RMSE of the trivial model that always predicts the training mean
        mac(f"NullRMSE{stag}", f(float(np.sqrt(np.mean((yv[te] - yv[tr].mean()) ** 2))), 3))

        preds = load(f"a5_matrix_{name}_predictions.json")
        table = load(f"a5_matrix_{name}.csv")
        if preds is None or table is None:
            continue
        yt = np.asarray(preds["y_test"], float)
        valid = table[table["representation"].astype(str).str.cat(
            table["learner"].astype(str), sep=" | ").isin(preds["predictions"])]
        if not len(valid):
            continue
        row = valid.sort_values("test_r2", ascending=False).iloc[0]
        key = f"{row['representation']} | {row['learner']}"
        p = np.asarray(preds["predictions"][key]["test"], float)
        mac(f"BestRMSE{stag}", f(float(np.sqrt(np.mean((yt - p) ** 2))), 3))
        mac(f"BestSpearman{stag}", f(float(_st.spearmanr(yt, p).statistic), 3))
        mac(f"BestName{stag}", tex(key))

        ft = table[table["family"] == "Fine-tuned chemical language model"]
        ft = ft[ft["representation"].astype(str).str.cat(
            ft["learner"].astype(str), sep=" | ").isin(preds["predictions"])]
        if len(ft):
            frow = ft.sort_values("test_r2", ascending=False).iloc[0]
            fkey = f"{frow['representation']} | {frow['learner']}"
            fp = np.asarray(preds["predictions"][fkey]["test"], float)
            mac(f"FtRMSE{stag}", f(float(np.sqrt(np.mean((yt - fp) ** 2))), 3))
            mac(f"FtSpearman{stag}", f(float(_st.spearmanr(yt, fp).statistic), 3))

# ------------------------------- how completely the source partition defeats us
# Every model there has a negative R2, but only some have an RMSE worse than the
# trivial training-mean predictor; the two are different statements because R2
# is measured against the hold-out's own mean.
if curated_df is not None and splits_payload is not None:
    src = load("a5_matrix_source.csv")
    s_split = splits_payload["splits"].get("source")
    if src is not None and s_split:
        _tr = np.asarray(s_split["train_idx"], int)
        _te = np.asarray(s_split["test_idx"], int)
        _null = float(np.sqrt(np.mean((yv[_te] - yv[_tr].mean()) ** 2)))
        mac("NSrcModels", str(len(src)))
        mac("NSrcNegativeRtwo", str(int((src["test_r2"] < 0).sum())))
        mac("NSrcWorseThanNull", str(int((src["test_rmse"] > _null).sum())))
        mac("BestRtwoSrc", f(float(src["test_r2"].max()), 3))
        _b = src.sort_values("test_r2", ascending=False).iloc[0]
        mac("BestNameSrc", tex(f"{_b['representation']} | {_b['learner']}"))

# -------------------------------------------------------------- a7 stats and AD
stats_ad = load("a7_stats_ad.json")
summary = load("a7_model_summary.csv")
if stats_ad and summary is not None:
    best_name = stats_ad["best_model"]
    top = summary.iloc[0]
    mac("BestModelName", tex(best_name))
    mac("BestTestRtwo", f(top["r2"]))
    mac("BestTestRMSE", f(top["rmse"]))
    mac("BestTestMAE", f(top["mae"]))
    mac("BestQtwo", f(top["q2_cv_oof"]))
    mac("BestTestRtwoCI", ci(eval(top["r2_ci95"]) if isinstance(top["r2_ci95"], str) else top["r2_ci95"]))
    mac("BestTestRMSECI", ci(eval(top["rmse_ci95"]) if isinstance(top["rmse_ci95"], str) else top["rmse_ci95"]))
    mac("NBoot", f"{stats_ad['bootstrap_resamples']:,}")

    stack = summary[summary["model"] == "Stacked model over representation families"]
    if len(stack):
        mac("StackTestRtwo", f(stack.iloc[0]["r2"]))
        mac("StackTestRMSE", f(stack.iloc[0]["rmse"]))
        mac("StackQtwo", f(stack.iloc[0]["q2_cv_oof"]))
        r2ci = stack.iloc[0]["r2_ci95"]
        mac("StackTestRtwoCI", ci(eval(r2ci) if isinstance(r2ci, str) else r2ci))
    ftrow = summary[summary["model"].str.contains("fine-tuned\\), fold ensemble \\(3-seed", na=False, regex=True)]
    if len(ftrow):
        mac("FtEnsembleRtwo", f(ftrow.iloc[0]["r2"]))
        mac("FtEnsembleRMSE", f(ftrow.iloc[0]["rmse"]))
        mac("FtEnsembleQtwo", f(ftrow.iloc[0]["q2_cv_oof"]))

    # composition of the stack, and what the language model adds to it
    abl = stats_ad.get("stack_ablation", {})
    coefs = stats_ad.get("stack_coefficients", {})
    with open(os.path.join(GEN, "tab_stack.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\multicolumn{3}{l}{\\emph{Members and ridge coefficients}} \\\\\n")
        for member, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
            fh.write(f"\\quad {tex(member)} & {c:+.3f} & \\\\\n")
        fh.write("\\addlinespace\n\\multicolumn{3}{l}{\\emph{Language-model ablation}} \\\\\n")
        if abl.get("with_language_model"):
            fh.write(
                f"\\quad Full stack & & {f(abl['with_language_model']['r2'])} \\\\\n"
                f"\\quad Without the language-model members & & {f(abl['without_language_model']['r2'])} \\\\\n"
            )
            pb = abl["paired_bootstrap"]
            fh.write(
                f"\\quad Difference (95\\% CI) & & {f(abl['delta_r2'], 3)} {ci(pb['delta_r2_ci95'])} \\\\\n"
                f"\\quad Wilcoxon $p$ on absolute errors & & {pval_cell(abl['wilcoxon_p'])} \\\\\n"
            )
            mac("StackAblationDelta", f(abl["delta_r2"], 3))
            mac("StackAblationCI", ci(pb["delta_r2_ci95"]))
            mac("StackAblationP", pval(abl["wilcoxon_p"]))
            mac("StackWithoutClmRtwo", f(abl["without_language_model"]["r2"]))
            mac("NStackMembers", str(len(abl.get("full_members", coefs))))
            clm = abl.get("language_model_members", [])
            mac("NStackClmMembers", str(len(clm)))
            mac("StackClmMembers", " and ".join(tex(m) for m in clm) if clm else "none")
    if coefs:
        member, coef = max(coefs.items(), key=lambda kv: kv[1])
        mac("StackTopMember", tex(member))
        mac("StackTopCoef", f"{coef:+.3f}")

    # the best model that is not itself a stack, for the sentence that asks
    # whether stacking bought anything on the hold-out
    single = summary[~summary["model"].str.startswith("Stacked model")]
    if len(single):
        mac("BestSingleName", tex(single.iloc[0]["model"]))
        mac("BestSingleRtwo", f(single.iloc[0]["r2"]))
        mac("BestSingleQtwo", f(single.iloc[0]["q2_cv_oof"]))

    with open(os.path.join(GEN, "tab_main_benchmark.tex"), "w", encoding="utf-8") as fh:
        for _, r in summary.head(14).iterrows():
            r2ci = eval(r["r2_ci95"]) if isinstance(r["r2_ci95"], str) else r["r2_ci95"]
            fh.write(
                f"{tex(r['model'])} & {f(r['q2_cv_oof'])} & {f(r['r2'])} & "
                f"{ci(r2ci)} & {f(r['rmse'])} & {f(r['mae'])} \\\\\n"
            )

    ad = stats_ad["applicability_domain_williams"]
    mac("ADnTest", str(ad["n_test"]))
    mac("ADhStar", f(ad["h_star"], 3))
    mac("ADnHighLeverage", str(ad["n_high_leverage"]))
    mac("ADnHighResidual", str(ad["n_high_standardised_residual"]))
    mac("ADnInside", str(ad["n_inside_domain"]))
    mac("ADPctInside", f(ad["pct_inside_domain"], 1))
    mac("ADRtwoInside", f(ad["metrics_inside_domain"]["r2"]))
    mac("ADRMSEInside", f(ad["metrics_inside_domain"]["rmse"]))
    mac("ADVarExplained", f(100 * ad["variance_explained"], 0))

    sim = stats_ad["applicability_domain_similarity"]
    mac("SimSpearman", f(sim["spearman_similarity_vs_abs_error"], 3))
    mac("SimSpearmanP", pval(sim["spearman_p"]))
    with open(os.path.join(GEN, "tab_similarity_bins.tex"), "w", encoding="utf-8") as fh:
        for b in sim["bins"]:
            fh.write(
                f"{tex(b['nn_tanimoto_bin'])} & {b['n']} & {f(b['rmse'])} & "
                f"{f(b['mae'])} & {f(b.get('r2'))} \\\\\n"
            )

    conf = stats_ad["conformal_prediction"]
    with open(os.path.join(GEN, "tab_conformal.tex"), "w", encoding="utf-8") as fh:
        for label, d in conf.items():
            hw = d.get("half_width", d.get("mean_half_width"))
            fh.write(
                f"{tex(label)} & {f(hw, 3)} & {100 * d['target_coverage']:.0f} & "
                f"{100 * d['empirical_coverage']:.1f} \\\\\n"
            )
    if "90%" in conf:
        mac("ConfHalfWidth", f(conf["90%"]["half_width"], 3))
        mac("ConfCoverage", f(100 * conf["90%"]["empirical_coverage"], 1))

    with open(os.path.join(GEN, "tab_classwise.tex"), "w", encoding="utf-8") as fh:
        for r in stats_ad["classwise"]:
            fh.write(
                f"{tex(r['category'])} & {r['n']} & {f(r['mean_pIC50'], 2)} & "
                f"{f(r.get('rmse'))} & {f(r.get('mae'))} & {f(r.get('r2'))} \\\\\n"
            )
    cw = [r for r in stats_ad["classwise"] if r.get("rmse") is not None]
    if cw:
        mac("ClasswiseRMSEMin", f(min(r["rmse"] for r in cw)))
        mac("ClasswiseRMSEMax", f(max(r["rmse"] for r in cw)))

pair = load("a7_pairwise_tests.csv")
if pair is not None and len(pair):
    # the one comparison four of the five reviewers asked for by name: the
    # leading model of this study against the reproduced reference workflow
    ref = "Ensemble mean (reproduced reference workflow)"
    m = pair[pair["model_b"].astype(str).str.contains(ref, regex=False)]
    if len(m):
        r = m.sort_values("delta_r2", ascending=False).iloc[0]
        d = eval(r["delta_r2_ci95"]) if isinstance(r["delta_r2_ci95"], str) else r["delta_r2_ci95"]
        mac("MatchedDeltaCI", ci(d))
        mac("MatchedDeltaP", pval(r["wilcoxon_p_bh"]))
        mac("MatchedDeltaModel", tex(r["model_a"]))
    # and the comparison that carries the language-model question
    ft = pair[pair["model_b"].astype(str).str.contains("(fine-tuned)", regex=False)]
    if len(ft):
        r = ft.sort_values("delta_r2", ascending=False).iloc[0]
        d = eval(r["delta_r2_ci95"]) if isinstance(r["delta_r2_ci95"], str) else r["delta_r2_ci95"]
        mac("BestVsFtDelta", f(r["delta_r2"], 3))
        mac("BestVsFtCI", ci(d))
        mac("BestVsFtP", pval(r["wilcoxon_p_bh"]))
    mac("NPairs", str(len(pair)))
    mac("NPairsSignificant", str(int((pair["wilcoxon_p_bh"] < 0.05).sum())))

    with open(os.path.join(GEN, "tab_pairwise.tex"), "w", encoding="utf-8") as fh:
        for _, r in pair.head(18).iterrows():
            d = eval(r["delta_r2_ci95"]) if isinstance(r["delta_r2_ci95"], str) else r["delta_r2_ci95"]
            fh.write(
                f"{tex(r['model_a'])} & {tex(r['model_b'])} & {f(r['delta_r2'], 3)} & "
                f"{ci(d)} & {pval_cell(r['wilcoxon_abs_error_p'])} & {pval_cell(r['wilcoxon_p_bh'])} \\\\\n"
            )

# --------------------------------------------------------------- a6 attribution
attr = load("a6_attribution.json")
enr = load("a6_attribution_enrichment.csv")
if attr:
    mac("NAttrMolecules", str(attr["n_molecules_analysed"]))
    mac("NAttrMethods", str(len(attr["methods"])))
    mac("NAttrSeeds", str(len(attr["seeds"])))
    cm = attr["cross_method_spearman"]
    if cm:
        meds = [v["median"] for v in cm.values()]
        mac("AttrCrossMethodRhoMin", f(float(np.min(meds)), 2))
        mac("AttrCrossMethodRhoMax", f(float(np.max(meds)), 2))
    cs = attr["cross_seed_spearman"]
    if cs:
        meds = [v["median"] for v in cs.values()]
        mac("AttrCrossSeedRho", f(float(np.median(meds)), 2))
    with open(os.path.join(GEN, "tab_attr_stability.tex"), "w", encoding="utf-8") as fh:
        for k, v in list(cm.items()) + list(cs.items()):
            key = " vs ".join(pretty(x.strip()) for x in k.split(" vs "))
            fh.write(f"{key} & {v['n']} & {v['median']:.3f} & [{v['iqr'][0]:.3f}, {v['iqr'][1]:.3f}] \\\\\n")
    with open(os.path.join(GEN, "tab_attr_faithfulness.tex"), "w", encoding="utf-8") as fh:
        for r in attr["faithfulness"]:
            fh.write(
                f"{pretty(r['method'])} & {r['n_molecules']} & "
                f"{r['mean_abs_change_top20pct_atoms']:.4f} & "
                f"{r['mean_abs_change_random_atoms']:.4f} & {r['ratio']:.2f} & "
                f"{pval_cell(r['wilcoxon_p_greater'])} \\\\\n"
            )
        fr = {r["method"]: r for r in attr["faithfulness"]}
        if "grad_x_input" in fr:
            mac("FaithRatio", f(fr["grad_x_input"]["ratio"], 2))
            mac("FaithP", pval(fr["grad_x_input"]["wilcoxon_p_greater"]))
if enr is not None and len(enr):
    with open(os.path.join(GEN, "tab_attr_enrichment.tex"), "w", encoding="utf-8") as fh:
        for _, r in enr.sort_values(["atom_group", "method"]).iterrows():
            fh.write(
                f"{pretty(r['atom_group'])} & {pretty(r['method'])} & {int(r['n_molecules'])} & "
                f"{r['mean_enrichment']:.3f} & {r['mean_permutation_null']:.3f} & "
                f"{r['pct_molecules_enriched']:.0f} & {pval_cell(r['p_bh_adjusted'])} \\\\\n"
            )
    ph = enr[enr["atom_group"] == "phenolic_OH_and_ipso_carbon"]
    if len(ph):
        mac("PhenolicEnrichMin", f(ph["mean_enrichment"].min(), 2))
        mac("PhenolicEnrichMax", f(ph["mean_enrichment"].max(), 2))
        mac("PhenolicEnrichPctMol", f(ph["pct_molecules_enriched"].max(), 0))
    cat = enr[enr["atom_group"] == "catechol_motif"]
    if len(cat):
        mac("CatecholEnrichMax", f(cat["mean_enrichment"].max(), 2))
    sig = enr[enr["p_bh_adjusted"] < 0.05]
    arom = enr[enr["atom_group"] == "aromatic"]
    if len(arom):
        mac("AromaticEnrichMin", f(arom["mean_enrichment"].min(), 2))
        mac("AromaticEnrichMax", f(arom["mean_enrichment"].max(), 2))
    carb = enr[enr["atom_group"] == "carbonyl"]
    if len(carb):
        mac("CarbonylEnrichMin", f(carb["mean_enrichment"].min(), 2))
        mac("CarbonylEnrichMax", f(carb["mean_enrichment"].max(), 2))
    mac("NAttrSignificant", str(len(sig)))
    mac("NAttrEnriched", str(int((sig["mean_enrichment"] > 1).sum())))
    mac("NAttrDepleted", str(int((sig["mean_enrichment"] < 1).sum())))
    mac("NAttrTests", str(len(enr)))

# ---------------------------------------------------- a8 chemical space, triage
cs = load("a8_chemspace_utility.json")
if cs:
    u = cs["umap_sensitivity"]
    mac("UmapPreservationMin", f(min(s["neighbourhood_preservation_k15"] for s in u), 3))
    mac("UmapPreservationMax", f(max(s["neighbourhood_preservation_k15"] for s in u), 3))
    mac("UmapSilhouetteMin", f(min(s["category_silhouette_2d"] for s in u), 3))
    mac("UmapSilhouetteMax", f(max(s["category_silhouette_2d"] for s in u), 3))
    mac("UmapActivityRhoMax", f(max(abs(s["spearman_activity_vs_umap1"]) for s in u), 3))
    with open(os.path.join(GEN, "tab_umap_scan.tex"), "w", encoding="utf-8") as fh:
        for s in u:
            fh.write(
                f"{s['n_neighbors']} & {s['min_dist']:.1f} & "
                f"{s['neighbourhood_preservation_k15']:.3f} & {s['category_silhouette_2d']:.3f} & "
                f"{s['spearman_activity_vs_umap1']:+.3f} & {s['spearman_activity_vs_umap2']:+.3f} \\\\\n"
            )
    al = cs["activity_landscape"]
    mac("NeighbourGap", f(al["mean_abs_activity_gap_5_nearest_neighbours"], 3))
    mac("RandomGap", f(al["mean_abs_activity_gap_random_pairs"], 3))
    mac("LandscapeP", pval(al["wilcoxon_p"]))
    pd_ = cs["potent_compound_dispersion"]
    mac("NTopDecile", str(pd_["n_top_decile"]))
    mac("TopDecileScaffolds", str(pd_["n_distinct_scaffolds_in_top_decile"]))
    mac("TopDecileNNSim", f(pd_["median_nn_tanimoto_among_top_decile"], 3))
    mmp = cs.get("matched_molecular_pairs") or {}
    if mmp:
        mac("NMMP", str(mmp["n_matched_pairs_H_to_OH"]))
        mac("MMPObsDelta", f(mmp["mean_observed_delta_pIC50"], 3))
        mac("MMPPredDelta", f(mmp["mean_predicted_delta_pIC50"], 3))
        mac("MMPRho", f(mmp["spearman_observed_vs_predicted"], 3))
        mac("MMPRhoP", pval(mmp["spearman_p"]))
        mac("MMPSignAgreement", f(mmp["sign_agreement_pct"], 0))
    tri = cs["prospective_triage"]["results"]
    with open(os.path.join(GEN, "tab_triage.tex"), "w", encoding="utf-8") as fh:
        for name in ("stratified", "scaffold", "cluster", "dissimilarity", "source"):
            if name not in tri:
                continue
            t = tri[name]
            fh.write(
                f"{pretty(name)} & {t['n_test']} & {t['n_actives']} & "
                f"{t['baseline_hit_rate_pct']:.1f} & {f(t['hit_rate_top10pct'], 1)} & "
                f"{f(t['enrichment_factor_top5pct'], 2)} & {f(t['enrichment_factor_top10pct'], 2)} & "
                f"{f(t['enrichment_factor_top20pct'], 2)} & {f(t['spearman_rank_correlation'], 3)} \\\\\n"
            )
    for name, tag in (("source", "Src"), ("scaffold", "Scaf"), ("cluster", "Clus"),
                      ("dissimilarity", "Diss"), ("stratified", "Strat")):
        if name in tri:
            mac(f"EFTen{tag}", f(tri[name]["enrichment_factor_top10pct"], 2))
            mac(f"HitTen{tag}", f(tri[name]["hit_rate_top10pct"], 1))
            mac(f"BaseHit{tag}", f(tri[name]["baseline_hit_rate_pct"], 1))
            # named explicitly: the triage model is chosen per partition by
            # training Q2 and is not the same model as the one whose R2 and
            # RMSE Section 3.5 quotes, so their rank correlations differ
            mac(f"TriageRho{tag}", f(tri[name]["spearman_rank_correlation"], 3))
            mac(f"TriageModel{tag}", tex(tri[name]["model"]))

# ------------------------------------------------------------- a9 legacy audit
leg = load("a9_legacy_audit.json")
if leg:
    pm = leg["provenance_match"]
    ftkey = "mean CLS over the ten supervised fold encoders"
    frkey = "frozen pretrained encoder, mean pooling"
    mac("LegacyMatchFT", f(pm[ftkey]["median_per_molecule_pearson_r"], 3))
    mac("LegacyMatchFrozen", f(pm[frkey]["median_per_molecule_pearson_r"], 3))
    ct = leg["contamination_test"]
    mac("NLegacySeen", str(ct["n_test_seen_by_supervised_encoders"]))
    mac("NLegacyUnseen", str(ct["n_test_never_seen"]))
    a = ct["archived_features"]
    mac("LegacyRMSESeen", f(a["rmse_seen"], 3))
    mac("LegacyRMSEUnseen", f(a["rmse_unseen"], 3))
    mac("LegacyRMSERatio", f(a["rmse_ratio_unseen_over_seen"], 2))
    mac("LegacyRMSEP", pval(a["mannwhitney_p_seen_better"]))
    mac("LegacyRtwoSeen", f(a["r2_seen"], 3))
    mac("LegacyRtwoUnseen", f(a["r2_unseen"], 3))
    probe = leg["ridge_probe_on_stratified_split"]
    mac("LegacyHeadlineRtwo", f(probe["archived feature matrix used for the submitted result"]["test_r2"]))
    mac("LegacyFrozenRtwo", f(probe[frkey]["test_r2"]))
    with open(os.path.join(GEN, "tab_legacy_audit.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\multicolumn{4}{l}{\\emph{Provenance of the archived feature matrix"
                 " (median per-molecule Pearson $r$)}} \\\\\n")
        for k, v in pm.items():
            fh.write(f"\\quad {tex(k)} & {v['median_per_molecule_pearson_r']:.3f} & & \\\\\n")
        fh.write("\\addlinespace\n")
        fh.write("\\multicolumn{4}{l}{\\emph{Held-out RMSE split by whether the encoders had"
                 " already seen the molecule}} \\\\\n")
        fh.write(f"\\quad Archived matrix & {a['rmse_seen']:.3f} & {a['rmse_unseen']:.3f} & "
                 f"{a['rmse_ratio_unseen_over_seen']:.2f} \\\\\n")
        for k, v in leg["contamination_test"]["label_free_controls"].items():
            fh.write(f"\\quad {tex(k)} & {v['rmse_seen']:.3f} & {v['rmse_unseen']:.3f} & "
                     f"{v['rmse_ratio_unseen_over_seen']:.2f} \\\\\n")

# --------------------------------------------------------------------- GNN row
gnn = load("a5b_gnn.json")
if gnn:
    mac("GnnRtwo", f(gnn.get("test_r2")))
    mac("GnnRMSE", f(gnn.get("test_rmse")))
    mac("GnnQtwo", f(gnn.get("q2_cv_oof")))

# ------------------------------------------- the two cross-validation depths
# Read off a5_matrix so the text cannot describe a protocol the code does not
# run: the inner grid search and the out-of-fold Q2 use different fold counts,
# and calling the result a nested-CV estimate would overstate it.
try:
    _a5 = open(os.path.join(HERE, "a5_matrix.py"), encoding="utf-8").read()
    _inner = re.search(r"INNER_CV\s*=\s*KFold\(n_splits=(\d+)", _a5)
    _oof = re.search(r"OOF_CV\s*=\s*KFold\(n_splits=(\d+)", _a5)
    if _inner:
        mac("InnerFolds", _inner.group(1))
    if _oof:
        mac("OofFolds", _oof.group(1))
except OSError:
    pass

# ------------------------------- a10 bootstrap intervals for every metric
# The response letter tells the reviewers that every reported R2, RMSE and MAE
# carries a bootstrap interval. a7 computed all three on the primary partition
# but only R2 reached a table, and the harder partitions carried point estimates
# only, so the promise was not kept anywhere the reviewer could check it.
cis = load("a10_metric_cis.csv")
if cis is not None:
    mac("NIntervalRows", str(len(cis)))
    with open(os.path.join(GEN, "tab_metric_cis.tex"), "w", encoding="utf-8") as fh:
        for name in ("stratified", "scaffold", "cluster", "dissimilarity", "source"):
            sub = cis[cis["split"] == name].sort_values("r2", ascending=False)
            for i, (_, r) in enumerate(sub.iterrows()):
                label = pretty(name) if i == 0 else ""
                cell = lambda v, lo, hi: (f"{f(r[v])} [{f(r[lo], 3)}, {f(r[hi], 3)}]")
                fh.write(
                    f"{label} & {tex(r['model'])} & {cell('r2', 'r2_lo', 'r2_hi')} & "
                    f"{cell('rmse', 'rmse_lo', 'rmse_hi')} & "
                    f"{cell('mae', 'mae_lo', 'mae_hi')} \\\\\n"
                )
            fh.write("\\addlinespace\n")

# ------------------------------------------ a10 assay-window sensitivity
sens = load("a10_assay_sensitivity.json")
if sens:
    mac("NSingleTimepoint", f"{sens['n_single_timepoint']:,}")
    mac("NAssayWindow", str(sens["n_window_wording"]))
    mac("PctAssayWindow", f(sens["pct_window_wording"], 1))
    mac("NTestSingleTimepoint", str(sens["n_test_single_timepoint"]))
    mac("NTrainSingleTimepoint", f"{sens['n_train_single_timepoint']:,}")
    deltas = [abs(c["delta_r2"]) for c in sens["cells"]]
    if deltas:
        mac("AssaySensMaxDelta", f(max(deltas), 3))
    for c in sens["cells"]:
        if c["representation"] == "ECFP4+Mordred":
            mac("AssaySensLeadRtwo", f(c["r2_single_timepoint"]))
    with open(os.path.join(GEN, "tab_assay_sensitivity.tex"), "w", encoding="utf-8") as fh:
        for c in sens["cells"]:
            fh.write(
                f"{tex(c['representation'])} + {tex(c['learner'])} & {f(c['r2_full'])} & "
                f"{f(c['r2_single_timepoint'])} & {f(c['delta_r2'], 4)} & "
                f"{f(c['rmse_full'])} & {f(c['rmse_single_timepoint'])} \\\\\n"
            )
    with open(os.path.join(GEN, "tab_assay_window.tex"), "w", encoding="utf-8") as fh:
        for text, n in sorted(sens["window_descriptions"].items(), key=lambda kv: -kv[1]):
            fh.write(f"{tex(text)} & {n} \\\\\n")

# -------------------------------------------------------- environment table
env = load("b0_environment.json")
if env:
    with open(os.path.join(GEN, "tab_environment.tex"), "w", encoding="utf-8") as fh:
        fh.write(f"Operating system & {tex(env['platform'])} ({tex(env['machine'])}) \\\\\n")
        fh.write(f"CPU threads available & {env['cpu_count']} \\\\\n")
        if env.get("gpu"):
            fh.write(f"GPU & {tex(env['gpu'])} \\\\\n")
        if env.get("cuda"):
            fh.write(f"CUDA & {tex(env['cuda'])} \\\\\n")
        fh.write(f"Python & {tex(env['python'])} \\\\\n")
        for name, ver in env["packages"].items():
            if ver:
                fh.write(f"{tex(name)} & {tex(ver)} \\\\\n")

# ----------------------------------------------------------- outlier table
if stats_ad and summary is not None:
    try:
        import pickle as _pickle

        with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
            smiles_all = _pickle.load(fh)["df"]["smiles"].tolist()
        curated = pd.read_csv(os.path.join(OUT, "curated_dataset.csv"))
        cats = curated["chemistry_category"].to_numpy()
        preds = json.load(
            open(os.path.join(OUT, "a5_matrix_stratified_predictions.json"), encoding="utf-8")
        )
        te = np.asarray(preds["test_idx"], int)
        yt = np.asarray(preds["y_test"], float)
        bm = stats_ad["best_model"]
        if bm in preds["predictions"]:
            pt = np.asarray(preds["predictions"][bm]["test"], float)
        else:
            ftj = load("a3_finetune_chemberta_zinc.json")
            pt = np.mean(
                [np.asarray(v["test_predictions"], float) for v in ftj["per_seed"].values()], axis=0
            )
        nn = np.asarray(stats_ad["applicability_domain_similarity"]["nn_tanimoto"], float)
        err = np.abs(yt - pt)
        order = np.argsort(err)[::-1][:12]
        with open(os.path.join(GEN, "tab_outliers.tex"), "w", encoding="utf-8") as fh:
            for i in order:
                smi = smiles_all[te[i]]
                smi = smi if len(smi) <= 46 else smi[:44] + "..."
                fh.write(
                    f"\\texttt{{\\seqsplit{{{tex(smi)}}}}} & {yt[i]:.2f} & {pt[i]:.2f} & "
                    f"{err[i]:.2f} & {nn[i]:.2f} & {tex(cats[te[i]])} \\\\\n"
                )
        mac("MaxAbsError", f(float(err.max()), 2))
        mac("NErrorAboveOne", str(int((err > 1.0).sum())))
    except Exception as exc:
        print("  (outlier table skipped:", exc, ")")

# ------------------------------------------------------- derived contrasts
def num(name):
    try:
        return float(MACROS[name])
    except (KeyError, ValueError):
        return None


best_r2, matched_r2 = num("BestTestRtwo"), num("MatchedEnsembleRtwo")
if best_r2 is not None and matched_r2 is not None:
    mac("MatchedDeltaRtwo", f(best_r2 - matched_r2, 3))
# The size of the leakage, measured like for like: the same ChemBERTa-ZINC
# encoder on the same hold-out, differing only in whether that hold-out leaked
# into encoder training. Comparing the retracted number against the revision's
# best model instead would compare different representation families and would
# not measure leakage at all.
legacy_r2 = num("LegacyHeadlineRtwo")
clean_r2 = num("BestFineTunedRtwo")
if legacy_r2 is not None and clean_r2 is not None:
    mac("LeakageLeakedRtwo", f(legacy_r2))
    mac("LeakageCleanRtwo", f(clean_r2))
    mac("LeakageDelta", f(legacy_r2 - clean_r2, 2))
fp_r2 = num("BestFingerprintRtwo")
if fp_r2 is not None and best_r2 is not None:
    mac("FingerprintDeltaRtwo", f(best_r2 - fp_r2, 3))

# --------------------------------------------- protect row-leading brackets
# A table row whose first cell begins with "[" is read by LaTeX as the optional
# vertical-space argument of the preceding \\, which swallows the row and ends
# the alignment. The similarity bins are labelled "[0.6, 0.8)", so brace every
# row that starts with a bracket rather than relying on each writer to remember.
for _fname in sorted(os.listdir(GEN)):
    if not _fname.startswith("tab_") or not _fname.endswith(".tex"):
        continue
    _path = os.path.join(GEN, _fname)
    _lines = open(_path, encoding="utf-8").read().split("\n")
    _fixed, _n = [], 0
    for _line in _lines:
        if _line.startswith("["):
            _head, _sep, _rest = _line.partition("&")
            _fixed.append("{" + _head.strip() + "} " + _sep + _rest)
            _n += 1
        else:
            _fixed.append(_line)
    if _n:
        open(_path, "w", encoding="utf-8").write("\n".join(_fixed))
        print(f"brace-protected {_n} row-leading brackets in {_fname}")

# --------------------------------------- flag macros used but not yet produced
import re

used, defined_elsewhere = set(), {"TeX", "LaTeX"}
for folder in (PKG, GEN):
    for fname in os.listdir(folder):
        if not fname.endswith(".tex") or fname == "results_macros.tex":
            continue
        body = open(os.path.join(folder, fname), encoding="utf-8").read()
        if folder == PKG:
            used.update(re.findall(r"\\([A-Z][A-Za-z]{3,})\{\}", body))
        # Supplementary table numbers, the manuscript cross-reference numbers
        # and any other hand-written definition live alongside this file.
        # Emitting a placeholder for one of those would clash with its real
        # \newcommand and break the build.
        defined_elsewhere.update(re.findall(r"\\(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?", body))
missing = sorted(used - set(MACROS) - defined_elsewhere)

# ------------------------------------------------------------------------ write
with open(os.path.join(GEN, "results_macros.tex"), "w", encoding="utf-8") as fh:
    fh.write("% Auto-generated by b1_make_macros.py -- do not edit by hand.\n")
    for k in sorted(MACROS):
        fh.write(f"\\newcommand*{{\\{k}}}{{{MACROS[k]}}}\n")
    if missing:
        fh.write("\n% Referenced in the text but not yet produced by the analysis.\n")
        for k in missing:
            fh.write(f"\\newcommand*{{\\{k}}}{{\\textbf{{??{k}??}}}}\n")

print(f"wrote {len(MACROS)} macros to {os.path.join(GEN, 'results_macros.tex')}")
if missing:
    print(f"\n{len(missing)} macros referenced in the documents are still missing:")
    for k in missing:
        print("  ??", k)
else:
    print("\nevery macro referenced in the documents is available")
