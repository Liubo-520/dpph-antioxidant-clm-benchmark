"""Generate the six figures of the revised manuscript from the analysis outputs.

Each figure is drawn only from files written by the a1--a9 scripts, so the
figures cannot disagree with the tables.
"""

import json
import os
import pickle
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

# The Hybrid family holds both fingerprint--descriptor and language-model
# concatenations. They behave differently -- one leads the stratified partition
# and collapses, the other does not lead anywhere -- so plotting them as one
# family draws a line through two different models.
CLM_TOKENS = ("ChemBERTa", "MoLFormer")


def split_hybrid(frame):
    """Return the frame with Hybrid separated by whether it contains a language model."""
    if frame is None or "family" not in frame:
        return frame
    out = frame.copy()
    is_clm = out["representation"].astype(str).apply(
        lambda r: any(t in r for t in CLM_TOKENS))
    out.loc[(out["family"] == "Hybrid") & ~is_clm, "family"] = "Hybrid fp+desc"
    out.loc[(out["family"] == "Hybrid") & is_clm, "family"] = "Hybrid with CLM"
    return out


HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")
FIG = os.path.join(ROOT, "revision", "Revision_R1_new", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
    }
)

C = {
    "fp": "#4C72B0",
    "desc": "#DD8452",
    "frozen": "#937860",
    "ft": "#C44E52",
    "hybrid": "#55A868",
    "hybrid_clm": "#2E6F51",
    "ens": "#8172B3",
    "gnn": "#DA8BC3",   # kept clearly apart from the fingerprint blue
    "grey": "#7F7F7F",
    "train": "#B0B8C4",
    "test": "#C44E52",
}
FAMILY_COLOUR = {
    "Fingerprint baseline": C["fp"],
    "Descriptor baseline": C["desc"],
    "Frozen chemical language model": C["frozen"],
    "Fine-tuned chemical language model": C["ft"],
    "Hybrid fp+desc": C["hybrid"],
    "Hybrid with CLM": C["hybrid_clm"],
    "Graph neural network": C["gnn"],
    "Descriptor ensemble member": C["ens"],
    "Matched descriptor ensemble": C["ens"],
}
SPLITS = ["stratified", "scaffold", "cluster", "dissimilarity", "source"]
SPLIT_LABEL = {
    "stratified": "Chemistry-\nstratified",
    "scaffold": "Scaffold-\ndisjoint",
    "cluster": "Cluster-\ndisjoint",
    "dissimilarity": "Maximum-\ndissimilarity",
    "source": "Source-\ndisjoint",
}


def panel(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def load(name, default=None):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return default
    return json.load(open(p, encoding="utf-8")) if name.endswith(".json") else pd.read_csv(p)


with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
    df = pickle.load(fh)["df"]
y_all = df["pIC50"].to_numpy(float)
curated = pd.read_csv(os.path.join(OUT, "curated_dataset.csv"))
categories = curated["chemistry_category"].to_numpy()
splits_payload = load("a4_splits.json")
train_idx = np.array(splits_payload["splits"]["stratified"]["train_idx"])
test_idx = np.array(splits_payload["splits"]["stratified"]["test_idx"])
audit = load("a1_dataset_audit.json")


# ============================================================ Fig 1: dataset
def figure1():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ic50_um = 10 ** (6 - y_all)

    ax = axes[0, 0]
    ax.hist(np.log10(ic50_um), bins=45, color=C["grey"], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("log$_{10}$(IC$_{50}$ / $\\mu$M)")
    ax.set_ylabel("Compounds")
    ax.set_title("Raw potency spans seven orders of magnitude", loc="left")
    e = audit["endpoint"]
    ax.text(0.97, 0.93, f"skewness {e['skew_ic50']:.1f}\nmedian {e['ic50_micromolar_median']:.0f} $\\mu$M",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    tidy(ax)
    panel(ax, "a")

    ax = axes[0, 1]
    bins = np.linspace(y_all.min(), y_all.max(), 40)
    ax.hist(y_all[train_idx], bins=bins, color=C["train"], edgecolor="white", linewidth=0.4,
            label=f"Training ($n$ = {len(train_idx):,})")
    ax.hist(y_all[test_idx], bins=bins, color=C["test"], alpha=0.85, edgecolor="white",
            linewidth=0.4, label=f"Held-out ($n$ = {len(test_idx)})")
    s = audit["split_endpoint_distribution"]
    ax.set_xlabel("pIC$_{50}$")
    ax.set_ylabel("Compounds")
    ax.set_title("Partitions match in location and spread", loc="left")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.55, f"KS $D$ = {s['ks_statistic']:.3f}, $p$ = {s['ks_p']:.2f}\n"
                        f"Levene $p$ = {s['levene_p']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    tidy(ax)
    panel(ax, "b")

    ax = axes[1, 0]
    cats = sorted(audit["categories"], key=lambda z: -z["n_total"])
    names = [c["category"] for c in cats]
    pos = np.arange(len(names))
    ax.barh(pos + 0.19, [c["n_train"] for c in cats], height=0.36, color=C["train"], label="Training")
    ax.barh(pos - 0.19, [c["n_test"] for c in cats], height=0.36, color=C["test"], label="Held-out")
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlabel("Compounds (log scale)")
    ax.set_title("Category composition is preserved but imbalanced", loc="left")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    tidy(ax)
    panel(ax, "c")

    ax = axes[1, 1]
    order = [c["category"] for c in sorted(audit["categories"], key=lambda z: -z["mean_pIC50"])]
    data = [y_all[categories == c] for c in order]
    bp = ax.boxplot(data, vert=False, widths=0.62, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.1))
    for patch in bp["boxes"]:
        patch.set(facecolor="#DCE3EC", edgecolor="#4C566A", linewidth=0.7)
    for i, c in enumerate(order):
        v = y_all[categories == c]
        ax.plot(v, np.full(len(v), i + 1) + np.random.default_rng(i).normal(0, 0.055, len(v)),
                ".", color=C["grey"], markersize=1.4, alpha=0.5)
        ax.text(7.9, i + 1, f"$n$ = {len(v)}", va="center", fontsize=6.8, color="#444444")
    ax.set_yticklabels(order)
    ax.set_xlabel("pIC$_{50}$")
    ax.set_xlim(-0.4, 9.1)
    ax.set_title("The residual class has the widest activity range", loc="left")
    tidy(ax)
    panel(ax, "d")

    fig.tight_layout(w_pad=2.2, h_pad=2.0)
    fig.savefig(os.path.join(FIG, "Fig1.png"))
    plt.close(fig)
    print("Fig1 written")


# ================================================== Fig 2: benchmark matrix
def figure2():
    matrix = load("a5_matrix_stratified.csv")
    if matrix is None:
        print("Fig2 skipped (benchmark matrix missing)")
        return
    summary = load("a7_model_summary.csv")
    preds = load("a5_matrix_stratified_predictions.json")

    # Panel (a) carries one row per benchmark cell, so its height has to scale
    # with the number of cells or the labels collide into an unreadable block.
    n_cells = len(matrix[~matrix["family"].str.contains("ensemble member", na=False)])
    panel_a_in = max(3.4, 0.165 * n_cells)
    fig = plt.figure(figsize=(7.2, panel_a_in + 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[panel_a_in, 2.6], hspace=0.16, wspace=0.30)

    ax = fig.add_subplot(gs[0, :])
    m = matrix[~matrix["family"].str.contains("ensemble member", na=False)].copy()
    m = m.sort_values("test_r2")
    labels = m["representation"] + "  ·  " + m["learner"]
    labels = labels.str.replace("Ensemble mean (reproduced reference workflow)", "matched ensemble", regex=False)
    colours = [FAMILY_COLOUR.get(f, C["grey"]) for f in m["family"]]
    pos = np.arange(len(m))
    ax.barh(pos, m["test_r2"], color=colours, height=0.76)
    ax.plot(m["q2_cv_oof"], pos, "|", color="black", markersize=5, markeredgewidth=1.1,
            label="Q$^2_{CV}$ (training out-of-fold)")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_ylim(-0.8, len(m) - 0.2)
    ax.set_xlabel("Held-out R$^2$")
    ax.set_xlim(0, max(0.9, m["test_r2"].max() * 1.08))
    ax.set_title("Every representation $\\times$ learner cell, tuned inside the training partition", loc="left")
    handles = [Line2D([], [], marker="s", linestyle="", color=v, label=k)
               for k, v in FAMILY_COLOUR.items() if k in set(m["family"])]
    handles.append(Line2D([], [], marker="|", linestyle="", color="black", markeredgewidth=1.1,
                          label="Q$^2_{CV}$"))
    # the short bars at the bottom leave the lower right of the panel free
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, ncol=1,
              handletextpad=0.5, borderpad=0.6)
    tidy(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    for fam, colour in FAMILY_COLOUR.items():
        sub = matrix[matrix["family"] == fam]
        if len(sub):
            ax.scatter(sub["q2_cv_oof"], sub["test_r2"], s=26, color=colour, edgecolor="white",
                       linewidth=0.5, label=fam, zorder=3)
    lo = min(matrix["q2_cv_oof"].min(), matrix["test_r2"].min()) - 0.05
    ax.plot([lo, 0.95], [lo, 0.95], "--", color=C["grey"], linewidth=0.8, zorder=1)
    ax.set_xlabel("Q$^2_{CV}$ (training)")
    ax.set_ylabel("Held-out R$^2$")
    ax.set_title("Held-out scores track\ncross-validation", loc="left", fontsize=8.8)
    tidy(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    if summary is not None and preds is not None:
        best = summary.iloc[0]["model"]
        yt = np.asarray(preds["y_test"], float)
        if best in preds["predictions"]:
            p = np.asarray(preds["predictions"][best]["test"], float)
        else:
            ft = load("a3_finetune_chemberta_zinc.json")
            p = np.mean([np.asarray(v["test_predictions"], float)
                         for v in ft["per_seed"].values()], axis=0)
        ax.scatter(yt, p, s=20, color=C["ft"], alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        lim = [min(yt.min(), p.min()) - 0.3, max(yt.max(), p.max()) + 0.3]
        ax.plot(lim, lim, "--", color=C["grey"], linewidth=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        r2 = summary.iloc[0]["r2"]
        rmse = summary.iloc[0]["rmse"]
        ax.text(0.04, 0.95, f"R$^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\n$n$ = {len(yt)}",
                transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_title("Leading model on the\nheld-out set", loc="left", fontsize=8.8)
    ax.set_xlabel("Experimental pIC$_{50}$")
    ax.set_ylabel("Predicted pIC$_{50}$")
    tidy(ax)
    panel(ax, "c")

    fig.savefig(os.path.join(FIG, "Fig2.png"))
    plt.close(fig)
    print("Fig2 written")


# ============================================== Fig 3: generalisation
def figure3():
    chars = load("a4_split_characterisation.csv")
    frames = {}
    for s in SPLITS:
        t = load(f"a5_matrix_{s}.csv")
        if t is not None:
            frames[s] = t
    if not frames or chars is None:
        print("Fig3 skipped (split benchmarks missing)")
        return
    chars = chars.set_index("split")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    nn_by_split = {c["split"]: np.asarray(c.get("test_nn_tanimoto", []), float)
                   for c in splits_payload["characterisation"]}

    ax = axes[0, 0]
    for i, s in enumerate(SPLITS):
        sim = nn_by_split.get(s, np.array([]))
        if sim.size == 0:
            continue
        parts = ax.violinplot([sim], positions=[i], widths=0.75, showextrema=False)
        for b in parts["bodies"]:
            b.set(facecolor=plt.cm.viridis(i / 4), alpha=0.75, edgecolor="white")
        ax.plot(i, np.median(sim), "o", color="black", markersize=3.2, zorder=4)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=6.6)
    ax.set_ylabel("Nearest-neighbour Tanimoto\nto the training set")
    ax.set_title("Partitions of increasing\nstructural difficulty", loc="left", fontsize=8.8)
    tidy(ax)
    panel(ax, "a")

    frames = {k: split_hybrid(v) for k, v in frames.items()}
    fams = ["Fingerprint baseline", "Descriptor baseline",
            "Frozen chemical language model", "Fine-tuned chemical language model",
            "Hybrid fp+desc", "Hybrid with CLM", "Graph neural network"]
    fams = [f for f in fams if any(f in set(t["family"]) for t in frames.values())]
    short = {"Fingerprint baseline": "ECFP / MACCS", "Descriptor baseline": "Descriptors",
             "Frozen chemical language model": "Frozen CLM",
             "Fine-tuned chemical language model": "Fine-tuned CLM",
             "Hybrid fp+desc": "Hybrid fp+desc", "Hybrid with CLM": "Hybrid + CLM",
             "Graph neural network": "Graph network"}

    ax = axes[0, 1]
    width = 0.9 / max(len(fams), 1)
    offset = (len(fams) - 1) / 2
    for k, fam in enumerate(fams):
        vals = []
        for s in SPLITS:
            sub = frames.get(s)
            v = sub[sub["family"] == fam]["test_r2"].max() if sub is not None else np.nan
            vals.append(v if np.isfinite(v) else np.nan)
        ax.bar(np.arange(len(SPLITS)) + (k - offset) * width, vals, width=width,
               color=FAMILY_COLOUR[fam], label=short[fam])
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=6.6)
    ax.set_ylabel("Best held-out R$^2$")
    ax.set_title("Accuracy by representation family", loc="left")
    ax.legend(fontsize=6.4, ncol=2, loc="upper right")
    tidy(ax)
    panel(ax, "b")

    ax = axes[1, 0]
    for fam in fams:
        base = frames["stratified"][frames["stratified"]["family"] == fam]["test_r2"].max()
        if not np.isfinite(base):
            continue
        xs, ys = [], []
        for i, s in enumerate(SPLITS):
            sub = frames.get(s)
            if sub is None:
                continue
            v = sub[sub["family"] == fam]["test_r2"].max()
            if np.isfinite(v):
                xs.append(i)
                ys.append(100 * v / base)
        ax.plot(xs, ys, "-o", color=FAMILY_COLOUR[fam], markersize=3.6, linewidth=1.3,
                label=short[fam])
    ax.axhline(100, ls="--", color=C["grey"], linewidth=0.8)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=6.6)
    ax.set_ylabel("Accuracy retained (% of stratified R$^2$)")
    ax.set_title("Accuracy retained under structural shift", loc="left")
    ax.legend(fontsize=6.4)
    tidy(ax)
    panel(ax, "c")

    ax = axes[1, 1]
    for fam in fams:
        xs, ys = [], []
        for s in SPLITS:
            sub = frames.get(s)
            if sub is None or s not in chars.index:
                continue
            v = sub[sub["family"] == fam]["test_r2"].max()
            if np.isfinite(v):
                xs.append(chars.loc[s, "test_nn_tanimoto_median"])
                ys.append(v)
        ax.plot(xs, ys, "o", color=FAMILY_COLOUR[fam], markersize=5, label=short[fam])
    ax.set_xlabel("Median nearest-neighbour Tanimoto of the partition")
    ax.set_ylabel("Best held-out R$^2$")
    ax.set_title("Accuracy tracks analogue overlap", loc="left")
    tidy(ax)
    panel(ax, "d")

    fig.tight_layout(w_pad=2.4, h_pad=2.2)
    fig.savefig(os.path.join(FIG, "Fig3.png"))
    plt.close(fig)
    print("Fig3 written")


# ====================================== Fig 4: applicability domain
def figure4():
    sa = load("a7_stats_ad.json")
    if sa is None:
        print("Fig4 skipped (statistics missing)")
        return
    ad = sa["applicability_domain_williams"]
    sim = sa["applicability_domain_similarity"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    ax = axes[0, 0]
    lev = np.asarray(ad["leverage"], float)
    sr = np.asarray(ad["standardised_residual"], float)
    inside = (lev <= ad["h_star"]) & (np.abs(sr) <= 3)
    ax.scatter(lev[inside], sr[inside], s=20, color=C["fp"], alpha=0.8, edgecolor="white",
               linewidth=0.4, label=f"Inside domain ($n$ = {inside.sum()})")
    if (~inside).sum():
        ax.scatter(lev[~inside], sr[~inside], s=26, color=C["ft"], alpha=0.9, edgecolor="white",
                   linewidth=0.4, label=f"Outside ($n$ = {(~inside).sum()})")
    ax.axvline(ad["h_star"], ls="--", color=C["grey"], linewidth=0.9)
    ax.axhline(3, ls=":", color=C["grey"], linewidth=0.9)
    ax.axhline(-3, ls=":", color=C["grey"], linewidth=0.9)
    ax.text(ad["h_star"], ax.get_ylim()[1], f"  $h^*$ = {ad['h_star']:.3f}", va="top", fontsize=7)
    # a handful of very high-leverage compounds otherwise compress the bulk of
    # the plot into the left-hand margin
    xmax = max(4 * ad["h_star"], float(np.percentile(lev, 97)))
    beyond = int((lev > xmax).sum())
    if beyond:
        ax.set_xlim(-0.02 * xmax, xmax * 1.05)
        # stated in the axis label rather than inside the panel, where it would
        # sit on top of the legend
        ax.set_xlabel(f"Leverage ({beyond} compounds up to {lev.max():.1f} lie beyond the axis)")
    else:
        ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardised residual")
    ax.set_title(f"Williams plot, {ad['n_test']} held-out compounds", loc="left")
    ax.legend(loc="lower right", fontsize=6.8)
    tidy(ax)
    panel(ax, "a")

    ax = axes[0, 1]
    nn = np.asarray(sim["nn_tanimoto"], float)
    err = np.asarray(sim["abs_error"], float)
    ax.scatter(nn, err, s=20, color=C["desc"], alpha=0.8, edgecolor="white", linewidth=0.4)
    if len(nn) > 5:
        fit = np.poly1d(np.polyfit(nn, err, 1))
        xs = np.linspace(nn.min(), nn.max(), 50)
        ax.plot(xs, fit(xs), "-", color="black", linewidth=1.1)
    ax.set_xlabel("Nearest-neighbour Tanimoto to training set")
    ax.set_ylabel("Absolute prediction error")
    ax.set_title("Error grows with structural novelty", loc="left")
    ax.text(0.96, 0.94, f"Spearman $\\rho$ = {sim['spearman_similarity_vs_abs_error']:.3f}\n"
                        f"$p$ = {sim['spearman_p']:.1e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2)
    tidy(ax)
    panel(ax, "b")

    ax = axes[1, 0]
    bins = sim["bins"]
    pos = np.arange(len(bins))
    ax.bar(pos, [b["rmse"] for b in bins], color=C["desc"], width=0.62)
    for i, b in enumerate(bins):
        ax.text(i, b["rmse"] + 0.012, f"$n$ = {b['n']}", ha="center", fontsize=6.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([b["nn_tanimoto_bin"] for b in bins], fontsize=7)
    ax.set_xlabel("Nearest-neighbour Tanimoto bin")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by similarity bin", loc="left")
    tidy(ax)
    panel(ax, "c")

    ax = axes[1, 1]
    conf = sa["conformal_prediction"]
    plain = {k: v for k, v in conf.items() if "normalised" not in k}
    norm = {k: v for k, v in conf.items() if "normalised" in k}
    for label, d, colour, marker in (("Plain conformal", plain, C["fp"], "o"),
                                     ("Novelty-normalised", norm, C["ft"], "s")):
        xs = [100 * v["target_coverage"] for v in d.values()]
        ys = [100 * v["empirical_coverage"] for v in d.values()]
        ax.plot(xs, ys, marker, color=colour, markersize=6, label=label)
    ax.plot([75, 95], [75, 95], "--", color=C["grey"], linewidth=0.8)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Conformal intervals are calibrated", loc="left")
    ax.legend(loc="lower right", fontsize=6.8)
    tidy(ax)
    panel(ax, "d")

    fig.tight_layout(w_pad=2.4, h_pad=2.2)
    fig.savefig(os.path.join(FIG, "Fig4.png"))
    plt.close(fig)
    print("Fig4 written")


# ============================================ Fig 5: attribution
def figure5():
    attr = load("a6_attribution.json")
    enr = load("a6_attribution_enrichment.csv")
    if attr is None or enr is None:
        print("Fig5 skipped (attribution missing)")
        return
    from b3_attr_render import render_examples  # local helper, see script

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.95, 1.3], hspace=0.55, wspace=0.3)

    method_names = attr["methods"]
    pretty_m = {"grad_x_input": "grad $\\times$ input", "integrated_gradients": "integ. grad.",
                "occlusion": "occlusion", "attention_rollout": "attn. rollout"}
    pretty_g = {"phenolic_OH_and_ipso_carbon": "phenolic O–H\n+ ipso C",
                "phenolic_OH_oxygen": "phenolic O",
                "catechol_motif": "catechol", "aromatic": "aromatic",
                "carbonyl": "carbonyl", "conjugated_CC": "conjugated C=C",
                "non_aromatic": "non-aromatic", "aliphatic_OH_oxygen": "aliphatic O–H"}

    ax = fig.add_subplot(gs[0, :])
    groups = [g for g in ["phenolic_OH_and_ipso_carbon", "catechol_motif", "aromatic",
                          "carbonyl", "conjugated_CC", "non_aromatic"]
              if g in set(enr["atom_group"])]
    width = 0.2
    for k, m in enumerate(method_names):
        vals, sig = [], []
        for g in groups:
            row = enr[(enr["method"] == m) & (enr["atom_group"] == g)]
            vals.append(row["mean_enrichment"].iloc[0] / row["mean_permutation_null"].iloc[0]
                        if len(row) else np.nan)
            sig.append(row["p_bh_adjusted"].iloc[0] < 0.05 if len(row) else False)
        x = np.arange(len(groups)) + (k - 1.5) * width
        bars = ax.bar(x, vals, width=width, color=plt.cm.plasma(0.15 + 0.22 * k),
                      label=pretty_m.get(m, m))
        for b, s in zip(bars, sig):
            if s and np.isfinite(b.get_height()):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, "*",
                        ha="center", fontsize=8)
    ax.axhline(1.0, ls="--", color=C["grey"], linewidth=0.9)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([pretty_g.get(g, g) for g in groups], fontsize=6.8)
    ax.set_ylabel("Attribution enrichment\n(observed / permutation null)")
    ax.set_title(f"Attribution is selective across all {attr['n_molecules_analysed']} held-out molecules",
                 loc="left")
    ax.legend(fontsize=6.4, ncol=4, loc="upper right")
    tidy(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    items = list(attr["cross_method_spearman"].items()) + list(attr["cross_seed_spearman"].items())
    labels, meds, los, his = [], [], [], []
    for k, v in items:
        parts = [pretty_m.get(p.strip(), p.strip()) for p in k.split(" vs ")]
        labels.append(" vs ".join(parts))
        meds.append(v["median"])
        los.append(v["median"] - v["iqr"][0])
        his.append(v["iqr"][1] - v["median"])
    pos = np.arange(len(labels))
    colours = [C["fp"]] * len(attr["cross_method_spearman"]) + [C["ft"]] * len(attr["cross_seed_spearman"])
    ax.barh(pos, meds, xerr=[los, his], color=colours, height=0.66,
            error_kw=dict(elinewidth=0.8, capsize=1.6))
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlabel("Median Spearman $\\rho$")
    ax.set_title("Stability across methods and seeds", loc="left")
    ax.invert_yaxis()
    tidy(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    fa = attr["faithfulness"]
    pos = np.arange(len(fa))
    ax.bar(pos - 0.19, [r["mean_abs_change_top20pct_atoms"] for r in fa], width=0.36,
           color=C["ft"], label="Top 20% attributed")
    ax.bar(pos + 0.19, [r["mean_abs_change_random_atoms"] for r in fa], width=0.36,
           color=C["grey"], label="Random atoms")
    for i, r in enumerate(fa):
        ax.text(i, max(r["mean_abs_change_top20pct_atoms"], r["mean_abs_change_random_atoms"]) * 1.04,
                f"{r['ratio']:.1f}$\\times$", ha="center", fontsize=6.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([pretty_m.get(r["method"], r["method"]) for r in fa], fontsize=6.4)
    ax.set_ylabel("|$\\Delta$ predicted pIC$_{50}$|")
    ax.set_title("Masking the top atoms changes predictions more", loc="left")
    # headroom so the ratio labels and the legend do not sit on the bars
    ax.set_ylim(0, max(r["mean_abs_change_top20pct_atoms"] for r in fa) * 1.32)
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              columnspacing=1.0, handletextpad=0.4)
    tidy(ax)
    panel(ax, "c")

    axes_row = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    try:
        render_examples(axes_row)
    except Exception as exc:  # rendering is cosmetic; never block the figure
        for ax in axes_row:
            ax.axis("off")
        print("  (attribution example panel skipped:", exc, ")")
    panel(axes_row[0], "d")

    fig.savefig(os.path.join(FIG, "Fig5.png"))
    plt.close(fig)
    print("Fig5 written")


# ============================================ Fig 6: chemical space, utility
def figure6():
    cs = load("a8_chemspace_utility.json")
    if cs is None:
        print("Fig6 skipped (chemical-space analysis missing)")
        return
    emb_path = os.path.join(FEAT, "umap_reference_embedding.npy")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    ax = axes[0, 0]
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_all, s=6, cmap="viridis", alpha=0.85,
                        linewidth=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
        cb.set_label("pIC$_{50}$", fontsize=7.5)
        cb.ax.tick_params(labelsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])
    u = cs["umap_sensitivity"]
    pres = np.mean([s["neighbourhood_preservation_k15"] for s in u])
    ax.set_title(f"UMAP overview only\n({100 * pres:.0f}% of neighbourhoods kept)",
                 loc="left", fontsize=8.8)
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, "a")

    ax = axes[0, 1]
    scan = pd.DataFrame(u)
    for nn_, grp in scan.groupby("n_neighbors"):
        ax.plot(grp["min_dist"], grp["neighbourhood_preservation_k15"], "-o", markersize=3.6,
                linewidth=1.2, label=f"n_neighbors = {nn_}")
    ax2 = ax.twinx()
    ax2.plot(scan["min_dist"], scan["category_silhouette_2d"], "x", color=C["ft"], markersize=4,
             label="category silhouette")
    ax2.set_ylabel("Category silhouette", color=C["ft"], fontsize=7.5)
    ax2.tick_params(axis="y", labelcolor=C["ft"], labelsize=6.5)
    ax2.spines[["top"]].set_visible(False)
    ax.set_xlabel("min_dist")
    ax.set_ylabel("Neighbourhood preservation ($k$ = 15)")
    ax.set_title("Categories never separate,\nat any setting", loc="left", fontsize=8.8)
    ax.legend(fontsize=6.2, loc="lower left")
    tidy(ax)
    panel(ax, "b")

    ax = axes[1, 0]
    mmp = cs.get("matched_molecular_pairs") or {}
    if mmp:
        mp = load("a8_mmp_pairs.csv")
        if mp is not None and {"observed_delta", "predicted_delta"} <= set(mp.columns):
            ax.scatter(mp["observed_delta"], mp["predicted_delta"], s=22, color=C["hybrid"],
                       alpha=0.85, edgecolor="white", linewidth=0.4)
            lim = [-2.6, 2.6]
            ax.plot(lim, lim, "--", color=C["grey"], linewidth=0.8)
            ax.axhline(0, color=C["grey"], linewidth=0.5)
            ax.axvline(0, color=C["grey"], linewidth=0.5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.text(0.04, 0.95,
                f"$n$ = {mmp['n_matched_pairs_H_to_OH']} pairs\n"
                f"Spearman $\\rho$ = {mmp['spearman_observed_vs_predicted']:.2f}\n"
                f"sign agreement {mmp['sign_agreement_pct']:.0f}%",
                transform=ax.transAxes, va="top", fontsize=7.2)
    ax.set_xlabel("Observed $\\Delta$pIC$_{50}$ (H $\\rightarrow$ OH)")
    ax.set_ylabel("Predicted $\\Delta$pIC$_{50}$")
    ax.set_title("Matched-pair effect of\nadding one hydroxyl", loc="left", fontsize=8.8)
    tidy(ax)
    panel(ax, "c")

    ax = axes[1, 1]
    tri = cs["prospective_triage"]["results"]
    fracs = [5, 10, 20]
    for i, s in enumerate(SPLITS):
        if s not in tri:
            continue
        vals = [tri[s].get(f"enrichment_factor_top{fr}pct", np.nan) for fr in fracs]
        ax.plot(fracs, vals, "-o", markersize=4, linewidth=1.3, color=plt.cm.viridis(i / 4),
                label=SPLIT_LABEL[s].replace("-\n", "-"))
    ax.axhline(1.0, ls="--", color=C["grey"], linewidth=0.8)
    ax.set_xticks(fracs)
    ax.set_xlabel("Fraction of ranking screened (%)")
    ax.set_ylabel("Enrichment factor")
    ax.set_title("Triage enrichment on\nunseen chemistry", loc="left", fontsize=8.8)
    ax.legend(fontsize=6.2)
    tidy(ax)
    panel(ax, "d")

    fig.tight_layout(w_pad=2.6, h_pad=2.2)
    fig.savefig(os.path.join(FIG, "Fig6.png"))
    plt.close(fig)
    print("Fig6 written")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
