"""Figures of the second revision.

Figures 1 (dataset) and the attribution figure are unchanged from the first
revision and are copied as they are. The others are redrawn here, in the style
of b2_make_figures.py, from the second-round outputs:

  Fig2  benchmark matrix; panel c now shows the reported model, chosen by
        training Q2_CV, and plots the predictions the annotated statistics
        belong to
  Fig3  generalisation on the five primary partitions, each family represented
        by the cell that training Q2_CV selects
  Fig4  (new) stability of the family ranking on the repeated partitions
  Fig5  applicability domain and uncertainty of the reported model
  Fig6  attribution (unchanged; was Figure 5)
  Fig7  chemical space, pair-aware matched pairs and retrospective triage with
        bootstrap intervals

    python d2_figures.py            all figures
    python d2_figures.py 4          one figure
"""

import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import FAMILIES, OUT, PRIMARY, R1_FEAT, R1_OUT, R2, REP_FOLDS, REVISION, family_key, read_json

FIG = os.path.join(R2, "Revision_R2", "figures")
R1_FIG = os.path.join(REVISION, "一审", "Revision_R1_new", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.8, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False, "savefig.dpi": 400, "savefig.bbox": "tight",
    "figure.dpi": 120,
})

C = {"fp": "#4C72B0", "desc": "#DD8452", "frozen": "#937860", "ft": "#C44E52", "hybrid": "#55A868",
     "hybrid_clm": "#2E6F51", "ens": "#8172B3", "gnn": "#DA8BC3", "grey": "#7F7F7F"}
FAMILY_COLOUR = {
    "Fingerprint baseline": C["fp"], "Descriptor baseline": C["desc"],
    "Frozen chemical language model": C["frozen"], "Fine-tuned chemical language model": C["ft"],
    "Hybrid fp+desc": C["hybrid"], "Hybrid with CLM": C["hybrid_clm"], "Graph neural network": C["gnn"],
    "Descriptor ensemble member": C["ens"], "Matched descriptor ensemble": C["ens"],
}
KEY_COLOUR = {"Fp": C["fp"], "Desc": C["desc"], "Frozen": C["frozen"], "Ft": C["ft"], "Hyb": C["hybrid"],
              "HybClm": C["hybrid_clm"], "Gnn": C["gnn"]}
KEY_SHORT = {"Fp": "ECFP4", "Desc": "Descriptors", "Frozen": "Frozen CLM", "Ft": "Fine-tuned CLM",
             "Hyb": "Hybrid fp+desc", "HybClm": "Hybrid + CLM", "Gnn": "Graph network"}
FAM_KEYS = [k for k, _ in FAMILIES]
SPLIT_LABEL = {"stratified": "Chemistry-\nstratified", "scaffold": "Scaffold-\ndisjoint",
               "cluster": "Cluster-\ndisjoint", "dissimilarity": "Maximum-\ndissimilarity",
               "source": "Source-\ndisjoint"}
CLM_TOKENS = ("ChemBERTa", "MoLFormer")


def panel(ax, letter, x=-0.14):
    ax.text(x, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def split_hybrid(frame):
    out = frame.copy()
    is_clm = out["representation"].astype(str).apply(lambda r: any(t in r for t in CLM_TOKENS))
    out.loc[(out["family"] == "Hybrid") & ~is_clm, "family"] = "Hybrid fp+desc"
    out.loc[(out["family"] == "Hybrid") & is_clm, "family"] = "Hybrid with CLM"
    return out


def r1(name):
    p = os.path.join(R1_OUT, name)
    return read_json(p) if name.endswith(".json") else pd.read_csv(p)


def r2(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    return read_json(p) if name.endswith(".json") else pd.read_csv(p)


# ================================================== Fig 2: benchmark matrix
def figure2():
    matrix = split_hybrid(r1("a5_matrix_stratified.csv"))
    reported = r2("c9_reported_model_predictions.json")
    summary = r2("a7_model_summary.csv").set_index("model")

    n_cells = len(matrix[~matrix["family"].str.contains("ensemble member", na=False)])
    panel_a_in = max(3.4, 0.165 * n_cells)
    fig = plt.figure(figsize=(7.2, panel_a_in + 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[panel_a_in, 2.6], hspace=0.16, wspace=0.30)

    ax = fig.add_subplot(gs[0, :])
    m = matrix[~matrix["family"].str.contains("ensemble member", na=False)].copy()
    m = m.sort_values("test_r2")
    labels = m["representation"] + "  ·  " + m["learner"]
    labels = labels.str.replace("Ensemble mean (reproduced reference workflow)", "matched ensemble", regex=False)
    pos = np.arange(len(m))
    ax.barh(pos, m["test_r2"], color=[FAMILY_COLOUR.get(f, C["grey"]) for f in m["family"]], height=0.76)
    ax.plot(m["q2_cv_oof"], pos, "|", color="black", markersize=5, markeredgewidth=1.1)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_ylim(-0.8, len(m) - 0.2)
    ax.set_xlabel("Held-out R$^2$")
    ax.set_xlim(0, max(0.9, m["test_r2"].max() * 1.08))
    ax.set_title("Every representation $\\times$ learner cell, tuned inside the training partition", loc="left")
    short = {"Fingerprint baseline": "Fingerprints", "Descriptor baseline": "Descriptors",
             "Frozen chemical language model": "Frozen CLM", "Fine-tuned chemical language model": "Fine-tuned CLM",
             "Hybrid fp+desc": "Hybrid fp+desc", "Hybrid with CLM": "Hybrid + CLM",
             "Graph neural network": "Graph network", "Matched descriptor ensemble": "Matched ensemble"}
    handles = [Line2D([], [], marker="s", linestyle="", color=v, label=short[k])
               for k, v in FAMILY_COLOUR.items() if k in set(m["family"])]
    handles.append(Line2D([], [], marker="|", linestyle="", color="black", markeredgewidth=1.1,
                          label="Q$^2_{CV}$"))
    # the short bars of the bottom rows leave the lower right of the panel free
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, ncol=1, handletextpad=0.5, borderpad=0.4,
              bbox_to_anchor=(1.0, 0.0))
    tidy(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    for fam, colour in FAMILY_COLOUR.items():
        sub = matrix[matrix["family"] == fam]
        if len(sub):
            ax.scatter(sub["q2_cv_oof"], sub["test_r2"], s=26, color=colour, edgecolor="white",
                       linewidth=0.5, zorder=3)
    lo = min(matrix["q2_cv_oof"].min(), matrix["test_r2"].min()) - 0.05
    ax.plot([lo, 0.95], [lo, 0.95], "--", color=C["grey"], linewidth=0.8, zorder=1)
    ax.set_xlabel("Q$^2_{CV}$ (training)")
    ax.set_ylabel("Held-out R$^2$")
    ax.set_title("Held-out scores track\ncross-validation", loc="left", fontsize=8.8)
    tidy(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    yt, p = np.asarray(reported["y_test"], float), np.asarray(reported["test"], float)
    row = summary.loc[reported["model"]]
    ax.scatter(yt, p, s=20, color="#3B4A5A", alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
    lim = [min(yt.min(), p.min()) - 0.3, max(yt.max(), p.max()) + 0.3]
    ax.plot(lim, lim, "--", color=C["grey"], linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.text(0.04, 0.95, f"R$^2$ = {row['r2']:.3f}\nRMSE = {row['rmse']:.3f}\n$n$ = {len(yt)}",
            transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_title("Reported model (highest Q$^2_{CV}$)\non the held-out set", loc="left", fontsize=8.8)
    ax.set_xlabel("Experimental pIC$_{50}$")
    ax.set_ylabel("Predicted pIC$_{50}$")
    tidy(ax)
    panel(ax, "c")

    fig.savefig(os.path.join(FIG, "Fig2.png"))
    plt.close(fig)
    print("Fig2 written")


# ============================================== Fig 3: generalisation
def figure3():
    sel = r2("c5_selection_primary.json")["partitions"]
    chars = r1("a4_split_characterisation.csv").set_index("split")
    nn_by_split = {c["split"]: np.asarray(c["test_nn_tanimoto"], float)
                   for c in r1("a4_splits.json")["characterisation"]}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))

    ax = axes[0, 0]
    for i, s in enumerate(PRIMARY):
        parts = ax.violinplot([nn_by_split[s]], positions=[i], widths=0.75, showextrema=False)
        for b in parts["bodies"]:
            b.set(facecolor=plt.cm.viridis(i / 4), alpha=0.75, edgecolor="white")
        ax.plot(i, np.median(nn_by_split[s]), "o", color="black", markersize=3.2, zorder=4)
    ax.set_xticks(range(len(PRIMARY)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in PRIMARY], fontsize=6.6)
    ax.set_ylabel("Nearest-neighbour Tanimoto\nto the training set")
    ax.set_title("Partitions of increasing\nstructural difficulty", loc="left", fontsize=8.8)
    tidy(ax)
    panel(ax, "a")

    val = {fam: [sel[s]["families"][fam]["test_r2"] for s in PRIMARY] for fam in FAM_KEYS}

    ax = axes[0, 1]
    width = 0.9 / len(FAM_KEYS)
    offset = (len(FAM_KEYS) - 1) / 2
    for k, fam in enumerate(FAM_KEYS):
        ax.bar(np.arange(len(PRIMARY)) + (k - offset) * width, val[fam], width=width,
               color=KEY_COLOUR[fam], label=KEY_SHORT[fam])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(PRIMARY)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in PRIMARY], fontsize=6.6)
    ax.set_ylabel("Held-out R$^2$ of the Q$^2_{CV}$-selected model")
    ax.set_title("Accuracy by representation family", loc="left")
    tidy(ax)
    panel(ax, "b")

    ax = axes[1, 0]
    for fam in FAM_KEYS:
        ax.plot(range(len(PRIMARY)), [100 * v / val[fam][0] for v in val[fam]], "-o",
                color=KEY_COLOUR[fam], markersize=3.6, linewidth=1.3, label=KEY_SHORT[fam])
    ax.axhline(100, ls="--", color=C["grey"], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(PRIMARY)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in PRIMARY], fontsize=6.6)
    ax.set_ylabel("Accuracy retained (% of stratified R$^2$)")
    ax.set_title("Accuracy retained under structural shift", loc="left")
    tidy(ax)
    panel(ax, "c")

    ax = axes[1, 1]
    for fam in FAM_KEYS:
        ax.plot([chars.loc[s, "test_nn_tanimoto_median"] for s in PRIMARY], val[fam], "o",
                color=KEY_COLOUR[fam], markersize=5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Median nearest-neighbour Tanimoto of the partition")
    ax.set_ylabel("Held-out R$^2$ of the Q$^2_{CV}$-selected model")
    ax.set_title("Accuracy tracks analogue overlap", loc="left")
    tidy(ax)
    panel(ax, "d")

    handles = [Line2D([], [], marker="s", linestyle="", color=KEY_COLOUR[f], label=KEY_SHORT[f]) for f in FAM_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=6.6, handletextpad=0.3,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(w_pad=2.4, h_pad=2.2)
    fig.savefig(os.path.join(FIG, "Fig3.png"))
    plt.close(fig)
    print("Fig3 written")


# ==================================== Fig 4: repeated partitions (new)
REP_DESIGNS = ("stratified", "scaffold", "cluster", "source")


def figure4():
    rep = r2("c8_repeated_summary.json")
    sel = r2("c5_selection_primary.json")
    if rep is None or any("families" not in rep["nested"]["designs"].get(d, {}) for d in REP_DESIGNS):
        print("Fig4 skipped (repeated partitions not complete)")
        return
    N = rep["nested"]["designs"]
    S = rep.get("same_rule", {}).get("nested", {}).get("designs", {})
    P = sel["partitions"]

    # (key, source dict, design, tick label, show the primary partition as a sibling)
    groups = [("strat", N, "stratified", "Chemistry-\nstratified", True)]
    short = ["Stratified"]
    for d, name in (("scaffold", "Scaffold-disjoint"), ("cluster", "Cluster-disjoint")):
        if "families" in S.get(d, {}):
            groups.append((f"{d}_iso", S, d, f"{name},\nisolated\ncompounds", True))
            short.append(name.split("-")[0] + ",\nisolated")
    n_iso = len(groups) - 1
    for d, name in (("scaffold", "Scaffold-disjoint"), ("cluster", "Cluster-disjoint"),
                    ("source", "Source-disjoint")):
        groups.append((f"{d}_whole", N, d, f"{name},\nwhole groups", False))
        short.append(name.split("-")[0] + ",\nwhole\ngroups")

    def values(src, d, fam):
        st = src[d]["families"][fam]
        return np.asarray(st.get("r2_by_fold", st.get("r2_by_partition")))

    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.62, wspace=0.34)

    # a: every partition, every family
    ax = fig.add_subplot(gs[0, :])
    width = 0.115
    offset = (len(FAM_KEYS) - 1) / 2
    for gi, (_key, src, d, _label, sibling) in enumerate(groups):
        for k, fam in enumerate(FAM_KEYS):
            x = gi + (k - offset) * width
            v = values(src, d, fam)
            ax.plot(np.full(len(v), x), v, "o", color=KEY_COLOUR[fam], markersize=2.8, alpha=0.55,
                    markeredgewidth=0)
            ax.plot([x - 0.045, x + 0.045], [v.mean(), v.mean()], "-", color=KEY_COLOUR[fam], linewidth=2.0)
            if sibling:
                ax.plot(x, P[d]["families"][fam]["test_r2"], "D", markerfacecolor="white",
                        markeredgecolor="black", markersize=3.2, markeredgewidth=0.7, zorder=5)
    ax.axhline(0, color="black", linewidth=0.6)
    if n_iso:
        ax.axvline(n_iso + 0.5, ymin=0.19, color=C["grey"], linewidth=0.7, linestyle=":")
        top = ax.get_ylim()[1]
        ax.text((n_iso) / 2, top, "the rule of the primary partitions, redrawn", ha="center", va="bottom",
                fontsize=6.6, color="#444444")
        ax.text(n_iso + 0.5 + (len(groups) - n_iso - 1) / 2, top, "whole groups of any size held out",
                ha="center", va="bottom", fontsize=6.6, color="#444444")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[3] for g in groups], fontsize=6.6)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylabel("Held-out R$^2$")
    handles = [Line2D([], [], marker="s", linestyle="", color=KEY_COLOUR[f], label=KEY_SHORT[f]) for f in FAM_KEYS]
    handles += [Line2D([], [], marker="o", linestyle="", color=C["grey"], markersize=3.5, label="one partition"),
                Line2D([], [], linestyle="-", color=C["grey"], linewidth=2, label="mean"),
                Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="black",
                       markersize=3.6, label="primary partition")]
    ax.legend(handles=handles, loc="lower left", ncol=5, fontsize=6.2, handletextpad=0.3, columnspacing=0.9)
    tidy(ax)
    panel(ax, "a", x=-0.065)

    # b: mean rank of each family in each set of partitions
    ax = fig.add_subplot(gs[1, 0])
    M = np.array([[src[d]["families"][fam]["rank_mean"] for (_k, src, d, _l, _s) in groups] for fam in FAM_KEYS])
    im = ax.imshow(M, cmap="YlGnBu_r", vmin=1, vmax=7, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.6,
                    color="white" if M[i, j] < 3.2 else "black")
    if n_iso:
        ax.axvline(n_iso + 0.5, color="white", linewidth=2.0)
    ax.set_yticks(range(len(FAM_KEYS)))
    ax.set_yticklabels([KEY_SHORT[f] for f in FAM_KEYS], fontsize=6.6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(short, fontsize=6.0)
    ax.set_title("Mean rank among the seven families\n(1 is the most accurate)", loc="left", fontsize=8.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel(ax, "b", x=-0.36)

    # c: the differences the reversal is made of, partition by partition
    ax = fig.add_subplot(gs[1, 1])
    for j, (pair, colour, label) in enumerate((("Frozen-Hyb", C["hybrid"], "Frozen CLM $-$ hybrid fp+desc"),
                                               ("Frozen-Fp", C["fp"], "Frozen CLM $-$ ECFP4"))):
        a, b = pair.split("-")
        for gi, (_key, src, d, _label, sibling) in enumerate(groups):
            x = gi + (j - 0.5) * 0.34
            diff = values(src, d, a) - values(src, d, b)
            c = src[d]["paired"][pair]
            ax.plot(np.full(len(diff), x), diff, "o", color=colour, markersize=3.0, alpha=0.55, markeredgewidth=0)
            ax.errorbar(x + 0.10, c["mean_delta"], yerr=[[c["mean_delta"] - c["ci95"][0]],
                                                         [c["ci95"][1] - c["mean_delta"]]],
                        fmt="s", color=colour, markersize=3.6, capsize=1.8, linewidth=1.0,
                        label=label if gi == 0 else None)
            if sibling:
                ax.plot(x - 0.10, P[d]["contrasts"][pair]["delta_r2"], "D", markerfacecolor="white",
                        markeredgecolor=colour, markersize=3.4, markeredgewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(short, fontsize=6.0)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylabel("Difference in held-out R$^2$")
    ax.set_title("Frozen language model against\nthe fingerprint-based families", loc="left", fontsize=8.6)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.24 * (hi - lo))          # head-room for the legend
    if n_iso:
        ax.axvline(n_iso + 0.5, ymax=0.80, color=C["grey"], linewidth=0.7, linestyle=":")
    ax.legend(loc="upper left", fontsize=6.2, handletextpad=0.4)
    tidy(ax)
    panel(ax, "c", x=-0.20)

    fig.savefig(os.path.join(FIG, "Fig4.png"))
    plt.close(fig)
    print("Fig4 written")


# ====================================== Fig 5: applicability domain
def figure5():
    sa = r2("a7_stats_ad.json")
    ad, sim = sa["applicability_domain_williams"], sa["applicability_domain_similarity"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    ax = axes[0, 0]
    lev = np.asarray(ad["leverage"], float)
    sr = np.asarray(ad["standardised_residual"], float)
    inside = (lev <= ad["h_star"]) & (np.abs(sr) <= 3)
    ax.scatter(lev[inside], sr[inside], s=20, color=C["fp"], alpha=0.8, edgecolor="white", linewidth=0.4,
               label=f"Inside domain ($n$ = {inside.sum()})")
    ax.scatter(lev[~inside], sr[~inside], s=26, color=C["ft"], alpha=0.9, edgecolor="white", linewidth=0.4,
               label=f"Outside ($n$ = {(~inside).sum()})")
    ax.axvline(ad["h_star"], ls="--", color=C["grey"], linewidth=0.9)
    ax.axhline(3, ls=":", color=C["grey"], linewidth=0.9)
    ax.axhline(-3, ls=":", color=C["grey"], linewidth=0.9)
    ax.text(ad["h_star"], ax.get_ylim()[1], f"  $h^*$ = {ad['h_star']:.3f}", va="top", fontsize=7)
    xmax = max(4 * ad["h_star"], float(np.percentile(lev, 97)))
    beyond = int((lev > xmax).sum())
    if beyond:
        ax.set_xlim(-0.02 * xmax, xmax * 1.05)
        ax.set_xlabel(f"Leverage ({beyond} compounds up to {lev.max():.1f} lie beyond the axis)")
    else:
        ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardised residual")
    ax.set_title(f"Williams plot, {ad['n_test']} held-out compounds", loc="left")
    ax.legend(loc="lower right", fontsize=6.8)
    tidy(ax)
    panel(ax, "a")

    ax = axes[0, 1]
    nn, err = np.asarray(sim["nn_tanimoto"], float), np.asarray(sim["abs_error"], float)
    ax.scatter(nn, err, s=20, color=C["desc"], alpha=0.8, edgecolor="white", linewidth=0.4)
    fit = np.poly1d(np.polyfit(nn, err, 1))
    xs = np.linspace(nn.min(), nn.max(), 50)
    ax.plot(xs, fit(xs), "-", color="black", linewidth=1.1)
    ax.set_xlabel("Nearest-neighbour Tanimoto to training set")
    ax.set_ylabel("Absolute prediction error")
    ax.set_title("Error grows with structural novelty", loc="left")
    ax.text(0.96, 0.94, f"Spearman $\\rho$ = {sim['spearman_similarity_vs_abs_error']:.3f}\n"
                        f"$p$ = {sim['spearman_p']:.1e}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.2)
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
        ax.plot([100 * v["target_coverage"] for v in d.values()],
                [100 * v["empirical_coverage"] for v in d.values()], marker, color=colour, markersize=6,
                label=label)
    ax.plot([75, 95], [75, 95], "--", color=C["grey"], linewidth=0.8)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Conformal intervals are calibrated", loc="left")
    ax.legend(loc="lower right", fontsize=6.8)
    tidy(ax)
    panel(ax, "d")

    fig.tight_layout(w_pad=2.4, h_pad=2.2)
    fig.savefig(os.path.join(FIG, "Fig5.png"))
    plt.close(fig)
    print("Fig5 written")


# ============================================ Fig 7: chemical space, utility
def figure7():
    cs = r1("a8_chemspace_utility.json")
    mmp = r2("c6_mmp.json")
    pairs = r2("c6_mmp_pairs.csv")
    tri = r2("c7_triage.json")
    y_all = pd.read_csv(os.path.join(R1_OUT, "curated_dataset.csv"))["pIC50"].to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    ax = axes[0, 0]
    emb = np.load(os.path.join(R1_FEAT, "umap_reference_embedding.npy"))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_all, s=6, cmap="viridis", alpha=0.85, linewidth=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("pIC$_{50}$", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])
    u = cs["umap_sensitivity"]
    pres = np.mean([s["neighbourhood_preservation_k15"] for s in u])
    ax.set_title(f"UMAP overview only\n({100 * pres:.0f}% of neighbourhoods kept)", loc="left", fontsize=8.8)
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, "a")

    ax = axes[0, 1]
    scan = pd.DataFrame(u)
    for nn_, grp in scan.groupby("n_neighbors"):
        ax.plot(grp["min_dist"], grp["neighbourhood_preservation_k15"], "-o", markersize=3.6, linewidth=1.2,
                label=f"n_neighbors = {nn_}")
    ax2 = ax.twinx()
    ax2.plot(scan["min_dist"], scan["category_silhouette_2d"], "x", color=C["ft"], markersize=4)
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
    s = mmp["protocols"]["RDKit-desc | ExtraTrees"]["component_aware"]
    ax.scatter(pairs["observed_delta"], pairs["predicted_delta_component_aware"], s=22, color=C["hybrid"],
               alpha=0.85, edgecolor="white", linewidth=0.4)
    lim = [-2.6, 2.6]
    ax.plot(lim, lim, "--", color=C["grey"], linewidth=0.8)
    ax.axhline(0, color=C["grey"], linewidth=0.5)
    ax.axvline(0, color=C["grey"], linewidth=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.text(0.04, 0.95, f"$n$ = {s['n_pairs']} pairs, pair-aware\nSpearman $\\rho$ = {s['spearman']:.2f}\n"
                        f"sign agreement {s['sign_agreement_pct']:.0f}%",
            transform=ax.transAxes, va="top", fontsize=7.2)
    ax.set_xlabel("Observed $\\Delta$pIC$_{50}$ (H $\\rightarrow$ OH)")
    ax.set_ylabel("Predicted $\\Delta$pIC$_{50}$")
    ax.set_title("Matched-pair effect of adding one hydroxyl,\nneither member seen in training", loc="left",
                 fontsize=8.8)
    tidy(ax)
    panel(ax, "c")

    ax = axes[1, 1]
    fracs = [5, 10, 20]
    for i, sp in enumerate(PRIMARY):
        t = tri["primary"][sp]
        vals = [t[f"ef_top{fr}"] for fr in fracs]
        lo = [t[f"ef_top{fr}"] - t[f"ef_top{fr}_ci95"][0] for fr in fracs]
        hi = [t[f"ef_top{fr}_ci95"][1] - t[f"ef_top{fr}"] for fr in fracs]
        x = np.array(fracs) + (i - 2) * 0.42
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="-o", markersize=3.6, linewidth=1.1, capsize=1.8,
                    elinewidth=0.8, color=plt.cm.viridis(i / 4), label=SPLIT_LABEL[sp].replace("-\n", "-"))
    ax.axhline(1.0, ls="--", color=C["grey"], linewidth=0.8)
    ax.set_xticks(fracs)
    ax.set_xlabel("Fraction of ranking screened (%)")
    ax.set_ylabel("Enrichment factor (95% CI)")
    ax.set_title("Retrospective triage enrichment\non held-out compounds", loc="left", fontsize=8.8)
    ax.legend(fontsize=6.0)
    tidy(ax)
    panel(ax, "d")

    fig.tight_layout(w_pad=2.6, h_pad=2.2)
    fig.savefig(os.path.join(FIG, "Fig7.png"))
    plt.close(fig)
    print("Fig7 written")


def carry_over():
    """Figures that did not change: the dataset figure and the attribution figure."""
    shutil.copy2(os.path.join(R1_FIG, "Fig1.png"), os.path.join(FIG, "Fig1.png"))
    shutil.copy2(os.path.join(R1_FIG, "Fig5.png"), os.path.join(FIG, "Fig6.png"))   # attribution: 5 -> 6
    print("Fig1 and Fig6 carried over unchanged")


if __name__ == "__main__":
    which = set(sys.argv[1:])
    todo = {"2": figure2, "3": figure3, "4": figure4, "5": figure5, "7": figure7}
    if not which:
        carry_over()
    for key, fn in todo.items():
        if not which or key in which:
            fn()
