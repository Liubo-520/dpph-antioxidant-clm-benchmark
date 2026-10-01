"""Generate the figures that appear in the point-by-point response letter.

The response letter answers five reviewers, and several of its answers rest on
evidence that has no figure in the manuscript itself -- above all the audit of
the originally submitted feature matrix, which is the reason the headline
result is retracted. These figures are drawn from the same a1--a9 outputs as
the manuscript figures, so the letter, the manuscript and the supplementary
file cannot disagree.

Written to Revision_R1_new/figures under content-based names, so that a
change in where a figure appears in the letter cannot desynchronise its
filename from its float number.
"""

import json
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "bad": "#C44E52",
    "good": "#4C72B0",
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
SHORT_FAMILY = {
    "Fingerprint baseline": "ECFP / MACCS",
    "Descriptor baseline": "Descriptors",
    "Frozen chemical language model": "Frozen CLM",
    "Fine-tuned chemical language model": "Fine-tuned CLM",
    "Hybrid fp+desc": "Hybrid fp+desc",
    "Hybrid with CLM": "Hybrid + CLM",
    "Graph neural network": "Graph network",
}


PRETTY_METHOD = {
    "grad_x_input": "grad $\\times$ input",
    "integrated_gradients": "integ. grad.",
    "occlusion": "occlusion",
    "attention_rollout": "attn. rollout",
}
PRETTY_GROUP = {
    "phenolic_OH_oxygen": "phenolic O",
    "phenolic_OH_and_ipso_carbon": "phenolic O--H\n+ ipso C",
    "catechol_motif": "catechol",
    "aromatic": "aromatic",
    "carbonyl": "carbonyl",
    "conjugated_CC": "conjugated C=C",
    "non_aromatic": "non-aromatic",
    "aliphatic_OH_oxygen": "aliphatic O--H",
}
# the groups worth showing in a letter-sized panel
GROUP_ORDER = ["phenolic_OH_oxygen", "phenolic_OH_and_ipso_carbon", "catechol_motif",
               "carbonyl", "conjugated_CC", "aromatic"]


def panel(ax, letter):
    ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def load(name, default=None):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return default
    return json.load(open(p, encoding="utf-8")) if name.endswith(".json") else pd.read_csv(p)


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# ================================================ RespFig1: the leakage audit
def resp_figure1():
    """The evidence behind the retraction of R2 = 0.8759.

    Three independent lines: what the archived matrix actually correlates
    with, how its error splits between molecules the encoders had and had not
    seen, and what a clean probe gives on the same partition.
    """
    audit = load("a9_legacy_audit.json")
    if audit is None:
        print("RespLeakageAudit skipped (a9_legacy_audit.json missing)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.9))

    def short_source(k):
        k = k.lower()
        if "ten supervised" in k or "mean cls" in k:
            return "Mean CLS of the ten\nlabel-supervised\nfold encoders"
        if "fold-1" in k:
            return "One label-supervised\nfold encoder (CLS)"
        return "Frozen pretrained\nencoder (mean pooling)"

    def short_rep(k):
        k = k.lower()
        if "archived" in k:
            return "Archived\nfeatures"
        if "mordred" in k:
            return "Mordred"
        if "frozen" in k:
            return "Frozen\nencoder"
        if "ecfp" in k:
            return "ECFP4"
        return wrap(k, 12)

    # (a) provenance: correlation of the archived matrix with each candidate source
    ax = axes[0]
    pm = audit["provenance_match"]
    names, vals, cols = [], [], []
    for k, v in pm.items():
        names.append(short_source(k))
        vals.append(v["median_per_molecule_pearson_r"])
        cols.append(C["bad"] if "supervised" in k.lower() else C["good"])
    order = np.argsort(vals)
    ax.barh(range(len(order)), [vals[i] for i in order],
            color=[cols[i] for i in order], height=0.58)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=6.4)
    for y, i in enumerate(order):
        ax.text(vals[i] + 0.02, y, f"{vals[i]:.3f}", va="center", fontsize=7.2)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Median per-molecule Pearson $r$ with\nthe archived feature matrix")
    ax.set_title("What the submitted\nfeatures really were", loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "a")

    # (b) error on molecules the encoders had seen versus never seen
    ax = axes[1]
    ct = audit["contamination_test"]
    entries = [("Archived\nfeatures", ct["archived_features"], C["bad"])]
    for k, v in ct["label_free_controls"].items():
        entries.append((short_rep(k), v, C["good"]))
    x = np.arange(len(entries))
    w = 0.36
    seen = [e[1]["rmse_seen"] for e in entries]
    unseen = [e[1]["rmse_unseen"] for e in entries]
    ax.bar(x - w / 2, seen, w, color=[e[2] for e in entries], alpha=0.95,
           label=f"seen by the encoders ($n$ = {ct['n_test_seen_by_supervised_encoders']})")
    ax.bar(x + w / 2, unseen, w, color=[e[2] for e in entries], alpha=0.42, hatch="///",
           edgecolor="white", label=f"never seen ($n$ = {ct['n_test_never_seen']})")
    for i, e in enumerate(entries):
        ax.text(i, max(seen[i], unseen[i]) + 0.015,
                f"$\\times${e[1]['rmse_ratio_unseen_over_seen']:.2f}",
                ha="center", fontsize=7.2,
                fontweight="bold" if i == 0 else "normal",
                color=C["bad"] if i == 0 else "black")
    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in entries], fontsize=6.6)
    ax.set_ylabel("RMSE on the held-out compounds")
    ax.set_ylim(0, max(max(seen), max(unseen)) * 1.32)
    ax.set_title("Only the archived features\nsplit by prior exposure", loc="left", fontsize=8.6)
    ax.legend(fontsize=6.2, loc="upper left")
    tidy(ax)
    panel(ax, "b")

    # (c) the same ridge probe under clean conditions
    ax = axes[2]
    probe = audit["ridge_probe_on_stratified_split"]
    names, vals, cols = [], [], []
    for k, v in probe.items():
        names.append(short_rep(k))
        vals.append(v["test_r2"])
        cols.append(C["bad"] if "archived" in k.lower() else C["good"])
    ax.bar(range(len(vals)), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.014, f"{v:.3f}", ha="center", fontsize=7.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel("Held-out R$^2$, identical ridge probe")
    ax.set_ylim(0, 1.02)
    ax.set_title("The same probe on\nlabel-free inputs", loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "c")

    fig.tight_layout(w_pad=2.8)
    fig.savefig(os.path.join(FIG, "RespLeakageAudit.png"))
    plt.close(fig)
    print("RespLeakageAudit written")


# ============================ RespFig2: partitions and how accuracy degrades
def resp_figure2():
    splits_payload = load("a4_splits.json")
    chars = load("a4_split_characterisation.csv")
    if splits_payload is None or chars is None:
        print("RespGeneralisation skipped (partition outputs missing)")
        return
    frames = {s: load(f"a5_matrix_{s}.csv") for s in SPLITS}
    frames = {k: v for k, v in frames.items() if v is not None}
    chars = chars.set_index("split")

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.9))
    nn_by_split = {c["split"]: np.asarray(c.get("test_nn_tanimoto", []), float)
                   for c in splits_payload["characterisation"]}

    ax = axes[0]
    for i, s in enumerate(SPLITS):
        sim = nn_by_split.get(s, np.array([]))
        if sim.size == 0:
            continue
        parts = ax.violinplot([sim], positions=[i], widths=0.78, showextrema=False)
        for b in parts["bodies"]:
            b.set(facecolor=plt.cm.viridis(i / 4), alpha=0.78, edgecolor="white")
        ax.plot(i, np.median(sim), "o", color="black", markersize=3.2, zorder=4)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=6.4)
    ax.set_ylabel("Nearest-neighbour Tanimoto\nto the training set")
    ax.set_title("Five partitions, characterised\nbefore any modelling", loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "a")

    frames = {k: split_hybrid(v) for k, v in frames.items()}
    fams = [f for f in SHORT_FAMILY if any(f in set(t["family"]) for t in frames.values())]

    ax = axes[1]
    width = 0.9 / max(len(fams), 1)
    offset = (len(fams) - 1) / 2
    for k, fam in enumerate(fams):
        vals = []
        for s in SPLITS:
            sub = frames.get(s)
            v = sub[sub["family"] == fam]["test_r2"].max() if sub is not None else np.nan
            vals.append(v if np.isfinite(v) else np.nan)
        ax.bar(np.arange(len(SPLITS)) + (k - offset) * width, vals, width=width,
               color=FAMILY_COLOUR[fam], label=SHORT_FAMILY[fam])
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=6.4)
    ax.set_ylabel("Best held-out R$^2$")
    ax.set_title("Accuracy by\nrepresentation family", loc="left", fontsize=8.6)
    ax.legend(fontsize=6.2, ncol=2)
    tidy(ax)
    panel(ax, "b")

    ax = axes[2]
    for fam in fams:
        if "stratified" not in frames:
            continue
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
                label=SHORT_FAMILY[fam])
    ax.axhline(100, ls="--", color=C["grey"], linewidth=0.8)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=6.4)
    ax.set_ylabel("Accuracy retained (% of stratified R$^2$)")
    ax.set_title("What survives the\nstructural shift", loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "c")

    fig.tight_layout(w_pad=2.6)
    fig.savefig(os.path.join(FIG, "RespGeneralisation.png"))
    plt.close(fig)
    print("RespGeneralisation written")


# ================= RespFig3: representation versus learner, matched tuning
def resp_figure3():
    matrix = load("a5_matrix_stratified.csv")
    if matrix is None:
        print("RespMatrix skipped (benchmark matrix missing)")
        return

    grid = matrix[matrix["learner"].isin(["Ridge", "SVR", "ExtraTrees", "CatBoost"])]
    piv = grid.pivot_table(index="representation", columns="learner",
                           values="test_r2", aggfunc="max")
    order = [c for c in ["Ridge", "SVR", "ExtraTrees", "CatBoost"] if c in piv.columns]
    piv = piv[order]
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    data = piv.to_numpy(float)
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=np.nanmin(data),
                   vmax=np.nanmax(data))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=7.4)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=6.8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=6.4,
                        color="white" if data[i, j] < np.nanmedian(data) else "black")
    ax.set_title("Held-out R$^2$: every\ncell tuned identically", loc="left", fontsize=8.6)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("Held-out R$^2$", fontsize=7.4)
    panel(ax, "a")

    # how much of the variation is the learner rather than the representation
    ax = axes[1]
    spread = (piv.max(axis=1) - piv.min(axis=1)).dropna().sort_values()
    ax.barh(range(len(spread)), spread.to_numpy(), color=C["grey"], height=0.62)
    ax.set_yticks(range(len(spread)))
    ax.set_yticklabels(spread.index, fontsize=6.8)
    ax.set_xlabel("Range of held-out R$^2$ across learners,\nwith the representation held fixed")
    family_best = matrix.groupby("family")["test_r2"].max()
    fam_spread = float(family_best.max() - family_best.min())
    ax.axvline(fam_spread, color=C["bad"], ls="--", linewidth=1.1)
    # placed at the bottom, where the bars are short and the label sits clear
    ax.text(fam_spread, 0.15, f"  spread between\n  families: {fam_spread:.3f}",
            color=C["bad"], fontsize=6.8, va="bottom")
    ax.set_title("The learner moves accuracy as\nmuch as the representation",
                 loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "b")

    fig.tight_layout(w_pad=2.6)
    fig.savefig(os.path.join(FIG, "RespMatrix.png"))
    plt.close(fig)
    print("RespMatrix written")


# ============ RespFig4: attribution made quantitative, and practical utility
def resp_figure4():
    attr = load("a6_attribution.json")
    util = load("a8_chemspace_utility.json")
    if attr is None and util is None:
        print("RespAttributionUtility skipped (attribution and utility outputs missing)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.9))

    ax = axes[0]
    if attr is not None:
        rows = pd.DataFrame(attr["enrichment"])
        groups = [g for g in GROUP_ORDER if g in set(rows["atom_group"])]
        methods = list(dict.fromkeys(rows["method"]))
        width = 0.86 / max(len(methods), 1)
        offset = (len(methods) - 1) / 2
        for k, m in enumerate(methods):
            sub = rows[rows["method"] == m].set_index("atom_group")
            vals = [sub.loc[g, "mean_enrichment"] if g in sub.index else np.nan for g in groups]
            bars = ax.bar(np.arange(len(groups)) + (k - offset) * width, vals, width=width,
                          color=plt.cm.tab10(k), label=PRETTY_METHOD.get(m, m))
            for g, b in zip(groups, bars):
                if g in sub.index and sub.loc[g, "p_bh_adjusted"] < 0.05:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "*",
                            ha="center", va="bottom", fontsize=7)
        ax.axhline(1.0, ls="--", color=C["grey"], linewidth=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([PRETTY_GROUP.get(g, g).replace("\n", " ") for g in groups],
                           fontsize=6.2, rotation=28, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel("Attribution relative to a\nwithin-molecule permutation null")
        ax.legend(fontsize=6.0, ncol=2)
    ax.set_title("Attribution is selective,\nnot decorative", loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "a")

    ax = axes[1]
    if attr is not None:
        faith = pd.DataFrame(attr["faithfulness"])
        x = np.arange(len(faith))
        w = 0.36
        ax.bar(x - w / 2, faith["mean_abs_change_top20pct_atoms"], w,
               color=C["ft"], label="top 20% attributed atoms masked")
        ax.bar(x + w / 2, faith["mean_abs_change_random_atoms"], w,
               color=C["grey"], label="same number of random atoms")
        for i, r in faith.iterrows():
            ax.text(i, max(r["mean_abs_change_top20pct_atoms"],
                           r["mean_abs_change_random_atoms"]) * 1.03,
                    f"$\\times${r['ratio']:.2f}", ha="center", fontsize=6.8)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY_METHOD.get(m, m) for m in faith["method"]],
                           fontsize=6.4)
        ax.set_ylabel("Change in predicted pIC$_{50}$")
        ax.legend(fontsize=6.2)
    ax.set_title("Masking test: the atoms\ncarry the prediction", loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "b")

    ax = axes[2]
    if util is not None and util.get("prospective_triage", {}).get("results"):
        tri = util["prospective_triage"]["results"]
        names = [s for s in SPLITS if s in tri]
        fracs = [5, 10, 20]
        width = 0.86 / len(fracs)
        offset = (len(fracs) - 1) / 2
        for k, f in enumerate(fracs):
            vals = [tri[s].get(f"enrichment_factor_top{f}pct", np.nan) for s in names]
            ax.bar(np.arange(len(names)) + (k - offset) * width, vals, width=width,
                   color=plt.cm.Blues(0.45 + 0.2 * k), label=f"top {f}%")
        ax.axhline(1.0, ls="--", color=C["grey"], linewidth=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SPLIT_LABEL[s] for s in names], fontsize=6.0)
        ax.set_ylabel("Enrichment factor over random selection")
        ax.legend(fontsize=6.4)
    ax.set_title("Triage on chemistry the\nmodel has never seen", loc="left", fontsize=8.6)
    tidy(ax)
    panel(ax, "c")

    fig.tight_layout(w_pad=2.6)
    fig.savefig(os.path.join(FIG, "RespAttributionUtility.png"))
    plt.close(fig)
    print("RespAttributionUtility written")


if __name__ == "__main__":
    resp_figure1()
    resp_figure2()
    resp_figure3()
    resp_figure4()
