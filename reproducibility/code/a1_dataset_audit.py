"""
Dataset provenance, curation and endpoint-homogeneity audit.

Answers Reviewer 4 Q1/Q3/Q15 and Reviewer 5 Q5: where the 1,911 DPPH records
come from, how heterogeneous the assay protocol is, how duplicates and
conflicting measurements were handled, how the eight chemistry-aware
categories are defined, and how activity is distributed across the split.
"""

import json
import os
import pickle
import re
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "code"))

from importlib.machinery import SourceFileLoader

chem_space = SourceFileLoader(
    "chem_space", os.path.join(ROOT, "code", "07_chemical_space.py")
).load_module()


def load_frames():
    raw = pd.read_excel(os.path.join(ROOT, "DPPH_30min_Dataset.xlsx"))
    raw.columns = [c.strip() for c in raw.columns]
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        proc = pickle.load(fh)
    return raw, proc["df"]


def canonical(smi, keep_stereo=True):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=keep_stereo)


def inchikey(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def main():
    raw, df = load_frames()
    smiles = df["smiles"].tolist()
    y = df["pIC50"].to_numpy(float)

    report = {}

    # ---------------------------------------------------------------- sources
    doi = raw["DOI"].astype(str).str.strip()
    assay = raw["Assay Description"].astype(str).str.strip()
    chembl = raw["ChEMBL ID"].astype(str).str.strip()

    doi_counts = Counter(doi)
    assay_counts = Counter(assay)
    n_src = len(doi_counts)
    sizes = np.array(sorted(doi_counts.values())[::-1])

    report["provenance"] = {
        "n_records": int(len(raw)),
        "n_unique_source_publications": int(n_src),
        "n_unique_chembl_ids": int(chembl.nunique()),
        "records_per_source_median": float(np.median(sizes)),
        "records_per_source_iqr": [float(np.percentile(sizes, 25)), float(np.percentile(sizes, 75))],
        "records_per_source_max": int(sizes.max()),
        "records_per_source_min": int(sizes.min()),
        "share_from_largest_source_pct": float(100 * sizes.max() / len(raw)),
        "share_from_top10_sources_pct": float(100 * sizes[:10].sum() / len(raw)),
        "n_sources_with_single_record": int((sizes == 1).sum()),
        "n_unique_assay_descriptions": int(len(assay_counts)),
        "assay_description_counts": {k: int(v) for k, v in assay_counts.most_common()},
        "top_sources": [
            {"doi": k, "n": int(v)} for k, v in doi_counts.most_common(15)
        ],
    }

    # every record must be a 30-min DPPH radical-scavenging IC50
    has_dpph = assay.str.contains("DPPH", case=False).mean()
    has_30 = assay.str.contains("30", case=False).mean()
    report["provenance"]["fraction_assay_text_mentions_DPPH"] = float(has_dpph)
    report["provenance"]["fraction_assay_text_mentions_30_min"] = float(has_30)

    # ------------------------------------------------------- endpoint / units
    mw = raw["Molecular Weight"].to_numpy(float)
    ic50_molar = 10.0 ** (-raw["minusLogIC50 (M)"].to_numpy(float))
    report["endpoint"] = {
        "definition": "pIC50 = -log10(IC50 in mol/L) for 30-min DPPH radical scavenging",
        "pIC50_mean": float(y.mean()),
        "pIC50_sd": float(y.std(ddof=1)),
        "pIC50_median": float(np.median(y)),
        "pIC50_min": float(y.min()),
        "pIC50_max": float(y.max()),
        "pIC50_iqr": [float(np.percentile(y, 25)), float(np.percentile(y, 75))],
        "ic50_micromolar_median": float(np.median(ic50_molar) * 1e6),
        "ic50_micromolar_range": [float(ic50_molar.min() * 1e6), float(ic50_molar.max() * 1e6)],
        "ic50_log_range_orders": float(np.log10(ic50_molar.max() / ic50_molar.min())),
        "mw_median": float(np.median(mw)),
        "mw_range": [float(mw.min()), float(mw.max())],
        "skew_ic50": float(stats.skew(ic50_molar)),
        "skew_pic50": float(stats.skew(y)),
    }

    # --------------------------------------------------------- duplicate audit
    keys_stereo = [canonical(s, True) for s in smiles]
    keys_flat = [canonical(s, False) for s in smiles]
    ikeys = [inchikey(s) for s in smiles]
    ikeys_skeleton = [k.split("-")[0] if k else None for k in ikeys]

    def dup_report(keys, label):
        buckets = defaultdict(list)
        for i, k in enumerate(keys):
            if k is not None:
                buckets[k].append(i)
        dups = {k: v for k, v in buckets.items() if len(v) > 1}
        spreads = [float(np.ptp(y[v])) for v in dups.values()]
        return {
            "level": label,
            "n_unique": int(len(buckets)),
            "n_duplicate_groups": int(len(dups)),
            "n_records_in_duplicate_groups": int(sum(len(v) for v in dups.values())),
            "max_pIC50_spread_within_group": float(max(spreads)) if spreads else 0.0,
            "median_pIC50_spread_within_group": float(np.median(spreads)) if spreads else 0.0,
            "n_groups_spread_gt_1_log": int(sum(s > 1.0 for s in spreads)),
        }

    report["duplicates"] = {
        "isomeric_smiles": dup_report(keys_stereo, "isomeric canonical SMILES"),
        "flat_smiles": dup_report(keys_flat, "stereochemistry-stripped SMILES"),
        "inchikey_full": dup_report(ikeys, "full InChIKey"),
        "inchikey_skeleton": dup_report(ikeys_skeleton, "InChIKey skeleton (first block)"),
    }

    # structural sanity of the delivered structures
    n_valid = sum(Chem.MolFromSmiles(s) is not None for s in smiles)
    frag_counts = [len(Chem.GetMolFrags(Chem.MolFromSmiles(s))) for s in smiles]
    charges = [Chem.GetFormalCharge(Chem.MolFromSmiles(s)) for s in smiles]
    organic = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Se", "Si", "B"}
    non_organic = 0
    for s in smiles:
        els = {a.GetSymbol() for a in Chem.MolFromSmiles(s).GetAtoms()}
        if not els.issubset(organic):
            non_organic += 1
    report["structure_qc"] = {
        "n_parsable": int(n_valid),
        "n_multi_fragment_after_curation": int(sum(c > 1 for c in frag_counts)),
        "n_charged_after_curation": int(sum(c != 0 for c in charges)),
        "n_with_non_standard_elements": int(non_organic),
        "n_with_defined_stereocentres": int(
            sum(
                len(Chem.FindMolChiralCenters(Chem.MolFromSmiles(s), useLegacyImplementation=False)) > 0
                for s in smiles
            )
        ),
    }

    # -------------------------------------------------- category definitions
    categories = np.array([chem_space.classify_compound(s) for s in smiles], dtype=object)
    strat = json.load(open(os.path.join(ROOT, "results", "stratified_split_llm_results.json"), encoding="utf-8"))
    train_idx = np.array(strat["train_idx"])
    test_idx = np.array(strat["test_idx"])

    cat_rows = []
    for cat in sorted(set(categories)):
        m = categories == cat
        cat_rows.append(
            {
                "category": cat,
                "n_total": int(m.sum()),
                "pct_total": float(100 * m.mean()),
                "n_train": int((categories[train_idx] == cat).sum()),
                "n_test": int((categories[test_idx] == cat).sum()),
                "mean_pIC50": float(y[m].mean()),
                "sd_pIC50": float(y[m].std(ddof=1)) if m.sum() > 1 else 0.0,
                "min_pIC50": float(y[m].min()),
                "max_pIC50": float(y[m].max()),
                "n_source_publications": int(doi[m].nunique()),
            }
        )
    report["categories"] = cat_rows
    report["category_definitions"] = [
        {"category": name, "smarts": pats} for name, pats in chem_space.CATEGORY_PATTERNS
    ]
    report["category_definition_note"] = (
        "First-match-wins SMARTS cascade evaluated in the listed priority order; "
        "molecules matching no pattern fall into the residual 'Other' class. "
        "Category labels are computed from structure only, never from pIC50, and "
        "are used solely to stratify the split and to report class-wise metrics."
    )

    # leakage check: are category labels informative about the label?
    groups = [y[categories == c] for c in sorted(set(categories))]
    f_stat, p_anova = stats.f_oneway(*groups)
    ss_between = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
    report["category_label_check"] = {
        "anova_F": float(f_stat),
        "anova_p": float(p_anova),
        "eta_squared": float(ss_between / ((y - y.mean()) ** 2).sum()),
        "note": (
            "Categories carry a small amount of activity information (eta^2 reported), "
            "which is why they are used only to balance the split and never as model input. "
            "Because stratification is applied to structure-derived labels and the test "
            "labels are untouched, this cannot leak test activity into training."
        ),
    }

    # ------------------------------------------------- train / test endpoint
    ks = stats.ks_2samp(y[train_idx], y[test_idx])
    report["split_endpoint_distribution"] = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_mean": float(y[train_idx].mean()),
        "train_sd": float(y[train_idx].std(ddof=1)),
        "test_mean": float(y[test_idx].mean()),
        "test_sd": float(y[test_idx].std(ddof=1)),
        "train_range": [float(y[train_idx].min()), float(y[train_idx].max())],
        "test_range": [float(y[test_idx].min()), float(y[test_idx].max())],
        "ks_statistic": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "levene_p": float(stats.levene(y[train_idx], y[test_idx]).pvalue),
    }

    # ---------------------------------------- source structure for split work
    scaffolds = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            sc = ""
        scaffolds.append(sc if sc else "<acyclic>")
    sc_counts = Counter(scaffolds)
    report["scaffold_inventory"] = {
        "n_unique_bemis_murcko_scaffolds": int(len(sc_counts)),
        "n_singleton_scaffolds": int(sum(1 for v in sc_counts.values() if v == 1)),
        "largest_scaffold_size": int(max(sc_counts.values())),
        "n_acyclic": int(sc_counts.get("<acyclic>", 0)),
        "top_scaffolds": [{"scaffold": k, "n": int(v)} for k, v in sc_counts.most_common(10)],
    }

    curated = pd.DataFrame(
        {
            "compound_index": np.arange(len(smiles)),
            "canonical_smiles": smiles,
            "inchikey": ikeys,
            "chembl_id": chembl.values,
            "source_doi": doi.values,
            "assay_description": assay.values,
            "molecular_weight": mw,
            "ic50_M": ic50_molar,
            "pIC50": y,
            "chemistry_category": categories,
            "bemis_murcko_scaffold": scaffolds,
            "split_stratified_9_1": np.where(np.isin(np.arange(len(smiles)), test_idx), "test", "train"),
        }
    )
    curated.to_csv(os.path.join(OUT, "curated_dataset.csv"), index=False)

    with open(os.path.join(OUT, "a1_dataset_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in report.items() if k not in ("category_definitions",)}, indent=2)[:6000])
    print("\nWrote", os.path.join(OUT, "a1_dataset_audit.json"))
    print("Wrote", os.path.join(OUT, "curated_dataset.csv"))


if __name__ == "__main__":
    main()
