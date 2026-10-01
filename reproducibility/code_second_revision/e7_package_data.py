"""Assemble the reproducibility deposit of the second revision.

The deposit of the first revision is taken over unchanged (curated dataset,
the five primary partitions, every representation matrix, all model outputs and
the analysis code of that round), and the material of the second revision is
added beside it: the repeated partitions, the per-partition model outputs, the
training-only selection, the pair-aware matched-pair test, the triage intervals
and the code that produced them. The same folder is what goes to the
DOI-assigning repository named in the Code availability statement.

    python e7_package_data.py
"""

import os
import shutil
import zipfile

import pandas as pd

from common import (DESIGNS, OUT, R2, REP_FOLDS, REVISION, ROOT, SAME_RULE_DESIGNS, SAME_RULE_SEEDS, read_json,
                    same_rule_ids)

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(R2, "Revision_R2")
REPRO = os.path.join(PKG, "reproducibility")
# Scripts that serve the correspondence with the journal (the Word copy of the
# response letter, the marked-up manuscript, the upload folder). They need the
# LaTeX sources, which are not part of the deposit, and reproduce no result.
CORRESPONDENCE_TOOLS = {"e3_response_docx.py", "e4_marked_manuscript.py", "e5_submission_package.py"}
R1_REPRO = os.path.join(REVISION, "一审", "Revision_R1_new", "reproducibility")

README = """# Reproducibility deposit

Supporting data and code for the manuscript *{title}*.

## Contents

```
data/                        curated dataset with provenance, the five primary
                             train/test partitions, the repeated partitions,
                             partition characterisation and every
                             representation matrix
results/                     analysis outputs of the first revision (JSON and
                             CSV) and flat prediction tables
results_second_revision/     outputs of the analyses added in the second
                             revision, including the per-partition model outputs
                             of the repeated partitions
code/                        the analysis pipeline of the first revision
code_second_revision/        the code of the analyses added in the second
                             revision
environment.txt              Python, package and hardware versions
```

## What the second revision added

| Script | Purpose |
| --- | --- |
| `common.py` | paths, family definitions and the list of repeated partitions |
| `c1_partitions.py` | build and characterise the repeated group-aware 9:1 partitions |
| `c1b_partitions_same_rule.py` | replicate the primary scaffold- and cluster-disjoint partitions under the rule that built them |
| `c2_grid.py` | representation x learner cells on the repeated partitions, tuned inside each training partition |
| `c3_finetune.py` | leak-free fine-tuned ChemBERTa-ZINC on the repeated partitions |
| `c4_gnn.py` | AttentiveFP graph network on the repeated partitions |
| `c5_selection_primary.py` | family representatives on the five primary partitions, chosen by training-set Q2_CV |
| `c6_mmp.py` | matched-molecular-pair test under molecule-level, pair-aware and scaffold-aware prediction |
| `c7_triage.py` | retrospective triage simulation with bootstrap intervals and chance probabilities |
| `c8_repeated_summary.py` | stability of the family ranking across the repeated partitions |
| `c9_stats_ad.py` | statistics, applicability domain and uncertainty for the model chosen by training Q2_CV |
| `d1_macros.py` | every reported number as a LaTeX macro or table body |
| `d2_figures.py` | the manuscript figures that changed or are new |
| `c0_reproduce_primary.py` | check that the code used on the repeated partitions returns the first-revision values on primary partitions |
| `e1_crossrefs.py`, `e2_lint.py` | section, table and figure numbers of the manuscript as macros; a lint of the LaTeX sources |
| `e6_verify_claims.py` | re-derive every directional statement of the text from the outputs |
| `e7_package_data.py`, `e8_github_release.py` | assemble this deposit and the public repository it is archived from |

The scripts of the second revision read the outputs of the first (`results/`,
`data/`) and never modify them. `c2`, `c3` and `c4` refit every model inside
each repeated training partition; everything else is computed from stored
predictions.

## Model selection

No model is chosen with held-out information. On the primary partitions the
representative of a family is the model with the highest training-set Q2_CV in
that family, and the reported model is the model with the highest Q2_CV of
all. On the repeated partitions the representative is re-selected inside every
partition by the inner cross-validated R2 of the grid search.

## Repeated partitions

`data/repeated_partitions.csv` lists, for each of the four designs
(chemistry-stratified, scaffold-disjoint, cluster-disjoint, source-disjoint),
the {n_rep} hold-outs used: the compounds were divided into ten parts of equal
size, whole groups to one part in the three group-disjoint designs, and parts
{folds} served in turn as the hold-out, so every partition is 9:1 and the
hold-outs of a design do not overlap. `data/repeated_partitions_all_folds.json`
holds the assignment of every compound to one of the ten parts.

`data/primary_rule_replicates.csv` lists a second set: {n_sr} replicates of each of
the primary scaffold- and cluster-disjoint partitions under the rule that built
them (hold-out filled with the smallest groups first, seeds {sr_seeds}). Their
hold-outs consist of singleton scaffolds or singleton clusters and overlap with
each other and with the primary partition, because the pool of singleton groups
is limited; `results_second_revision/c1b_same_rule_characterisation.csv` gives
the overlap of every replicate.

## Random seeds

| Step | Seed |
| --- | --- |
| the five primary data partitions | 42 |
| inner cross-validation for hyperparameter selection | 42 |
| language-model fine-tuning | 42, 1, 2026 (42 on the repeated partitions) |
| graph network | 42 |
| Extra Trees / CatBoost benchmark learners | 1, 2 |
| matched descriptor-ensemble members | 11, 12, 13, 14 |
| attribution permutation nulls | 0 |
| activity-landscape random pairing | 7 |
| bootstrap of the model summary and paired comparisons, primary stratified partition | 20260908 |
| bootstrap intervals of every model on every primary partition | 42 |
| repeated partitions, pair-aware folds and every bootstrap of the second revision | 20260930 |
| replicates of the primary scaffold- and cluster-disjoint partitions | 1, 2, 3, 4, 5 |

## Notes on two methodological points

**Encoders are fitted inside the training partition.** For every partition the
training compounds are divided into folds and each fold encoder sees only its
own training folds; held-out compounds never contribute a gradient update, an
early-stopping decision or a normalisation statistic. Violating this condition
inflates the apparent held-out accuracy substantially, which is documented in
`results/a9_legacy_audit.json`.

**Fold-averaged embeddings are not used as downstream features.** The embedding
of a *training* molecule produced by averaging fold encoders carries that
molecule's own label. The fine-tuned models therefore enter the benchmark as
end-to-end predictors, whose held-out predictions come only from encoders that
never saw those compounds and whose training predictions are strictly
out-of-fold.

## Row order

`curated_dataset.csv`, every matrix in `representations.npz` and every
`compound_index` in the partition and prediction tables share one zero-based
compound index.
"""


def title():
    text = open(os.path.join(PKG, "manuscript.tex"), encoding="utf-8").read()
    start = text.index("]{", text.index("\\title[")) + 2
    depth, i = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return " ".join(text[start:i - 1].split())


def main():
    if os.path.isdir(REPRO):
        shutil.rmtree(REPRO)
    shutil.copytree(R1_REPRO, REPRO)

    # ------------------------------------------------------------------ data
    data = os.path.join(REPRO, "data")
    parts = read_json(os.path.join(OUT, "c1_partitions.json"))
    rows = []
    for design in DESIGNS:
        for k in REP_FOLDS:
            p = parts["partitions"][f"{design}_f{k:02d}"]
            rows += [{"design": design, "partition": k + 1, "compound_index": i, "role": "train"}
                     for i in p["train_idx"]]
            rows += [{"design": design, "partition": k + 1, "compound_index": i, "role": "test"}
                     for i in p["test_idx"]]
    pd.DataFrame(rows).to_csv(os.path.join(data, "repeated_partitions.csv"), index=False)
    rows = []
    for pid in same_rule_ids():
        p = parts["partitions"][pid]
        rows += [{"design": p["design"], "seed": p["seed"], "compound_index": i, "role": "train"}
                 for i in p["train_idx"]]
        rows += [{"design": p["design"], "seed": p["seed"], "compound_index": i, "role": "test"}
                 for i in p["test_idx"]]
    pd.DataFrame(rows).to_csv(os.path.join(data, "primary_rule_replicates.csv"), index=False)
    shutil.copy(os.path.join(OUT, "c1_partitions.json"), os.path.join(data, "repeated_partitions_all_folds.json"))
    chars = pd.read_csv(os.path.join(OUT, "c1_partition_characterisation.csv"))
    chars[chars["fold"].isin(REP_FOLDS)].to_csv(
        os.path.join(data, "repeated_partition_characterisation.csv"), index=False)
    with open(os.path.join(data, "README.txt"), "a", encoding="utf-8") as fh:
        fh.write(
            "repeated_partitions.csv        compound_index / role for the repeated 9:1 partitions of the\n"
            "                               four resampleable designs (second revision)\n"
            "primary_rule_replicates.csv    compound_index / role for the replicates of the primary scaffold-\n"
            "                               and cluster-disjoint partitions under their own rule\n"
            "repeated_partitions_all_folds.json  assignment of every compound to one of ten parts per design\n"
            "repeated_partition_characterisation.csv  similarity and scaffold-overlap statistics per\n"
            "                               repeated partition\n")

    # --------------------------------------------------------------- results
    res = os.path.join(REPRO, "results_second_revision")
    os.makedirs(res)
    for f in sorted(os.listdir(OUT)):
        full = os.path.join(OUT, f)
        if os.path.isfile(full) and f.endswith((".json", ".csv", ".txt")) and f != "c1_partitions.json":
            shutil.copy(full, res)
    keep = {f"{d}_f{k:02d}.json" for d in DESIGNS for k in REP_FOLDS} | {f"{pid}.json" for pid in same_rule_ids()}
    for folder in ("grid", "finetune", "gnn"):
        os.makedirs(os.path.join(res, folder))
        for f in sorted(os.listdir(os.path.join(OUT, folder))):
            if f in keep:
                shutil.copy(os.path.join(OUT, folder, f), os.path.join(res, folder))

    # ------------------------------------------------------------------ code
    code = os.path.join(REPRO, "code_second_revision")
    os.makedirs(code)
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py") and f not in CORRESPONDENCE_TOOLS:
            shutil.copy(os.path.join(HERE, f), code)

    with open(os.path.join(REPRO, "README.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(README.format(title=title(), n_rep=len(REP_FOLDS),
                               folds=", ".join(str(k + 1) for k in REP_FOLDS),
                               n_sr=len(SAME_RULE_SEEDS), sr_seeds=", ".join(str(x) for x in SAME_RULE_SEEDS)))

    cleaned = strip_local_paths(REPRO)
    print(f"local paths removed from {len(cleaned)} files: {', '.join(cleaned) if cleaned else 'none found'}")

    archive = os.path.join(PKG, "Supplementary_Data.zip")
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for folder, _, files in os.walk(REPRO):
            for f in files:
                full = os.path.join(folder, f)
                z.write(full, os.path.relpath(full, REPRO))
    n = sum(len(fs) for _, _, fs in os.walk(REPRO))
    print(f"wrote {archive} ({os.path.getsize(archive) / 1e6:.1f} MB, {n} files)")
    # The deployable model is not packed here: it lives in the public repository,
    # which e8_github_release.py assembles from this folder and the model files,
    # and the DOI-assigned archive is a snapshot of that repository.


def strip_local_paths(root):
    """Remove the path of the machine that ran the analysis from the deposit.

    A few result files of the first revision record where a checkpoint was read
    from, as an absolute path on the workstation. That path says nothing about
    the analysis and names folders that have no place in a public archive, so
    it is cut back to the part below the project folder
    (hf_local/<checkpoint>). Only the copy in the deposit is touched.
    """
    import json

    prefix = ROOT + os.sep
    variants = {prefix, prefix.replace("\\", "/"),
                json.dumps(prefix)[1:-1], json.dumps(prefix, ensure_ascii=False)[1:-1],
                json.dumps(prefix.replace("\\", "/"))[1:-1]}
    changed = []
    for folder, _, files in os.walk(root):
        for f in files:
            if not f.endswith((".json", ".csv", ".txt", ".md", ".py")):
                continue
            path = os.path.join(folder, f)
            raw = open(path, "rb").read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            new = text
            for v in sorted(variants, key=len, reverse=True):
                new = new.replace(v, "")
            if new != text:
                open(path, "wb").write(new.encode("utf-8"))
                changed.append(os.path.relpath(path, root).replace(os.sep, "/"))
    # nothing of the local folder names may survive, in any spelling
    leftovers = []
    needles = [p for p in ROOT.split(os.sep)[1:-1] if not p.isascii()] + ["Users" + os.sep]
    needles += [json.dumps(p)[1:-1] for p in list(needles)]
    for folder, _, files in os.walk(root):
        for f in files:
            if f.endswith((".json", ".csv", ".txt", ".md", ".py")):
                text = open(os.path.join(folder, f), "rb").read().decode("utf-8", errors="ignore")
                if any(n in text for n in needles):
                    leftovers.append(f)
    if leftovers:
        raise SystemExit("local folder names remain in: " + ", ".join(sorted(set(leftovers))))
    return sorted(changed)


if __name__ == "__main__":
    main()
