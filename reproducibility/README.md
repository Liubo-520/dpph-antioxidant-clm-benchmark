# Reproducibility deposit

Supporting data and code for the manuscript *Leakage-controlled multi-partition benchmarking of chemical language models for antioxidant activity prediction with quantitative molecular attribution*.

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
the 5 hold-outs used: the compounds were divided into ten parts of equal
size, whole groups to one part in the three group-disjoint designs, and parts
1, 2, 3, 4, 5 served in turn as the hold-out, so every partition is 9:1 and the
hold-outs of a design do not overlap. `data/repeated_partitions_all_folds.json`
holds the assignment of every compound to one of the ten parts.

`data/primary_rule_replicates.csv` lists a second set: 5 replicates of each of
the primary scaffold- and cluster-disjoint partitions under the rule that built
them (hold-out filled with the smallest groups first, seeds 1, 2, 3, 4, 5). Their
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
