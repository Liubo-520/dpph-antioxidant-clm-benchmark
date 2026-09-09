# Model card

| | |
|---|---|
| Task | regression of pIC50 = -log10(IC50 / M) |
| Endpoint | DPPH radical scavenging, nominal 30-minute readout |
| Input | SMILES |
| Representation | ECFP4+Mordred (3475 features) |
| Learner | ExtraTrees |
| Selected hyperparameters | mdl__max_features = 0.3 |
| Training compounds | 1719 |
| Held-out compounds | 192 |
| Partition | chemistry-stratified 9:1, fixed seed 42 |

## Measured performance

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Held-out R2 | 0.8137 | 0.714--0.875 |
| Held-out RMSE | 0.3490 | 0.299--0.400 |
| Held-out MAE | 0.2502 | 0.217--0.286 |
| Out-of-fold Q2 on the training set | 0.7706 | |

Intervals are percentile bootstrap over 2,000 resamples of the hold-out. The Q2
is the pooled 10-fold out-of-fold R2 of the estimator the grid search selected;
because the hyperparameters were chosen using all of the training data it is a
slightly optimistic summary of training-set performance, not a nested
cross-validation estimate.

## Limits

- This is the best *individual* model of the benchmark, not the top line of the
  paper's main table; that line is Stacked model without language-model members at R2 = 0.8140, 0.0003 above this
  one and well inside both confidence intervals.
- Hyperparameters were selected inside the training partition only, and the
  held-out compounds were scored exactly once.
- On a source-publication-disjoint partition the model scores R2 = -0.033. Treat
  predictions for compounds unlike the training set as ranking hints.
- The training labels are aggregated from many publications and differ in
  solvent, temperature, illumination and instrument. That heterogeneity places a
  floor on attainable accuracy.
- The model is not a mechanistic model. It does not explain why a compound
  scavenges radicals.
