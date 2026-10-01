# Chemical language models for DPPH antioxidant activity: data, code and deployable model

Data, analysis code and the released model of the article

> *Leakage-controlled multi-partition benchmarking of chemical language models for antioxidant activity prediction with quantitative molecular attribution*
> (Chao Chi, Chunmin Yang, Dongyang Cui).

## What is here

| Path | Content |
|---|---|
| `reproducibility/` | the curated dataset with provenance, every train/test partition (the five primary ones and the repeated ones), every representation matrix, the predictions of every model, the complete analysis code and the environment record. Start with `reproducibility/README.md` |
| `model/`, `predict.py`, `requirements.txt` | a deployable model that predicts DPPH pIC50 from a SMILES string and reports how similar each query is to its training set |

An archived snapshot of this repository carries the DOI given in the Code
availability statement of the article.

## The deployable model

### What it is, and what it is not

An Extra Trees regressor on ECFP4 fingerprints concatenated with Mordred
descriptors, fitted on 1719 compounds. In the benchmark it is the
representative of the fingerprint-descriptor hybrid family: the model with the
highest cross-validated Q2 on the training compounds in that family. It
contains no chemical language model.

It is not the model reported in the main table of the article. That one is a
stacked predictor over the representation families, chosen because it has the
highest cross-validated Q2 of all models (0.8091). On the hold-out it reaches
R2 = 0.8013 against 0.8137 here, a difference of -0.012 (95% CI -0.053 to
+0.019) that 192 held-out compounds cannot resolve. What is released is the
model that can be deployed as it stands -- one estimator, one feature pipeline,
reproducible from `reproducibility/`.

### Install and run

```
pip install -r requirements.txt
python predict.py "Oc1ccc(cc1O)/C=C/C(=O)O"
python predict.py --input molecules.smi --output predictions.csv
```

Output columns: the predicted pIC50, the nearest-neighbour ECFP4 Tanimoto
similarity of the query to the training set, and the held-out RMSE that was
measured for compounds in that similarity range.

### How well it works, and where it does not

On the 192-compound chemistry-stratified hold-out: R2 = 0.8137
(95% CI 0.714--0.875), RMSE = 0.3490, MAE = 0.2502.

That number describes compounds resembling the training set. The same
representation and learner, refitted on the training side of the harder
partitions of the article, score R2 = 0.577 (scaffold-disjoint), 0.385
(cluster-disjoint) and 0.251 (maximum-dissimilarity), and -0.183 on a partition
that separates whole source publications, which is worse than predicting the
training mean. Held-out RMSE by nearest-neighbour similarity:

| NN Tanimoto to training | Held-out compounds | RMSE |
|---|---|---|
| [0.0, 0.4) | 6 | 0.564 |
| [0.4, 0.6) | 16 | 0.543 |
| [0.6, 0.8) | 83 | 0.313 |
| [0.8, 1.0] | 87 | 0.315 |

`predict.py` prints the bin each query falls into. Use it. A prediction for a
compound with no training neighbour above Tanimoto 0.6 is a ranking hint, not an
estimate.

The endpoint is DPPH radical scavenging at a nominal 30-minute readout, pooled
from many publications. Nothing here transfers to ABTS, FRAP or ORAC. The model
has not been tested prospectively: no compound was synthesised or assayed for
the study.

## Citation

Please cite the article. The model card in `model/model_card.md` records the
training partition, the selected hyperparameters and the measured metrics.
