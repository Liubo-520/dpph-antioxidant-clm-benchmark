# DPPH antioxidant activity predictor

Predicts pIC50 for DPPH radical scavenging from a SMILES string.

This is the model released with *Chemical Language Models for Antioxidant
Activity Prediction: A Leakage-Controlled Multi-Partition Benchmark with
Quantitative Molecular Attribution*. The repository holds the fitted model and
the instructions for running it, and nothing else; the curated dataset, the five
partitions, every representation matrix, all predictions and the complete
analysis code are in the supplementary archive that accompanies the paper.

## What it is, and what it is not

An Extra Trees regressor on ECFP4 fingerprints concatenated with Mordred
descriptors, fitted on 1719 compounds. It is the most accurate **individual**
model of the 55 scored in the benchmark, and it contains no chemical language
model -- that is one of the findings of the paper rather than an implementation
detail.

It is not the single top line of the paper's main table. That line is a stacked
predictor with its language-model members removed, at R2 = 0.8140 against
0.8137 here: a difference of 0.0003, far inside either model's confidence interval,
and the paper says as much. What is released is the best model you can actually
deploy -- one estimator, one feature pipeline, reproducible from the source
archive -- rather than an ablation construct built to answer a question about
whether the language models contribute anything.

## Install and run

```
pip install -r requirements.txt
python predict.py "Oc1ccc(cc1O)/C=C/C(=O)O"
python predict.py --input molecules.smi --output predictions.csv
```

Output columns: the predicted pIC50, the nearest-neighbour ECFP4 Tanimoto
similarity of the query to the training set, and the held-out RMSE that was
measured for compounds in that similarity range.

## How well it works, and where it does not

On the 192-compound chemistry-stratified hold-out: R2 = 0.8137
(95% CI 0.714--0.875), RMSE = 0.3490, MAE = 0.2502.

That number describes compounds resembling the training set. Accuracy falls as
the hold-out is made structurally harder, and on a partition that separates
whole source publications the model scores a **negative** R2 of -0.033: worse than
predicting the training mean. Held-out RMSE by nearest-neighbour similarity:

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
from many publications. Nothing here transfers to ABTS, FRAP or ORAC.

## Citation

See the paper. The model card in `model/model_card.md` records the training
partition, the selected hyperparameters and the full set of measured metrics.
