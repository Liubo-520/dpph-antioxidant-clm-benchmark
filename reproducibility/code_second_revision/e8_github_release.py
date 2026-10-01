"""Assemble the public repository that the DOI-assigned archive is a snapshot of.

After the first revision the repository held the deployable model and nothing
else. The editorial office then asked for the code to be deposited in a
DOI-assigning repository. Zenodo archives a GitHub release as it stands, so
the repository now holds the whole deposit: the reproducibility folder built by
e7_package_data.py beside the model files.

This script writes into revision/二审/release, which must be a clone of the
repository. It replaces README.md, the model card and the metrics file, whose
first-revision wording described a model selection that the second revision
changed, copies the reproducibility folder in, and writes CITATION.cff. It does
not commit and does not push.

    python e8_github_release.py
"""

import importlib.util
import json
import os
import shutil

import pandas as pd

from common import OUT, PRIMARY, R1_OUT, R2, read_json

PKG = os.path.join(R2, "Revision_R2")
REPRO = os.path.join(PKG, "reproducibility")
RELEASE = os.path.join(R2, "release")
CELL = ("ECFP4+Mordred", "ExtraTrees")
REPO_URL = "https://github.com/Liubo-520/dpph-antioxidant-clm-benchmark"
AUTHORS = [("Chi", "Chao"), ("Yang", "Chunmin"), ("Cui", "Dongyang")]
AFFILIATION = ("School of Food and Pharmaceutical Science and Technology, "
               "Guangzhou College of Technology and Business, Guangzhou 510850, China")
PARTITION_NAME = {"scaffold": "scaffold-disjoint", "cluster": "cluster-disjoint",
                  "dissimilarity": "maximum-dissimilarity", "source": "source-publication-disjoint"}


def paper_title():
    text = open(os.path.join(PKG, "paper_title.tex"), encoding="utf-8").read()
    start = text.index("{", text.index("\\newcommand{\\PaperTitle}") + len("\\newcommand{\\PaperTitle}")) + 1
    return text[start:text.index("}", start)]


def facts():
    old = read_json(os.path.join(RELEASE, "model", "metrics.json"))
    out = {k: old[k] for k in ("n_train", "n_test", "q2", "r2", "rmse", "mae", "ci", "best_params",
                               "model_file", "model_mb", "n_features")}
    # the released representation and learner on every primary partition
    out["same_cell_r2_by_partition"] = {}
    for split in PRIMARY:
        m = pd.read_csv(os.path.join(R1_OUT, f"a5_matrix_{split}.csv"))
        row = m[(m["representation"] == CELL[0]) & (m["learner"] == CELL[1])]
        out["same_cell_r2_by_partition"][split] = float(row["test_r2"].iloc[0])
    assert abs(out["same_cell_r2_by_partition"]["stratified"] - out["r2"]) < 1e-9
    # the model the paper reports, chosen by training-set Q2
    sa = read_json(os.path.join(OUT, "a7_stats_ad.json"))
    summary = pd.read_csv(os.path.join(OUT, "a7_model_summary.csv"))
    rep = summary[summary["model"] == sa["best_model"]].iloc[0]
    out["reported_model_name"] = sa["best_model"]
    out["reported_model_q2"] = float(rep["q2_cv_oof"])
    out["reported_model_r2"] = float(rep["r2"])
    pairs = pd.read_csv(os.path.join(OUT, "a7_pairwise_tests.csv"))
    row = pairs[pairs["model_b"] == " | ".join(CELL)].iloc[0]
    ci = row["delta_r2_ci95"]
    ci = json.loads(ci) if isinstance(ci, str) else list(ci)
    out["reported_minus_released_r2"] = float(row["delta_r2"])
    out["reported_minus_released_ci95"] = [float(ci[0]), float(ci[1])]
    return out


def similarity_bins():
    spec = importlib.util.spec_from_file_location("predict", os.path.join(RELEASE, "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SIMILARITY_BINS


README = """# Chemical language models for DPPH antioxidant activity: data, code and deployable model

Data, analysis code and the released model of the article

> *{title}*
> ({authors}).

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
descriptors, fitted on {n_train} compounds. In the benchmark it is the
representative of the fingerprint-descriptor hybrid family: the model with the
highest cross-validated Q2 on the training compounds in that family. It
contains no chemical language model.

It is not the model reported in the main table of the article. That one is a
stacked predictor over the representation families, chosen because it has the
highest cross-validated Q2 of all models ({rep_q2}). On the hold-out it reaches
R2 = {rep_r2} against {r2} here, a difference of {delta} (95% CI {dlo} to
{dhi}) that {n_test} held-out compounds cannot resolve. What is released is the
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

On the {n_test}-compound chemistry-stratified hold-out: R2 = {r2}
(95% CI {r2lo}--{r2hi}), RMSE = {rmse}, MAE = {mae}.

That number describes compounds resembling the training set. The same
representation and learner, refitted on the training side of the harder
partitions of the article, score R2 = {scaf} (scaffold-disjoint), {clus}
(cluster-disjoint) and {diss} (maximum-dissimilarity), and {src} on a partition
that separates whole source publications, which is worse than predicting the
training mean. Held-out RMSE by nearest-neighbour similarity:

| NN Tanimoto to training | Held-out compounds | RMSE |
|---|---|---|
{bins}

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
"""

CARD = """# Model card

| | |
|---|---|
| Task | regression of pIC50 = -log10(IC50 / M) |
| Endpoint | DPPH radical scavenging, nominal 30-minute readout |
| Input | SMILES |
| Representation | {rep} ({n_features} features) |
| Learner | {learner} |
| Selected hyperparameters | {params} |
| Training compounds | {n_train} |
| Held-out compounds | {n_test} |
| Partition | chemistry-stratified 9:1, fixed seed 42 |

## Measured performance

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Held-out R2 | {r2} | {r2lo}--{r2hi} |
| Held-out RMSE | {rmse} | {rmselo}--{rmsehi} |
| Held-out MAE | {mae} | {maelo}--{maehi} |
| Out-of-fold Q2 on the training set | {q2} | |

Intervals are percentile bootstrap over 2,000 resamples of the hold-out. The Q2
is the pooled 10-fold out-of-fold R2 of the estimator the grid search selected;
because the hyperparameters were chosen using all of the training data it is a
slightly optimistic summary of training-set performance, not a nested
cross-validation estimate.

## Limits

- This is the representative of the fingerprint-descriptor hybrid family, the
  model with the highest training-set Q2 in that family. It is not the model
  reported in the main table of the article; that one is the
  {rep_name_lower}, chosen by the highest Q2 of all models, at
  R2 = {rep_r2}. The difference from this model is {delta} (95% CI {dlo} to {dhi}).
- Hyperparameters were selected inside the training partition only, and the
  held-out compounds were scored exactly once.
- Refitted on the training side of a source-publication-disjoint partition, the
  same representation and learner score R2 = {src}. Treat predictions for
  compounds unlike the training set as ranking hints.
- The training labels are aggregated from many publications and differ in
  solvent, temperature, illumination and instrument. That heterogeneity places a
  floor on attainable accuracy.
- The model is not a mechanistic model. It does not explain why a compound
  scavenges radicals.
- The model has not been validated experimentally.
"""


def main():
    if not os.path.isdir(os.path.join(RELEASE, ".git")):
        raise SystemExit(f"{RELEASE} is not a clone of the repository; clone {REPO_URL} there first")
    f = facts()
    title = paper_title()
    bins = "\n".join(f"| [{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'} | {n} | {rmse:.3f} |"
                     for lo, hi, rmse, n in similarity_bins())
    by = f["same_cell_r2_by_partition"]
    common = dict(
        title=title, authors=", ".join(f"{g} {s}" for s, g in AUTHORS), n_train=f["n_train"], n_test=f["n_test"],
        r2=f"{f['r2']:.4f}", r2lo=f"{f['ci']['r2'][0]:.3f}", r2hi=f"{f['ci']['r2'][1]:.3f}",
        rmse=f"{f['rmse']:.4f}", rmselo=f"{f['ci']['rmse'][0]:.3f}", rmsehi=f"{f['ci']['rmse'][1]:.3f}",
        mae=f"{f['mae']:.4f}", maelo=f"{f['ci']['mae'][0]:.3f}", maehi=f"{f['ci']['mae'][1]:.3f}",
        q2=f"{f['q2']:.4f}", rep_q2=f"{f['reported_model_q2']:.4f}", rep_r2=f"{f['reported_model_r2']:.4f}",
        delta=f"{f['reported_minus_released_r2']:+.3f}", dlo=f"{f['reported_minus_released_ci95'][0]:+.3f}",
        dhi=f"{f['reported_minus_released_ci95'][1]:+.3f}",
        scaf=f"{by['scaffold']:.3f}", clus=f"{by['cluster']:.3f}", diss=f"{by['dissimilarity']:.3f}",
        src=f"{by['source']:.3f}", bins=bins,
        rep=CELL[0], learner=CELL[1], n_features=f["n_features"],
        params=", ".join(f"{k} = {v}" for k, v in f["best_params"].items()),
        rep_name_lower=f["reported_model_name"][0].lower() + f["reported_model_name"][1:],
    )

    def write(rel, text):
        with open(os.path.join(RELEASE, rel.replace("/", os.sep)), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

    write("README.md", README.format(**common))
    write("model/model_card.md", CARD.format(**common))
    write("model/metrics.json", json.dumps(f, indent=2) + "\n")

    cff = ["cff-version: 1.2.0",
           'message: "If you use this repository, please cite the article it accompanies."',
           "type: dataset",
           f'title: "Data, code and deployable model for: {title}"',
           "authors:"]
    for family, given in AUTHORS:
        cff += [f"  - family-names: {family}", f"    given-names: {given}", f'    affiliation: "{AFFILIATION}"']
    cff += [f'repository-code: "{REPO_URL}"',
            "keywords:", "  - antioxidant activity", "  - DPPH", "  - chemical language model", "  - QSAR",
            "  - benchmark", ""]
    write("CITATION.cff", "\n".join(cff))

    target = os.path.join(RELEASE, "reproducibility")
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.copytree(REPRO, target)

    n = sum(len(fs) for folder, _, fs in os.walk(RELEASE) if ".git" not in folder.split(os.sep))
    size = sum(os.path.getsize(os.path.join(folder, x)) for folder, _, fs in os.walk(RELEASE)
               if ".git" not in folder.split(os.sep) for x in fs)
    big = max(((os.path.getsize(os.path.join(folder, x)), os.path.join(folder, x))
               for folder, _, fs in os.walk(RELEASE) if ".git" not in folder.split(os.sep) for x in fs))
    print(f"assembled {RELEASE}: {n} files, {size / 1e6:.0f} MB; largest file {big[0] / 1e6:.1f} MB "
          f"({os.path.relpath(big[1], RELEASE)})")
    if big[0] > 95e6:
        raise SystemExit("a file is too large for GitHub (100 MB limit)")
    print(f"released model: R2 {common['r2']}; reported model {f['reported_model_name']}: "
          f"Q2 {common['rep_q2']}, R2 {common['rep_r2']}, difference {common['delta']} "
          f"[{common['dlo']}, {common['dhi']}]")
    print("same representation and learner on the harder partitions:",
          {PARTITION_NAME[k]: round(v, 3) for k, v in by.items() if k != "stratified"})


if __name__ == "__main__":
    main()
