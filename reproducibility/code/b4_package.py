"""Assemble the reproducibility deposit that accompanies the revision.

Everything a reader needs to reproduce the reported numbers, and nothing that
would make the archive unnecessarily large: the curated dataset with its
provenance, the index files of all five partitions, every representation
matrix, every model prediction, the analysis code and the environment record.
"""

import json
import os
import shutil
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")
PKG = os.path.join(ROOT, "revision", "Revision_R1_new")
REPRO = os.path.join(PKG, "reproducibility")

SPLITS = ["stratified", "scaffold", "cluster", "dissimilarity", "source"]


def fresh(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def main():
    data = fresh(os.path.join(REPRO, "data"))
    results = fresh(os.path.join(REPRO, "results"))
    code = fresh(os.path.join(REPRO, "code"))

    # ------------------------------------------------------------------ data
    shutil.copy(os.path.join(OUT, "curated_dataset.csv"), data)

    splits = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))
    rows = []
    for name in SPLITS:
        s = splits["splits"].get(name)
        if not s:
            continue
        for i in s["train_idx"]:
            rows.append({"split": name, "compound_index": i, "partition": "train"})
        for i in s["test_idx"]:
            rows.append({"split": name, "compound_index": i, "partition": "test"})
    pd.DataFrame(rows).to_csv(os.path.join(data, "partitions.csv"), index=False)

    rep_rows = []
    for name in SPLITS:
        for c in splits["characterisation"]:
            if c["split"] == name:
                rep_rows.append({k: v for k, v in c.items() if not isinstance(v, (list, dict))})
    pd.DataFrame(rep_rows).to_csv(os.path.join(data, "partition_characterisation.csv"), index=False)

    if os.path.exists(os.path.join(FEAT, "representations.npz")):
        shutil.copy(os.path.join(FEAT, "representations.npz"), data)

    with open(os.path.join(data, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "curated_dataset.csv            one row per compound: ChEMBL identifier, source DOI,\n"
            "                               assay description, canonical SMILES, InChIKey,\n"
            "                               IC50, pIC50, Bemis-Murcko scaffold, category\n"
            "partitions.csv                 compound_index / partition membership for all five splits\n"
            "partition_characterisation.csv similarity and scaffold-overlap statistics per split\n"
            "representations.npz            every molecular representation matrix, keyed by name,\n"
            "                               row order identical to curated_dataset.csv\n"
        )

    # --------------------------------------------------------------- results
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".json", ".csv")) and not f.startswith("curated_"):
            shutil.copy(os.path.join(OUT, f), results)

    # predictions in a flat, tool-agnostic form
    for name in SPLITS:
        p = os.path.join(OUT, f"a5_matrix_{name}_predictions.json")
        if not os.path.exists(p):
            continue
        payload = json.load(open(p, encoding="utf-8"))
        frames = []
        for model, pred in payload["predictions"].items():
            frames.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "compound_index": payload["test_idx"],
                        "partition": "test",
                        "experimental_pIC50": payload["y_test"],
                        "predicted_pIC50": pred["test"],
                    }
                )
            )
            frames.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "compound_index": payload["train_idx"],
                        "partition": "train_out_of_fold",
                        "experimental_pIC50": payload["y_train"],
                        "predicted_pIC50": pred["oof"],
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(results, f"predictions_{name}.csv"), index=False
        )

    # ------------------------------------------------------------------ code
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py") and not f.startswith("tmp"):
            shutil.copy(os.path.join(HERE, f), code)

    env = json.load(open(os.path.join(OUT, "b0_environment.json"), encoding="utf-8"))
    with open(os.path.join(REPRO, "environment.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Python {env['python']} on {env['platform']} ({env['machine']})\n")
        fh.write(f"CPU threads {env['cpu_count']}\n")
        if env.get("gpu"):
            fh.write(f"GPU {env['gpu']}\n")
        if env.get("cuda"):
            fh.write(f"CUDA {env['cuda']}\n\n")
        for k, v in env["packages"].items():
            if v:
                fh.write(f"{k}=={v}\n")

    with open(os.path.join(REPRO, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(REPRO_README)

    # -------------------------------------------------------------- zip it up
    archive = os.path.join(PKG, "Supplementary_Data.zip")
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for folder, _, files in os.walk(REPRO):
            for f in files:
                full = os.path.join(folder, f)
                z.write(full, os.path.relpath(full, REPRO))
    print(f"wrote {archive} ({os.path.getsize(archive) / 1e6:.1f} MB)")


REPRO_README = """# Reproducibility deposit

Supporting data and code for *Chemical Language Models for Antioxidant Activity
Prediction: A Leakage-Controlled Multi-Partition Benchmark with Quantitative
Molecular Attribution*.

## Contents

```
data/       curated dataset with provenance, the five train/test partitions,
            partition characterisation, and every representation matrix
results/    all analysis outputs (JSON and CSV) and flat prediction tables
code/       the complete analysis pipeline
environment.txt   Python, package and hardware versions used for the reported run
```

## Pipeline

The scripts are numbered in execution order. Each writes to `results/` and
reads only from earlier outputs, so the pipeline can be resumed at any step.

| Script | Purpose |
| --- | --- |
| `a0_fetch_checkpoints.py` | download the four pretrained encoders and convert them to `safetensors` |
| `a1_dataset_audit.py` | provenance, assay-text, duplicate and structure-quality audit |
| `a2_representations.py` | build every molecular representation and measure extraction cost |
| `a2b_pooling.py` | sensitivity of the frozen encoder representation to layer and pooling |
| `a3_finetune_clean.py` | fold-nested supervised fine-tuning inside the training partition only |
| `a4_splits.py` | construct and characterise the five partitions |
| `a5_matrix.py` | representation x learner grid search and the matched descriptor ensemble |
| `a5b_gnn.py` | graph neural network baseline |
| `a5c_merge.py` | assemble the per-partition benchmark table |
| `a6_attribution.py` | four-method attribution with permutation nulls and a faithfulness test |
| `a7_stats_ad.py` | bootstrap intervals, paired tests, applicability domain, conformal intervals |
| `a8_chemspace_utility.py` | UMAP characterisation, matched pairs, triage simulation |
| `a9_legacy_audit.py` | audit of the archived feature matrix of the original submission |
| `b0_environment.py` | record the software environment |
| `b1_make_macros.py` | emit every reported number as a LaTeX macro or table body |
| `b2_make_figures.py` | draw the manuscript figures from the analysis outputs |
| `b2b_response_figures.py` | draw the figures used in the response letter |
| `b3_attr_render.py` | render the representative atom-level attribution maps |
| `b4_package.py` | assemble this deposit |
| `b5_verify_claims.py` | re-derive every directional statement in the text from the outputs |
| `b6_crossrefs.py` | emit the manuscript's section, table and figure numbers as macros for the other documents |
| `b7_response_docx.py` | render the response letter as an editable Word document from the same macros |

The `run_*.py` files are drivers that sequence the expensive steps:
`run_finetune_all.py` (the fine-tuning schedule), `run_matrix_all.py` (the
benchmark passes), `run_gpu_rest.py` (the accelerator jobs that follow
fine-tuning), `run_finish.py` (every downstream step once fitting is done)
and `run_supervisor.py`, which restarts any driver that stops early.

## Random seeds

| Step | Seed |
| --- | --- |
| all five data partitions | 42 |
| inner cross-validation for hyperparameter selection | 42 |
| language-model fine-tuning | 42, 1, 2026 |
| Extra Trees / CatBoost benchmark learners | 1, 2 |
| matched descriptor-ensemble members | 11, 12, 13, 14 |
| attribution permutation nulls | 0 |
| activity-landscape random pairing | 7 |
| all bootstrap resampling | 20260908 |

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
`compound_index` in `partitions.csv` and the prediction tables share one
zero-based compound index.
"""


if __name__ == "__main__":
    main()
