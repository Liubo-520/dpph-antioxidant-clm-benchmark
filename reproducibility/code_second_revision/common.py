"""Paths and helpers shared by the second-round (R2) analysis scripts.

The R2 analyses reuse, unchanged, the curated dataset, the label-free
representation matrices, the five primary partitions and every model output of
the first revision (revision/一审/analysis). Nothing in that tree is modified;
all new outputs go to revision/二审/analysis/outputs.
"""

import json
import os
import pickle
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)                      # revision/二审/analysis
R2 = os.path.dirname(ANALYSIS)                        # revision/二审
REVISION = os.path.dirname(R2)                        # revision
ROOT = os.path.dirname(REVISION)                      # project root

R1_ANALYSIS = os.path.join(REVISION, "一审", "analysis")
R1_OUT = os.path.join(R1_ANALYSIS, "outputs")
R1_FEAT = os.path.join(R1_ANALYSIS, "features")
R1_SCRIPTS = os.path.join(R1_ANALYSIS, "scripts")

OUT = os.path.join(ANALYSIS, "outputs")
LOGS = os.path.join(ANALYSIS, "logs")
for _d in (OUT, LOGS):
    os.makedirs(_d, exist_ok=True)

# The four designs that can be repeated. The maximum-dissimilarity partition is
# deterministic (the 10% most isolated compounds), so it has no assignment
# variability to resample and is not part of the repeated analysis.
DESIGNS = ("stratified", "scaffold", "cluster", "source")
PRIMARY = ("stratified", "scaffold", "cluster", "dissimilarity", "source")
N_FOLDS = 10
# The compounds are assigned to ten folds so that every hold-out is a 9:1
# partition, like the primary ones. Five of the ten, fixed in advance as the
# first five, are used as repeated hold-outs: five 9:1 partitions per design
# whose hold-outs do not overlap.
REP_FOLDS = (0, 1, 2, 3, 4)
PARTITION_SEED = 20260930
# Replicates of the primary scaffold- and cluster-disjoint partitions under the
# rule that built them (smallest groups first, c1b_partitions_same_rule.py). The
# source-disjoint design is not replicated this way: under that rule nine tenths
# of its hold-out is the same in every draw.
SAME_RULE_DESIGNS = ("scaffold", "cluster")
SAME_RULE_SEEDS = (1, 2, 3, 4, 5)

CLM_TOKENS = ("ChemBERTa", "MoLFormer")

# family key -> label printed in tables, in the row order of manuscript Table 3
FAMILIES = [
    ("Fp", "ECFP4"),
    ("Desc", "Mordred / RDKit"),
    ("Frozen", "Frozen CLM"),
    ("Ft", "Fine-tuned CLM"),
    ("Hyb", "Hybrid, ECFP4 + Mordred"),
    ("HybClm", "Hybrid with a CLM"),
    ("Gnn", "Graph network"),
]
FAMILY_LABEL = dict(FAMILIES)


def family_key(family, representation):
    """Map a benchmark row to one of the seven family keys of Table 3."""
    family, representation = str(family), str(representation)
    if family == "Fingerprint baseline":
        return "Fp"
    if family == "Descriptor baseline":
        return "Desc"
    if family == "Frozen chemical language model":
        return "Frozen"
    if family == "Fine-tuned chemical language model":
        return "Ft"
    if family == "Graph neural network":
        return "Gnn"
    if family == "Hybrid":
        return "HybClm" if any(t in representation for t in CLM_TOKENS) else "Hyb"
    return None


def load_dataset():
    """SMILES, pIC50 and the curated annotation table, in dataset order."""
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        data = pickle.load(fh)
    curated = pd.read_csv(os.path.join(R1_OUT, "curated_dataset.csv"))
    smiles = data["df"]["smiles"].tolist()
    y = data["df"]["pIC50"].to_numpy(float)
    assert len(curated) == len(smiles)
    return smiles, y, curated, data


def load_light():
    """SMILES, pIC50 and the curated table without unpickling the graph objects.

    The curated table carries the same SMILES strings and activity values, in
    the same order, as processed_data.pkl (checked: identical element for
    element). CPU-only workers use this so that they do not have to import
    torch, which on Windows commits several gigabytes per process.
    """
    curated = pd.read_csv(os.path.join(R1_OUT, "curated_dataset.csv"))
    return curated["canonical_smiles"].tolist(), curated["pIC50"].to_numpy(float), curated, None


def load_partitions():
    return json.load(open(os.path.join(OUT, "c1_partitions.json"), encoding="utf-8"))


def partition_ids(designs=DESIGNS, folds=range(N_FOLDS)):
    return [f"{d}_f{k:02d}" for d in designs for k in folds]


def same_rule_ids(designs=SAME_RULE_DESIGNS, seeds=SAME_RULE_SEEDS):
    return [f"{d}_s{s:02d}" for d in designs for s in seeds]


def unbreakable_minus(value):
    """Keep a negative number on one line.

    TeX may break a line after any explicit hyphen, so a value written as
    "[-0.080, 0.157]" can be typeset with "[-" at the end of one line and
    "0.080" at the start of the next, which reads as a positive number. The
    sign is therefore boxed. Only a hyphen that opens a number is touched: one
    inside a word, a range ("0.03--0.08") or a model name is left alone, and
    applying the function twice changes nothing.
    """
    import re

    return re.sub(r"(?<![\w.}\-])-(?=\d)", lambda m: "\\mbox{-}", value)


def read_json(path):
    return json.load(open(path, encoding="utf-8"))


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def pin(kind):
    """Keep long-running workers off each other's cores.

    On this hybrid CPU Windows schedules windowless background processes onto
    the eight efficiency cores only and leaves the performance cores idle, which
    makes a five-thread fit run at the speed of one. CPU-bound workers are
    therefore pinned to the performance-core threads (0-11) and GPU-bound
    workers, which need one core each to feed the GPU, to the efficiency cores
    (12-19). This changes how fast a result is produced, never what it is.
    """
    try:
        import psutil

        p = psutil.Process()
        n = psutil.cpu_count()
        # PIN_CORES=e sends a CPU-bound helper to the efficiency cores once the
        # GPU-bound jobs that were using them have finished
        if os.environ.get("PIN_CORES") == "e":
            kind = "gpu"
        if n == 20:
            p.cpu_affinity(list(range(12)) if kind == "cpu" else list(range(12, 20)))
    except Exception:
        pass


def r1_import():
    """Make the first-revision analysis modules importable (learner grids etc.)."""
    if R1_SCRIPTS not in sys.path:
        sys.path.insert(0, R1_SCRIPTS)
    code = os.path.join(ROOT, "code")
    if code not in sys.path:
        sys.path.insert(0, code)
