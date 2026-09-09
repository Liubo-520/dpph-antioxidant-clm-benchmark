"""Predict DPPH pIC50 from SMILES.

    python predict.py "Oc1ccc(cc1O)C=CC(=O)O" "CC(=O)Oc1ccccc1C(=O)O"
    python predict.py --input molecules.smi --output predictions.csv

Every prediction is returned with the nearest-neighbour Tanimoto similarity of
the query to the training set and the held-out RMSE that was measured for
compounds in that similarity range. Read them together: the model was accurate
on compounds close to its training set and not accurate on compounds far from
it, and it cannot tell you which case you are in unless you look.
"""

import argparse
import csv
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model", "dpph_pic50_ecfp4_mordred_extratrees.joblib")

# Held-out RMSE by nearest-neighbour ECFP4 Tanimoto to the training set,
# measured on the 192-compound chemistry-stratified hold-out.
SIMILARITY_BINS = [
    (0.0, 0.4, 0.564, 6),
    (0.4, 0.6, 0.543, 16),
    (0.6, 0.8, 0.313, 83),
    (0.8, 1.01, 0.315, 87),
]


def featurise(smiles, bundle):
    """ECFP4 followed by Mordred, in the order and the widths the model was fitted on."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from mordred import Calculator, descriptors

    RDLogger.DisableLog("rdApp.*")
    mols, bad = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad.append(i)
        mols.append(mol)
    if bad:
        raise SystemExit("could not parse SMILES at position(s): %s"
                         % ", ".join(str(i + 1) for i in bad))

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=bundle["ecfp4"]["radius"], fpSize=bundle["ecfp4"]["n_bits"])
    fp = np.asarray([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)

    calc = Calculator(descriptors, ignore_3D=True)
    frame = calc.pandas(mols, quiet=True)
    frame.columns = [str(c) for c in frame.columns]
    missing = [n for n in bundle["mordred_names"] if n not in frame.columns]
    if missing:
        raise SystemExit("this mordred version does not provide %d of the descriptors the "
                         "model expects, starting with %s. Install the pinned version in "
                         "requirements.txt." % (len(missing), missing[0]))
    desc = frame[bundle["mordred_names"]].apply(
        lambda c: c.map(lambda v: v if isinstance(v, (int, float)) else np.nan))
    return np.hstack([fp, desc.to_numpy(dtype=float)]), fp


def nearest_neighbour(fp, bundle):
    """Maximum Tanimoto similarity of each query to the training set."""
    train = np.unpackbits(bundle["train_ecfp4"], axis=1)[:, :bundle["train_ecfp4_shape"][1]]
    train = train.astype(bool)
    query = fp.astype(bool)
    out = np.empty(len(query))
    train_sum = train.sum(1)
    for i, q in enumerate(query):
        inter = (train & q).sum(1)
        union = train_sum + q.sum() - inter
        out[i] = (inter / np.maximum(union, 1)).max()
    return out


def bin_for(sim):
    for lo, hi, rmse, n in SIMILARITY_BINS:
        if lo <= sim < hi:
            return "[%.1f, %.1f)" % (lo, min(hi, 1.0)), rmse, n
    return "?", float("nan"), 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("smiles", nargs="*", help="SMILES strings")
    ap.add_argument("--input", help="file with one SMILES per line")
    ap.add_argument("--output", help="write CSV here instead of to the screen")
    a = ap.parse_args()

    queries = list(a.smiles)
    if a.input:
        with open(a.input, encoding="utf-8") as fh:
            queries += [ln.split()[0] for ln in fh if ln.strip() and not ln.startswith("#")]
    if not queries:
        ap.error("give at least one SMILES, or --input")

    if not os.path.exists(MODEL):
        raise SystemExit("model file not found at %s" % MODEL)
    bundle = joblib.load(MODEL)

    X, fp = featurise(queries, bundle)
    pred = bundle["model"].predict(X)
    sim = nearest_neighbour(fp, bundle)

    rows = []
    for s, p, v in zip(queries, pred, sim):
        label, rmse, n = bin_for(v)
        rows.append({
            "smiles": s,
            "predicted_pIC50": round(float(p), 3),
            "nn_tanimoto_to_training": round(float(v), 3),
            "similarity_bin": label,
            "measured_holdout_rmse_for_bin": rmse,
            "n_holdout_compounds_in_bin": n,
        })

    if a.output:
        with open(a.output, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s (%d rows)" % (a.output, len(rows)))
        return

    width = max(len(r["smiles"]) for r in rows)
    print("%-*s  %8s  %6s  %-12s  %s" % (width, "SMILES", "pIC50", "NN sim", "bin",
                                         "RMSE for that bin"))
    for r in rows:
        print("%-*s  %8.3f  %6.3f  %-12s  %.3f"
              % (width, r["smiles"], r["predicted_pIC50"], r["nn_tanimoto_to_training"],
                 r["similarity_bin"], r["measured_holdout_rmse_for_bin"]))
    if any(r["nn_tanimoto_to_training"] < 0.6 for r in rows):
        print("\nAt least one query has no training neighbour above Tanimoto 0.6. The "
              "measured error roughly doubles there; treat those predictions as ranking "
              "hints, not estimates.", file=sys.stderr)


if __name__ == "__main__":
    main()
