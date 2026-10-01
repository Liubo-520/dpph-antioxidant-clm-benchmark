"""
Build and document every molecular representation used in the revision.

Answers Reviewer 4 Q13/Q14/Q16 and Reviewer 5 Q4/Q6: exact ChemBERTa checkpoint,
tokenizer, sequence length, pooling and freezing protocol, plus the additional
baseline representations (ECFP, MACCS, RDKit descriptors, Mordred, a second and
third pretrained chemical language model) and the wall-clock cost of each.
"""

import argparse
import json
import os
import pickle
import time
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FEAT, exist_ok=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--device", default="cpu", help="cpu keeps every featuriser on the same hardware for a fair cost comparison")
_ARGS, _ = _ap.parse_known_args()

DEVICE = torch.device(_ARGS.device)
MAX_LENGTH = 256
BATCH = 64

CLM_CHECKPOINTS = {
    # local snapshot of seyonec/ChemBERTa-zinc-base-v1 (RoBERTa, 6 layers, 768 hidden)
    "ChemBERTa-ZINC": os.path.join(ROOT, "hf_local", "chemberta_zinc_safetensors"),
    # local snapshot of DeepChem/ChemBERTa-77M-MTR (RoBERTa, 3 layers, 384 hidden)
    "ChemBERTa-MTR": os.path.join(ROOT, "hf_local", "chemberta_77m_mtr_safetensors"),
    # added for the revision: masked-LM sibling of the 77M model
    "ChemBERTa-MLM": os.path.join(ROOT, "hf_local", "chemberta_77m_mlm_safetensors"),
    # added for the revision: a different transformer family (linear attention, 1.1B-molecule corpus)
    "MoLFormer-XL": os.path.join(ROOT, "hf_local", "molformer_xl_safetensors"),
}


def load_smiles():
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        data = pickle.load(fh)
    return data["df"]["smiles"].tolist(), data["df"]["pIC50"].to_numpy(float)


def get_encoder(path):
    """Load the frozen encoder.

    Masked-LM checkpoints tie the input embedding matrix to the LM head, so they
    must be instantiated through the masked-LM class before the base encoder is
    extracted; loading them directly as a bare encoder silently reinitialises the
    token embeddings.
    """
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    architectures = getattr(config, "architectures", None) or []
    if any("MaskedLM" in a for a in architectures):
        backbone = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
        model = getattr(backbone, backbone.base_model_prefix)
    else:
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
    model.to(DEVICE).eval()
    n_missing = sum(
        1 for p in model.parameters() if p.numel() and not torch.isfinite(p).all()
    )
    assert n_missing == 0, f"non-finite weights in {path}"
    return tok, model


@torch.no_grad()
def embed(smiles, path, pooling="mean"):
    """Frozen encoder; mean pooling of the final hidden layer over real tokens."""
    tok, model = get_encoder(path)
    vecs, lengths = [], []
    for i in range(0, len(smiles), BATCH):
        chunk = smiles[i : i + BATCH]
        enc = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        lengths.extend(enc["attention_mask"].sum(1).tolist())
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if pooling == "mean":
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = hidden[:, 0]
        vecs.append(pooled.float().cpu().numpy())
    del model
    return np.vstack(vecs), np.array(lengths)


def morgan(smiles, radius, nbits, counts=False):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fp = gen.GetCountFingerprintAsNumPy(mol) if counts else gen.GetFingerprintAsNumPy(mol)
        rows.append(fp)
    return np.asarray(rows, dtype=np.float32)


def maccs(smiles):
    return np.asarray(
        [np.asarray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)), dtype=np.float32) for s in smiles]
    )


def rdkit_descriptors(smiles):
    names = [n for n, _ in Descriptors._descList]
    calc = [f for _, f in Descriptors._descList]
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        vals = []
        for f in calc:
            try:
                v = f(mol)
            except Exception:
                v = np.nan
            vals.append(v)
        rows.append(vals)
    X = np.asarray(rows, dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X, names


def main():
    smiles, y = load_smiles()
    timing, meta = {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        result = fn()
        timing[name] = float(time.perf_counter() - t0)
        print(f"{name:28s} {timing[name]:8.2f} s")
        return result

    store = {}

    store["ECFP4"] = timed("ECFP4 (r=2, 2048 bit)", lambda: morgan(smiles, 2, 2048))
    store["ECFP6"] = timed("ECFP6 (r=3, 2048 bit)", lambda: morgan(smiles, 3, 2048))
    store["ECFP4-count"] = timed("ECFP4 counts", lambda: morgan(smiles, 2, 2048, counts=True))
    store["MACCS"] = timed("MACCS (167 keys)", lambda: maccs(smiles))

    rdkit_X, rdkit_names = timed("RDKit descriptors", lambda: rdkit_descriptors(smiles))
    store["RDKit-desc"] = rdkit_X
    meta["RDKit-desc_names"] = rdkit_names

    z = np.load(os.path.join(ROOT, "results", "mordred_cache.npz"), allow_pickle=True)
    store["Mordred"] = z["X"].astype(float)
    meta["Mordred_names"] = [str(v) for v in z["feat"]]

    for name, path in CLM_CHECKPOINTS.items():
        vecs, lengths = timed(f"{name} embeddings", lambda p=path: embed(smiles, p))
        store[name] = vecs
        meta[f"{name}_token_length"] = {
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
            "p99": float(np.percentile(lengths, 99)),
            "n_truncated_at_max_length": int((lengths >= MAX_LENGTH).sum()),
        }

    # 199 normalised multitask-regression outputs kept from the original workflow
    store["ChemBERTa-MTR-199"] = np.load(
        os.path.join(ROOT, "results", "chemberta_mtr_199_features.npy")
    ).astype(float)

    # hybrid controls
    store["ChemBERTa-ZINC+Mordred"] = np.hstack([store["ChemBERTa-ZINC"], store["Mordred"]])
    store["ChemBERTa-MTR-199+Mordred"] = np.hstack([store["ChemBERTa-MTR-199"], store["Mordred"]])

    # reproducibility check against the embeddings used in the first submission
    legacy = np.load(os.path.join(ROOT, "results", "chemberta_avg_embeddings.npy")).astype(float)
    new = store["ChemBERTa-ZINC"].astype(float)
    cols = min(legacy.shape[1], new.shape[1])
    per_mol_r = np.array(
        [np.corrcoef(legacy[i, :cols], new[i, :cols])[0, 1] for i in range(len(smiles))]
    )
    meta["embedding_reproduction_check"] = {
        "legacy_shape": list(legacy.shape),
        "recomputed_shape": list(new.shape),
        "max_abs_difference": float(np.abs(legacy[:, :cols] - new[:, :cols]).max()),
        "median_per_molecule_pearson_r": float(np.median(per_mol_r)),
        "min_per_molecule_pearson_r": float(per_mol_r.min()),
    }

    np.savez_compressed(
        os.path.join(FEAT, "representations.npz"),
        **{k: v.astype(np.float32) for k, v in store.items()},
    )

    summary = {
        "extraction_protocol": {
            "input": "isomeric canonical SMILES produced by RDKit MolToSmiles",
            "tokenizer": "checkpoint-matched byte-level BPE tokenizer shipped with each model",
            "max_sequence_length": MAX_LENGTH,
            "padding": "dynamic within batch, batch size 64",
            "encoder_state": "frozen; no gradient update and no fine-tuning of encoder weights",
            "layer": "final hidden layer (last_hidden_state)",
            "pooling": "attention-mask-weighted mean over all non-padding tokens",
            "dtype": "float32",
            "device": str(DEVICE),
        },
        "checkpoints": {
            "ChemBERTa-ZINC": "seyonec/ChemBERTa-zinc-base-v1 (RoBERTa, 6 layers, 12 heads, hidden 768, vocab 767)",
            "ChemBERTa-MTR": "DeepChem/ChemBERTa-77M-MTR (RoBERTa, 3 layers, 12 heads, hidden 384, vocab 600)",
            "ChemBERTa-MLM": "DeepChem/ChemBERTa-77M-MLM (RoBERTa, 3 layers, 12 heads, hidden 384)",
            "MoLFormer-XL": "ibm/MoLFormer-XL-both-10pct (linear-attention transformer, 12 layers, hidden 768)",
        },
        "dimensions": {k: list(v.shape) for k, v in store.items()},
        "wall_clock_seconds": timing,
        "seconds_per_1000_molecules": {
            k: float(1000 * v / len(smiles)) for k, v in timing.items()
        },
        "n_molecules": len(smiles),
        "meta": meta,
    }
    with open(os.path.join(OUT, "a2_representations.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote", os.path.join(OUT, "a2_representations.json"))
    print(json.dumps(summary["dimensions"], indent=2))
    print(json.dumps(meta["embedding_reproduction_check"], indent=2))


if __name__ == "__main__":
    main()
