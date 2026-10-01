"""
Quantitative, multi-method, multi-seed attribution analysis.

Answers Reviewer 1 Q3, Reviewer 3, Reviewer 4 Q10/Q11 and Reviewer 5 Q3.

Attribution is computed on the deployed end-to-end predictor (the fold ensemble
of leak-free fine-tuned ChemBERTa-ZINC regressors), not on a separate model, and
is evaluated with four independent explanation methods:

  * gradient x input on token embeddings
  * integrated gradients against a masked baseline
  * single-token occlusion
  * attention rollout

For every method we report (i) dataset-wide enrichment of attribution on
chemically defined atom groups with a within-molecule permutation null and
Benjamini-Hochberg correction, (ii) cross-method and cross-seed rank agreement,
and (iii) a faithfulness test that deletes the top-ranked atoms and measures the
resulting change in predicted activity against a random-deletion control.
"""

import json
import os
import pickle
import re
import warnings
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy import stats
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(ANALYSIS))
OUT = os.path.join(ANALYSIS, "outputs")
FEAT = os.path.join(ANALYSIS, "features")
CKPT = os.path.join(ANALYSIS, "checkpoints")
import sys

sys.path.insert(0, os.path.join(ROOT, "code"))
from llm_models import HFSmilesRegressor

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT, "hf_local", "chemberta_zinc_safetensors")
MAX_LENGTH = 256
IG_STEPS = 32
N_FOLDS = 5

ATOM_TOKEN = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|se|As|as|B|C|N|O|P|S|F|I|b|c|n|o|p|s")


# ------------------------------------------------------------ atom bookkeeping
def atom_spans(smiles):
    """Character span of every atom, in RDKit atom-index order."""
    spans = [(m.start(), m.end()) for m in ATOM_TOKEN.finditer(smiles)]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(spans) != mol.GetNumAtoms():
        return None, mol
    return spans, mol


def atom_groups(mol):
    """Chemically defined atom sets used to test whether attribution is selective."""
    groups = defaultdict(set)
    n = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.GetIsAromatic():
            groups["aromatic"].add(i)
        else:
            groups["non_aromatic"].add(i)
        if atom.GetSymbol() == "O" and atom.GetTotalNumHs() >= 1:
            nbrs = atom.GetNeighbors()
            if nbrs and nbrs[0].GetIsAromatic():
                groups["phenolic_OH_oxygen"].add(i)
                groups["phenolic_OH_and_ipso_carbon"].add(i)
                groups["phenolic_OH_and_ipso_carbon"].add(nbrs[0].GetIdx())
            else:
                groups["aliphatic_OH_oxygen"].add(i)
    catechol = Chem.MolFromSmarts("[OX2H]c1ccccc1[OX2H]")
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts("c([OX2H])c[OX2H]")):
        groups["catechol_motif"].update(match)
    for match in mol.GetSubstructMatches(catechol):
        groups["catechol_motif"].update(match)
    # [#6] rather than [CX3], so that the 4-oxo group of a flavone or flavonol,
    # whose carbon RDKit perceives as aromatic, is counted as a carbonyl
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[#6X3]=[OX1]")):
        groups["carbonyl"].update(match)
    # a C=C conjugated either with a second C=C or with an aromatic ring; the
    # latter is what makes stilbenes and cinnamic acids conjugated, and a
    # diene-only pattern matches none of the held-out compounds
    for smarts in ("[#6]=[#6]-[#6]=[#6]", "[#6]=[#6]-[a]"):
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            groups["conjugated_CC"].update(match)
    groups["all"] = set(range(n))
    return {k: sorted(v) for k, v in groups.items() if v}


# --------------------------------------------------------------- model wrapper
class Ensemble:
    """Fold ensemble of fine-tuned SMILES regressors with attribution hooks."""

    def __init__(self, seed, n_folds=N_FOLDS, split_suffix=""):
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.models = []
        for fold in range(1, n_folds + 1):
            p = os.path.join(CKPT, f"chemberta_zinc{split_suffix}_seed{seed}_fold{fold}.pt")
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = HFSmilesRegressor(
                model_name=MODEL_PATH,
                dropout=ck["dropout"],
                pooling=ck["pooling"],
                freeze_encoder=False,
                local_files_only=True,
            )
            m.load_state_dict(ck["model_state_dict"])
            m.to(DEVICE).eval()
            # The default scaled-dot-product attention kernel cannot return
            # attention weights: transformers warns and hands back an empty
            # tuple, which silently turns attention rollout into the identity
            # matrix and every atom-level attention score into zero. The eager
            # kernel is numerically equivalent and does return them.
            try:
                m.encoder.set_attn_implementation("eager")
            except (AttributeError, ValueError):
                m.encoder.config._attn_implementation = "eager"
            self.models.append(m)
        self.mask_id = self.tok.mask_token_id
        self.embedding_layers = [m.encoder.embeddings.word_embeddings for m in self.models]

    def encode(self, smiles):
        enc = self.tok(
            smiles,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        return {k: v.to(DEVICE) for k, v in enc.items()}, offsets

    @torch.no_grad()
    def predict_ids(self, input_ids, attention_mask):
        return float(
            np.mean([m(input_ids=input_ids, attention_mask=attention_mask).item() for m in self.models])
        )

    def _forward_from_embeds(self, model, embeds, attention_mask):
        """Same read-out as the deployed predictor, entered at the embedding layer."""
        out = model.encoder(inputs_embeds=embeds, attention_mask=attention_mask)
        return model.regressor(model._pool(out, attention_mask)).squeeze(-1)

    def grad_x_input(self, enc):
        scores = []
        for model, emb_layer in zip(self.models, self.embedding_layers):
            embeds = emb_layer(enc["input_ids"]).detach().clone().requires_grad_(True)
            pred = self._forward_from_embeds(model, embeds, enc["attention_mask"])
            grad = torch.autograd.grad(pred.sum(), embeds)[0]
            scores.append((grad * embeds).sum(-1).detach().abs()[0].cpu().numpy())
        return np.mean(scores, axis=0)

    def integrated_gradients(self, enc, steps=IG_STEPS):
        base_ids = enc["input_ids"].clone()
        special = torch.tensor(
            self.tok.get_special_tokens_mask(base_ids[0].tolist(), already_has_special_tokens=True),
            device=DEVICE,
        ).bool()
        base_ids[0, ~special] = self.mask_id
        scores = []
        for model, emb_layer in zip(self.models, self.embedding_layers):
            x = emb_layer(enc["input_ids"]).detach()
            b = emb_layer(base_ids).detach()
            total = torch.zeros_like(x)
            for a in np.linspace(1.0 / steps, 1.0, steps):
                point = (b + a * (x - b)).requires_grad_(True)
                pred = self._forward_from_embeds(model, point, enc["attention_mask"])
                total = total + torch.autograd.grad(pred.sum(), point)[0].detach()
            ig = ((x - b) * total / steps).sum(-1).abs()[0].cpu().numpy()
            scores.append(ig)
        return np.mean(scores, axis=0)

    def occlusion(self, enc):
        base = self.predict_ids(enc["input_ids"], enc["attention_mask"])
        ids = enc["input_ids"]
        n = ids.shape[1]
        out = np.zeros(n)
        for t in range(n):
            if ids[0, t].item() in self.tok.all_special_ids:
                continue
            perturbed = ids.clone()
            perturbed[0, t] = self.mask_id
            out[t] = abs(base - self.predict_ids(perturbed, enc["attention_mask"]))
        return out

    @torch.no_grad()
    def attention_rollout(self, enc):
        rolled = []
        for model in self.models:
            out = model.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_attentions=True,
            )
            n = enc["input_ids"].shape[1]
            joint = torch.eye(n, device=DEVICE)
            for layer in out.attentions:
                a = layer[0].mean(0)
                a = a + torch.eye(n, device=DEVICE)
                a = a / a.sum(-1, keepdim=True)
                joint = a @ joint
            rolled.append(joint[0].cpu().numpy())
        return np.mean(rolled, axis=0)

    def delete_atoms(self, enc, token_ids_to_mask):
        ids = enc["input_ids"].clone()
        for t in token_ids_to_mask:
            ids[0, t] = self.mask_id
        return self.predict_ids(ids, enc["attention_mask"])


# ----------------------------------------------------------- token -> atom map
def tokens_to_atoms(offsets, spans, n_tokens):
    """Map each token position onto the atoms whose characters it covers."""
    token_atoms = [[] for _ in range(n_tokens)]
    atom_tokens = [[] for _ in range(len(spans))]
    for t, (a, b) in enumerate(offsets):
        if a == b:
            continue
        for ai, (s, e) in enumerate(spans):
            if a < e and s < b:
                token_atoms[t].append(ai)
                atom_tokens[ai].append(t)
    return token_atoms, atom_tokens


def project(token_scores, token_atoms, n_atoms):
    """Distribute each token score evenly over the atoms it covers."""
    out = np.zeros(n_atoms)
    for t, atoms in enumerate(token_atoms):
        if not atoms:
            continue
        out[np.array(atoms)] += token_scores[t] / len(atoms)
    return out


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[idx] * m / k)
        ranked[idx] = prev
    return ranked


def main():
    with open(os.path.join(ROOT, "results", "processed_data.pkl"), "rb") as fh:
        df = pickle.load(fh)["df"]
    smiles_all = df["smiles"].tolist()
    y_all = df["pIC50"].to_numpy(float)
    splits = json.load(open(os.path.join(OUT, "a4_splits.json"), encoding="utf-8"))["splits"]
    test_idx = np.array(splits["stratified"]["test_idx"])

    seeds = [42, 1, 2026]
    available = [
        s
        for s in seeds
        if os.path.exists(os.path.join(CKPT, f"chemberta_zinc_seed{s}_fold{N_FOLDS}.pt"))
    ]
    print("Attribution seeds available:", available)
    ensembles = {s: Ensemble(s) for s in available}
    primary = ensembles[available[0]]

    methods = ["grad_x_input", "integrated_gradients", "occlusion", "attention_rollout"]
    group_scores = defaultdict(lambda: defaultdict(list))
    perm_null = defaultdict(lambda: defaultdict(list))
    cross_method_rho = defaultdict(list)
    cross_seed_rho = defaultdict(list)
    faithfulness = defaultdict(list)
    per_molecule = []
    rng = np.random.default_rng(0)

    subset = test_idx  # every held-out molecule, not selected examples
    for count, mi in enumerate(subset):
        smi = smiles_all[mi]
        spans, mol = atom_spans(smi)
        if spans is None or mol.GetNumAtoms() < 4:
            continue
        enc, offsets = primary.encode(smi)
        n_tokens = enc["input_ids"].shape[1]
        token_atoms, atom_tokens = tokens_to_atoms(offsets, spans, n_tokens)
        if not any(token_atoms):
            continue
        groups = atom_groups(mol)
        n_atoms = mol.GetNumAtoms()

        atom_attr = {}
        for method in methods:
            fn = getattr(primary, method)
            token_scores = fn(enc)
            a = project(np.asarray(token_scores, dtype=float), token_atoms, n_atoms)
            if a.sum() <= 0:
                a = np.ones(n_atoms)
            a = a / a.mean()
            atom_attr[method] = a

            for gname, members in groups.items():
                if gname == "all" or len(members) == 0 or len(members) == n_atoms:
                    continue
                obs = float(a[members].mean())
                group_scores[method][gname].append(obs)
                null = [float(a[rng.permutation(n_atoms)[: len(members)]].mean()) for _ in range(200)]
                perm_null[method][gname].append(float(np.mean(null)))

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                rho = stats.spearmanr(atom_attr[methods[i]], atom_attr[methods[j]]).statistic
                if np.isfinite(rho):
                    cross_method_rho[f"{methods[i]} vs {methods[j]}"].append(float(rho))

        for s in available[1:]:
            other = ensembles[s].grad_x_input(enc)
            a2 = project(np.asarray(other, dtype=float), token_atoms, n_atoms)
            rho = stats.spearmanr(atom_attr["grad_x_input"], a2).statistic
            if np.isfinite(rho):
                cross_seed_rho[f"seed {available[0]} vs seed {s}"].append(float(rho))

        # faithfulness: mask the tokens of the k highest-attributed atoms
        base = primary.predict_ids(enc["input_ids"], enc["attention_mask"])
        k = max(1, int(round(0.20 * n_atoms)))
        for method in methods:
            top = np.argsort(atom_attr[method])[::-1][:k]
            toks = sorted({t for ai in top for t in atom_tokens[ai]})
            drop = base - primary.delete_atoms(enc, toks)
            rand_drops = []
            for _ in range(10):
                ra = rng.permutation(n_atoms)[:k]
                rtoks = sorted({t for ai in ra for t in atom_tokens[ai]})
                rand_drops.append(base - primary.delete_atoms(enc, rtoks))
            faithfulness[method].append(
                {"top_k_drop": float(abs(drop)), "random_drop": float(np.mean(np.abs(rand_drops)))}
            )

        per_molecule.append(
            {
                "index": int(mi),
                "smiles": smi,
                "pIC50": float(y_all[mi]),
                "prediction": float(base),
                "n_atoms": int(n_atoms),
                "n_phenolic_OH": int(len(groups.get("phenolic_OH_oxygen", []))),
                "mean_attr_phenolic": float(
                    atom_attr["grad_x_input"][groups["phenolic_OH_oxygen"]].mean()
                )
                if groups.get("phenolic_OH_oxygen")
                else None,
                "mean_attr_aromatic": float(atom_attr["grad_x_input"][groups["aromatic"]].mean())
                if groups.get("aromatic")
                else None,
            }
        )
        if (count + 1) % 25 == 0:
            print(f"  {count + 1}/{len(subset)} molecules processed")

    # ---------------------------------------------------------------- summary
    enrichment_rows = []
    raw_p = []
    for method in methods:
        for gname, obs in group_scores[method].items():
            obs = np.asarray(obs)
            null = np.asarray(perm_null[method][gname])
            if len(obs) < 8:
                continue
            w = stats.wilcoxon(obs, null, alternative="two-sided")
            enrichment_rows.append(
                {
                    "method": method,
                    "atom_group": gname,
                    "n_molecules": int(len(obs)),
                    "mean_enrichment": float(obs.mean()),
                    "mean_permutation_null": float(null.mean()),
                    "median_enrichment": float(np.median(obs)),
                    "pct_molecules_enriched": float(100 * (obs > null).mean()),
                    "wilcoxon_statistic": float(w.statistic),
                    "p_raw": float(w.pvalue),
                }
            )
            raw_p.append(w.pvalue)
    if raw_p:
        adj = bh(raw_p)
        for row, q in zip(enrichment_rows, adj):
            row["p_bh_adjusted"] = float(q)

    faith_rows = []
    for method, entries in faithfulness.items():
        top = np.array([e["top_k_drop"] for e in entries])
        rnd = np.array([e["random_drop"] for e in entries])
        w = stats.wilcoxon(top, rnd, alternative="greater")
        faith_rows.append(
            {
                "method": method,
                "n_molecules": int(len(top)),
                "mean_abs_change_top20pct_atoms": float(top.mean()),
                "mean_abs_change_random_atoms": float(rnd.mean()),
                "ratio": float(top.mean() / max(rnd.mean(), 1e-9)),
                "wilcoxon_p_greater": float(w.pvalue),
            }
        )

    payload = {
        "target_model": (
            "fold ensemble of leak-free fine-tuned ChemBERTa-ZINC regressors "
            "(the deployed predictor, not a separate interpretability model)"
        ),
        "n_molecules_analysed": len(per_molecule),
        "molecule_selection": "all held-out compounds of the chemistry-stratified split",
        "seeds": available,
        "methods": methods,
        "token_to_atom_rule": (
            "byte-level BPE offsets are intersected with the character spans of the atom "
            "symbols in the canonical SMILES; each token contributes its score in equal parts "
            "to every atom it covers, and atoms split over several tokens accumulate all of them"
        ),
        "enrichment": enrichment_rows,
        "faithfulness": faith_rows,
        "cross_method_spearman": {
            k: {
                "median": float(np.median(v)),
                "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                "n": len(v),
            }
            for k, v in cross_method_rho.items()
        },
        "cross_seed_spearman": {
            k: {
                "median": float(np.median(v)),
                "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                "n": len(v),
            }
            for k, v in cross_seed_rho.items()
        },
    }

    mol_df = pd.DataFrame(per_molecule)
    if len(mol_df) > 5:
        sub = mol_df.dropna(subset=["mean_attr_phenolic"])
        if len(sub) > 5:
            r = stats.spearmanr(sub["n_phenolic_OH"], sub["pIC50"])
            payload["phenolic_count_vs_activity"] = {
                "spearman_rho": float(r.statistic),
                "p": float(r.pvalue),
                "n": int(len(sub)),
            }
    mol_df.to_csv(os.path.join(OUT, "a6_attribution_per_molecule.csv"), index=False)
    pd.DataFrame(enrichment_rows).to_csv(os.path.join(OUT, "a6_attribution_enrichment.csv"), index=False)
    with open(os.path.join(OUT, "a6_attribution.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps({k: v for k, v in payload.items() if k != "enrichment"}, indent=2)[:4000])
    print(pd.DataFrame(enrichment_rows).to_string(index=False)[:6000])


if __name__ == "__main__":
    main()
