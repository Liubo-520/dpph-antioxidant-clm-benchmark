"""
Fetch the two additional pretrained checkpoints added during revision and store
them locally in safetensors form so that every representation can be rebuilt
offline from a pinned local snapshot.
"""

import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOCAL = os.path.join(ROOT, "hf_local")


def bin_to_safetensors(repo_id, out_name, allow=None):
    dest = os.path.join(LOCAL, out_name)
    if os.path.exists(os.path.join(dest, "model.safetensors")):
        print(f"{out_name}: already present")
        return dest
    src = snapshot_download(repo_id, allow_patterns=allow)
    os.makedirs(dest, exist_ok=True)
    for fn in os.listdir(src):
        if fn in ("pytorch_model.bin", "training_args.bin", ".gitattributes") or fn.startswith("."):
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dest, fn))
    bin_path = os.path.join(src, "pytorch_model.bin")
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        # keep the frozen encoder only; the masked-LM head is not used for embeddings
        encoder = {
            k[len("roberta.") :]: v.clone().contiguous()
            for k, v in state.items()
            if isinstance(v, torch.Tensor) and k.startswith("roberta.")
        }
        save_file(encoder, os.path.join(dest, "model.safetensors"), metadata={"format": "pt"})
        cfg_path = os.path.join(dest, "config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["architectures"] = ["RobertaModel"]
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2)
    print(f"{out_name}: written to {dest}")
    return dest


def copy_safetensors(repo_id, out_name):
    dest = os.path.join(LOCAL, out_name)
    if os.path.exists(os.path.join(dest, "model.safetensors")):
        print(f"{out_name}: already present")
        return dest
    src = snapshot_download(repo_id)
    os.makedirs(dest, exist_ok=True)
    for fn in os.listdir(src):
        s = os.path.join(src, fn)
        if os.path.isfile(s) and not fn.startswith(".") and not fn.endswith((".jpeg", ".png")):
            shutil.copy2(s, os.path.join(dest, fn))
    print(f"{out_name}: written to {dest}")
    return dest


if __name__ == "__main__":
    bin_to_safetensors("DeepChem/ChemBERTa-77M-MLM", "chemberta_77m_mlm_safetensors")
    copy_safetensors("ibm/MoLFormer-XL-both-10pct", "molformer_xl_safetensors")
