"""Record the software environment and hardware used for the reported analyses."""

import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")

PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "rdkit", "mordred", "torch",
    "transformers", "tokenizers", "safetensors", "catboost", "xgboost",
    "umap-learn", "matplotlib", "Pillow",
]


def version(name):
    try:
        from importlib.metadata import version as v

        return v(name)
    except Exception:
        return None


def gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 else None
    except Exception:
        return None


def main():
    payload = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu(),
        "packages": {p: version(p) for p in PACKAGES},
    }
    try:
        import torch

        payload["cuda"] = torch.version.cuda
        payload["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    with open(os.path.join(OUT, "b0_environment.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
