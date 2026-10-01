"""Render atom-level attribution maps for four representative antioxidants.

Used as the bottom row of manuscript Figure 5. The maps illustrate the
dataset-wide statistics reported in the panels above them; they are not the
evidence themselves.
"""

import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

EXAMPLES = [
    ("Quercetin (flavonol)", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"),
    ("Luteolin (flavone)", "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"),
    ("Caffeic acid (phenolic acid)", "OC(=O)/C=C/c1ccc(O)c(O)c1"),
    ("Resveratrol (stilbene)", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1"),
]


def _attribution(smiles):
    """Atom attributions from the deployed fine-tuned ensemble."""
    from a6_attribution import Ensemble, atom_spans, project, tokens_to_atoms

    ens = _attribution.cache
    if ens is None:
        ens = _attribution.cache = Ensemble(42)
    out = []
    for _, smi in [(None, smiles)]:
        spans, mol = atom_spans(smi)
        enc, offsets = ens.encode(smi)
        token_atoms, _ = tokens_to_atoms(offsets, spans, enc["input_ids"].shape[1])
        scores = ens.grad_x_input(enc)
        a = project(np.asarray(scores, float), token_atoms, mol.GetNumAtoms())
        out.append((mol, a / max(a.max(), 1e-12)))
    return out[0]


_attribution.cache = None


def render_examples(axes):
    """Draw two composite panels, two molecules each, onto the supplied axes."""
    images = []
    for name, smi in EXAMPLES:
        mol, attr = _attribution(smi)
        colours, radii = {}, {}
        import matplotlib.cm as mcm

        for i, v in enumerate(attr):
            colours[i] = tuple(mcm.YlOrRd(0.15 + 0.85 * float(v))[:3])
            radii[i] = 0.42
        drawer = rdMolDraw2D.MolDraw2DCairo(420, 300)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = False
        opts.bondLineWidth = 2
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=list(range(mol.GetNumAtoms())),
            highlightAtomColors=colours,
            highlightAtomRadii=radii,
            highlightBonds=[],
        )
        drawer.FinishDrawing()
        import io

        from PIL import Image

        images.append((name, Image.open(io.BytesIO(drawer.GetDrawingText()))))

    for ax, pair in zip(axes, [images[:2], images[2:]]):
        widths = sum(im.width for _, im in pair)
        height = max(im.height for _, im in pair)
        from PIL import Image

        canvas = Image.new("RGB", (widths, height), "white")
        x = 0
        for _, im in pair:
            canvas.paste(im, (x, 0))
            x += im.width
        ax.imshow(np.asarray(canvas))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("   ·   ".join(n for n, _ in pair), loc="left", fontsize=7.2)
