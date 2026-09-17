#!/usr/bin/env python3
# diag_largescale.py -- rende esplicita la differenza di SCALA (non di conteggio)
# tra controlli e patologici: dimensione della lacuna midollare piu' grande e
# persistenza dominante, su tutte le 36 immagini, per gruppo/famiglia.
import os, sys, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gtda.homology import CubicalPersistence
from full_persistence import (white_mask, signed_distance, IMG_DIR, OUT_DIR,
                              family_of, GROUP, GROUP_COLOR, discover)

log = logging.getLogger("ls")
OUT = os.path.join(OUT_DIR, "diag_largescale.png")   # Figure 6


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    import warnings; warnings.filterwarnings("ignore")
    files, fams, groups = discover(None)
    masks = [white_mask(os.path.join(IMG_DIR, f)) for f in files]
    H = min(m.shape[0] for m in masks); W = min(m.shape[1] for m in masks)
    sdts = np.stack([signed_distance(m[:H, :W]) for m in masks]).astype(float)
    cp = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=1)  # +SDT: H1 = vuoti
    diags = cp.fit_transform(sdts)

    max_sdt = sdts.reshape(len(files), -1).max(axis=1)        # raggio lacuna max
    pers_void = np.array([(d[d[:, 2] == 1][:, 1] - d[d[:, 2] == 1][:, 0]).max()
                          if (d[:, 2] == 1).any() else 0.0 for d in diags])
    frac = np.array([m[:H, :W].mean() for m in masks])

    metrics = [("Largest adipocytic-void radius (max SDT, px)", max_sdt),
               ("Dominant void persistence (H1)", pers_void),
               ("Adipocyte (white) area fraction", frac)]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    gx = {"control": 0, "patho": 1}
    for a, (title, vals) in zip(ax, metrics):
        for i, (fam, gr) in enumerate(zip(fams, groups)):
            x = gx[gr] + np.random.uniform(-0.13, 0.13)
            a.scatter(x, vals[i], color=GROUP_COLOR.get(gr, "gray"), s=42,
                      edgecolor="k", linewidth=0.4, zorder=3)
        for g, xc in gx.items():
            v = vals[[i for i, gg in enumerate(groups) if gg == g]]
            a.hlines(np.median(v), xc - 0.25, xc + 0.25, color="k", lw=2, zorder=4)
        a.set_xticks([0, 1]); a.set_xticklabels(["control", "patho"])
        a.set_title(title, fontsize=10)
        a.set_xlim(-0.5, 1.5)
    fig.suptitle("Large-scale separation: marrow-space topology is dominated by "
                 "SIZE (persistence), not by feature COUNT", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140); plt.close(fig)
    log.info("Salvato: %s", OUT)

    # stampa mediane per gruppo
    for title, vals in metrics:
        c = np.median(vals[[i for i, g in enumerate(groups) if g == "control"]])
        p = np.median(vals[[i for i, g in enumerate(groups) if g == "patho"]])
        log.info("%-45s ctrl=%.2f  patho=%.2f  ratio=%.2f", title, c, p, p / c)


if __name__ == "__main__":
    main()
