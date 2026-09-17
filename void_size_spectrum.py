#!/usr/bin/env python3
# =============================================================================
# void_size_spectrum.py -- lo SPETTRO DELLE TAGLIE DEI VUOTI BIANCHI.
#
# Formalizza l'idea originale ("piu' un vuoto e' grande, piu' lentamente
# l'erosione lo elimina"): il valore della distance transform al centro di un
# vuoto = numero di erosioni che sopravvive = il suo raggio.  Quindi:
#
#  - Pannello A: CURVA DI SOPRAVVIVENZA ALL'EROSIONE, A(k) = frazione di area
#    bianca contenuta in vuoti con raggio >= k (= area(SDT>=k)/area(SDT>=1)).
#    E' il vecchio metodo erode, ma ESATTO e senza l'artefatto iter 9->10.
#  - Pannello B: distribuzione del RAGGIO del vuoto piu' grande per immagine
#    (per componente connessa), controllo vs patologico.
#
# Threshold-free: confrontiamo le distribuzioni; il punto dove i gruppi
# divergono suggerisce una soglia "vuoto grande = osteoporosi" data-driven.
# =============================================================================
import argparse, os, sys, logging
import numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from full_persistence import (white_mask, IMG_DIR, OUT_DIR, GROUP_COLOR, discover)

log = logging.getLogger("void")


def run(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    files, fams, groups = discover(None)
    masks = [white_mask(os.path.join(IMG_DIR, f)) for f in files]
    H = min(m.shape[0] for m in masks); W = min(m.shape[1] for m in masks)

    # due strati: patologici e controlli (WT 6m e WT 8m inclusi fra i controlli;
    # grandi spazi sinusoidali fisiologici) tenuti separati.
    def layer(i):
        return groups[i]
    LAYERS = {"patho": "tab:red", "control": "tab:blue"}
    lay = [layer(i) for i in range(len(files))]

    kgrid = np.arange(0, args.kmax + 1)
    surv = np.zeros((len(files), len(kgrid)))    # frazione di IMMAGINE in voids>=k
    max_radii = np.zeros(len(files))             # raggio del vuoto piu' grande
    imgarea = H * W
    for i, m in enumerate(masks):
        m = m[:H, :W]
        dt = ndi.distance_transform_edt(m)       # raggio dentro i vuoti bianchi
        surv[i] = [(dt >= k).sum() / imgarea for k in kgrid]
        max_radii[i] = dt.max()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- A: void burden vs scala, per strato (frazione di immagine) ---
    for g, c in LAYERS.items():
        idx = [i for i in range(len(files)) if lay[i] == g]
        if not idx:
            continue
        mean = surv[idx].mean(axis=0); sd = surv[idx].std(axis=0)
        ax[0].plot(kgrid, mean, color=c, lw=2, label=f"{g} (n={len(idx)})")
        ax[0].fill_between(kgrid, np.clip(mean - sd, 1e-6, None), mean + sd,
                           color=c, alpha=0.13, lw=0)
    ip = [i for i in range(len(files)) if lay[i] == "patho"]
    ax[0].set_xlabel("void radius threshold  k  (= erosion steps survived, px)")
    ax[0].set_ylabel("fraction of IMAGE area in voids of radius ≥ k")
    ax[0].set_title("Large-void burden vs scale (exact erosion, via DT)")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8)

    # --- B: distribuzione del raggio del vuoto piu' grande per strato ---
    bins = np.arange(0, args.kmax + 6, 6)
    for g, c in LAYERS.items():
        idx = [i for i in range(len(files)) if lay[i] == g]
        ax[1].hist(max_radii[idx], bins=bins, color=c, alpha=0.5, label=g,
                   edgecolor="k")
    ax[1].set_xlabel("radius of the LARGEST void per image (px)")
    ax[1].set_ylabel("# images")
    ax[1].set_title("Largest-void size distribution per layer")
    ax[1].legend(fontsize=8)

    fig.suptitle("Void-size spectrum — large white voids (adipocytic/rarefied) "
                 "as the osteoporosis signature", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140); plt.close(fig)
    log.info("Salvato: %s", args.out)
    for g in LAYERS:
        idx = [i for i in range(len(files)) if lay[i] == g]
        log.info("max-void radius  %-18s = %.1f±%.1f px",
                 g, max_radii[idx].mean(), max_radii[idx].std())


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--kmax", type=int, default=60)
    p.add_argument("--rmin", type=float, default=2.0)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "void_size_spectrum.png"))  # Figure 5
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    import warnings; warnings.filterwarnings("ignore")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
