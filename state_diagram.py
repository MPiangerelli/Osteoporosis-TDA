#!/usr/bin/env python3
# =============================================================================
# state_diagram.py -- DIAGRAMMA DI STATO: traiettoria parametrica (H1(a), H0(a))
# nel piano dei numeri di Betti, parametrizzata da alpha = -SDT.
#
# E' la versione corretta della vecchia "Figure 1" (H0 vs H1): invece di un
# fit log su valori mediati, mostriamo come lo STATO topologico (H0,H1) si
# muove nel piano al crescere della filtrazione alpha = -SDT.  Ogni punto della
# curva = uno stato; la stella = stato alpha=0 (maschera binaria di Otsu).
#
# Uso:  python pipeline/state_diagram.py
# =============================================================================
import argparse, os, sys, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gtda.homology import CubicalPersistence
from gtda.diagrams import Filtering
from full_persistence import white_mask, signed_distance, IMG_DIR, OUT_DIR, family_of

log = logging.getLogger("state")

SUBSET = {
    "WT 3m":   ("control", "tab:blue"),
    "SO 3m":   ("control", "tab:cyan"),
    "OVX 3m":  ("patho",   "tab:red"),
    "RANKL Tg":("patho",   "tab:orange"),
    "Aged 2y": ("patho",   "tab:brown"),
    "p62-3m":  ("patho",   "tab:pink"),
    "WT 6m":  ("control", "tab:green"),
    "WT 8m":  ("control", "tab:olive"),
}


def files_for(fam):
    return [f for f in sorted(os.listdir(IMG_DIR))
            if f.lower().endswith((".tif", ".tiff", ".png")) and family_of(f) == fam]


def betti_at(diag, dim, grid):
    """beta_dim(alpha) per ogni alpha in grid: # feature con birth<=alpha<death."""
    sub = diag[diag[:, 2] == dim]
    sub = sub[sub[:, 1] != sub[:, 0]]              # scarta il padding
    b, d = sub[:, 0], sub[:, 1]
    return np.array([np.sum((b <= a) & (d > a)) for a in grid], float)


def run(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    files, fam_of = [], []
    for fam in SUBSET:
        for f in files_for(fam):
            files.append(f); fam_of.append(fam)

    masks = [white_mask(os.path.join(IMG_DIR, f)) for f in files]
    H = min(m.shape[0] for m in masks); W = min(m.shape[1] for m in masks)
    sdts = np.stack([signed_distance(m[:H, :W]) for m in masks]).astype(float)
    cp = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=1)  # +SDT: H0 tessuto, H1 vuoti
    diags = Filtering(epsilon=args.epsilon).fit_transform(cp.fit_transform(sdts))

    grid = np.linspace(args.amin, args.amax, args.steps)
    a0 = int(np.argmin(np.abs(grid)))              # indice di alpha=0

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for fam, (grp, col) in SUBSET.items():
        idx = [i for i, fm in enumerate(fam_of) if fm == fam]
        if not idx:
            continue
        b0 = np.mean([betti_at(diags[i], 0, grid) for i in idx], axis=0)
        b1 = np.mean([betti_at(diags[i], 1, grid) for i in idx], axis=0)
        ax.plot(b1, b0, "-", color=col, lw=1.8, label=f"{fam} ({grp})", zorder=3)
        ax.scatter(b1[a0], b0[a0], color=col, marker="*", s=180,
                   edgecolor="k", linewidth=0.6, zorder=5)        # stato alpha=0
        ax.scatter(b1[0], b0[0], color=col, marker="o", s=18, zorder=4)  # inizio

    ax.set_xlabel(r"$H_1(\alpha)$ — adipocytic voids")
    ax.set_ylabel(r"$H_0(\alpha)$ — tissue components")
    ax.set_title(r"Topological state diagram in the $(H_1,H_0)$ plane, "
                 r"parametrised by $\alpha=+\mathrm{SDT}$ (tissue foreground)"
                 "\n(● start $\\alpha$ small  →  line  →  ★ = state at $\\alpha=0$, "
                 "binary tissue mask)")
    ax.legend(fontsize=9, title="family (group)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140); plt.close(fig)
    log.info("Salvato: %s", args.out)


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--amin", type=float, default=-14.0)
    p.add_argument("--amax", type=float, default=16.0)   # +SDT: i vuoti muoiono a alpha>0
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "state_diagram.png"))  # Figure 7
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    import warnings; warnings.filterwarnings("ignore")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
