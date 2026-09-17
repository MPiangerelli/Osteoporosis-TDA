#!/usr/bin/env python3
# =============================================================================
# full_persistence.py -- core pipeline of the paper (stage 1)
#
#   colour segmentation of adipocytic voids (fixed HSV thresholds, see below)
#   -> signed distance transform -> cubical persistent homology (H0, H1)
#   -> diagram filtering -> persistent entropy / Wasserstein amplitude
#   -> persistence-landscape distance matrix -> Ward clustering
#   -> leave-one-replicate-out validation.
#
# Configuration (environment variables, all optional):
#   OSTEO_TDA_IMAGES   folder with the 36 micrographs  (default: ./data/images)
#   OSTEO_TDA_RESULTS  output folder                    (default: ./results)
#   OSTEO_TDA_MASK     "fixed" (paper, default) or "hue" (stain-robust variant
#                      used by the screening app; NOT the paper configuration)
#
# Original Italian development notes follow.
# -----------------------------------------------------------------------------
#
# Generalizza proto_persistence.py alle 36 immagini (12 famiglie x 3 repliche).
# Per ogni REPLICA:  maschera a COLORI dei vuoti adipocitari (bianco vero in HSV,
#   esclude i vasi rosa) -> signed distance transform -> persistenza cubica
#   (H0,H1) -> descrittori vettoriali (entropia, ampiezza Wasserstein) ->
#   matrice di distanze (persistence landscape) replica x replica.
#
# Analisi a valle:
#   - dendrogramma gerarchico (Ward su Wasserstein) colorato per gruppo
#   - heatmap 36x36 ordinata per famiglia
#   - scatter dei descrittori (entropia vs ampiezza) per replica/gruppo
#   - VALIDAZIONE leave-one-replica-out: kNN sulla Wasserstein, accuratezza
#     nel ritrovare (a) la famiglia e (b) il gruppo controllo/patologico.
#
# Risolve i problemi aperti del README della pipeline:
#   #1 (artefatto iter9->10): niente conteggio su immagine erosa, si usa la DT
#   #2 (fit log degenere): niente fit, descrittori stabili dalla persistenza
#   #3 (scale incomparabili): metrica di Wasserstein canonica e calibrata
#   #4 (no validazione): leave-one-replica-out esplicito
#
# Uso:  python pipeline/full_persistence.py            # run completo
#       python pipeline/full_persistence.py --smoke 6  # solo prime 6 immagini
# =============================================================================

import argparse
import logging
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from gtda.homology import CubicalPersistence
from gtda.diagrams import (PersistenceEntropy, Amplitude, PairwiseDistance,
                           BettiCurve, Filtering)

log = logging.getLogger("full")

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.environ.get("OSTEO_TDA_IMAGES", os.path.join(_HERE, "data", "images"))
OUT_DIR = os.environ.get("OSTEO_TDA_RESULTS", os.path.join(_HERE, "results"))
MASK = os.environ.get("OSTEO_TDA_MASK", "fixed")   # "fixed" = paper

# Gruppo biologico per famiglia. *** DA CONFERMARE CON DIMITRIOS ***
# control = controllo/sano ; patho = modello osteoporotico/patologico
GROUP = {
    "WT 3m": "control", "WT 6m": "control", "WT 8m": "control",
    "SO 3m": "control", "RANKL Ctrl": "control",
    "OVX 3m": "patho", "RANKL Tg": "patho",
    "p62-3m": "patho", "p62-1y": "patho", "p62-2y": "patho",
    "Aged 1y": "patho", "Aged 2y": "patho",
}
GROUP_COLOR = {"control": "tab:blue", "patho": "tab:red"}


def family_of(fname):
    """Estrae la famiglia dal nome file rimuovendo il suffisso di replica
    (numerale romano finale I/II/III). '_' -> spazio per normalizzare."""
    base = re.sub(r"\.(tiff?|png)$", "", fname, flags=re.I).replace("_", " ").strip()
    base = re.sub(r"\s+I{1,3}$", "", base).strip()
    return base


def otsu(gray, nbins=256):
    h, edges = np.histogram(gray.ravel(), bins=nbins, range=(0, 1))
    p = h.astype(float) / h.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * ((edges[:-1] + edges[1:]) / 2))
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return (edges[k] + edges[k + 1]) / 2


def load_gray(path):
    g = np.array(Image.open(path)).astype(float)
    if g.ndim == 3:
        g = g[..., :3].mean(axis=2)
    return g / g.max()


# --- segmentazione a COLORI dei vuoti adipocitari (driver dell'osteoporosi) ---
# Il grigio (media RGB) confonde il BIANCO degli adipociti con il ROSA dei vasi.
# Un vuoto adipocitario e' un pixel CHIARO che NON e' eosina rosa/rossa satura:
# resta valido anche quando il vuoto e' tinto d'azzurro (cast di colorazione tra
# vetrini, es. OVX 1y), mentre i vasi/tessuto (rosa saturo) vengono esclusi.
WHITE_V_THR = 0.78      # luminosita' minima (vuoto chiaro)
VOID_S_CAP = 0.35       # saturazione massima ammessa per un vuoto
PINK_S_MIN = 0.15       # sopra questa satur., una tinta rosa/rossa = tessuto/vaso


# Bring high-resolution acquisitions to the common pixel scale: any image wider
# than 1.5x the target width is resampled (LANCZOS) to the target size before
# segmentation. The six WT 6m and WT 8m micrographs were acquired at 2088x1550 px over
# the same field of view as the ~508x376 px sections of the other groups.
# Paper configuration: "508x377". Set OSTEO_TDA_RESCALE="" to disable.
RESCALE = os.environ.get("OSTEO_TDA_RESCALE", "508x377")


def load_rgb(path):
    im = Image.open(path)
    if RESCALE:
        tw, th = (int(v) for v in RESCALE.lower().split("x"))
        if im.size[0] > 1.5 * tw:
            im = im.convert("RGB").resize((tw, th), Image.LANCZOS)
    a = np.array(im).astype(float)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return a[..., :3] / max(a[..., :3].max(), 1.0)


def white_balance(rgb, pct=98):
    """von Kries: porta a bianco neutro il punto di bianco (pixel piu' chiari),
    correggendo il CAST di colorazione tra vetrini SENZA alterare la QUANTITA'
    di bianco (quasi-identita' per immagini gia' calibrate come i 36 riferimenti)."""
    wp = np.percentile(rgb.reshape(-1, 3), pct, axis=0)
    return np.clip(rgb / np.maximum(wp, 1e-3), 0, 1)


# --- PAPER mask: true white in HSV, thresholds fixed a priori ---------------
WHITE_S_MAX = 0.15      # saturation below this  = unstained
WHITE_V_MIN = 0.82      # value above this       = bright


def fixed_white_mask(path):
    """Adipocytic-void mask used in the paper: low saturation AND high value.
    No white balance, no hue rule."""
    hsv = mcolors.rgb_to_hsv(load_rgb(path))
    return (hsv[..., 1] < WHITE_S_MAX) & (hsv[..., 2] > WHITE_V_MIN)


def hue_white_mask(path):
    """Stain-robust variant (screening app): white-balance -> bright pixel that
    is NOT saturated pink/red eosin. Not used for the paper results."""
    hsv = mcolors.rgb_to_hsv(white_balance(load_rgb(path)))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = V > WHITE_V_THR
    pink = ((H > 0.83) | (H < 0.08)) & (S > PINK_S_MIN)   # eosina = vaso/tessuto
    return bright & ~pink & (S < VOID_S_CAP)


def white_mask(path):
    return fixed_white_mask(path) if MASK == "fixed" else hue_white_mask(path)


def signed_distance(mask):
    return ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)


def discover(smoke=None):
    if not os.path.isdir(IMG_DIR):
        raise SystemExit(f"Image folder not found: {IMG_DIR}  (set OSTEO_TDA_IMAGES)")
    files = sorted(f for f in os.listdir(IMG_DIR)
                   if f.lower().endswith((".tif", ".tiff", ".png")))
    if smoke:
        files = files[:smoke]
    fams = [family_of(f) for f in files]
    groups = [GROUP.get(fam, "unknown") for fam in fams]
    return files, fams, groups


def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    files, fams, groups = discover(args.smoke)
    log.info("Immagini: %d | famiglie: %d | gruppi sconosciuti: %d",
             len(files), len(set(fams)), groups.count("unknown"))
    for f, fam, gr in zip(files, fams, groups):
        if gr == "unknown":
            log.warning("  gruppo non mappato: %-18s (famiglia '%s')", f, fam)

    # --- maschera a colori dei vuoti adipocitari, signed distance transform ---
    masks = [white_mask(os.path.join(IMG_DIR, f)) for f in files]
    for f, m in zip(files, masks):
        log.info("  %-18s frazione bianco(adipociti)=%.3f", f, m.mean())
    H = min(m.shape[0] for m in masks)
    W = min(m.shape[1] for m in masks)
    sdts = np.stack([signed_distance(m[:H, :W]) for m in masks]).astype(float)
    log.info("SDT calcolate, griglia comune %dx%d", H, W)

    # --- persistenza cubica (sublevel su +SDT = foreground TESSUTO) ---
    # cresce dal cuore del tessuto: H0 = componenti di tessuto, H1 = VUOTI
    # (buchi bianchi racchiusi dalle cellule; persistenza H1 ~ raggio del vuoto).
    cp = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=1)
    diagrams = cp.fit_transform(sdts)
    log.info("Diagrammi grezzi: %s", diagrams.shape)

    # potatura del rumore vicino alla diagonale: la SDT genera migliaia di
    # coppie a persistenza ~0; tolte rendono le distanze stabili (e tractabili).
    diagrams = Filtering(epsilon=args.epsilon).fit_transform(diagrams)
    real = (diagrams[..., 1] != diagrams[..., 0]).sum(axis=1)
    log.info("Dopo Filtering(eps=%.2f): %s | punti reali/img mediana %d max %d",
             args.epsilon, diagrams.shape, int(np.median(real)), int(real.max()))

    # --- descrittori vettoriali per replica ---
    pe = PersistenceEntropy().fit_transform(diagrams)        # (n, 2)
    amp = Amplitude(metric="wasserstein").fit_transform(diagrams)  # (n, 2)
    feat = np.hstack([pe, amp])

    # --- distanze replica x replica via PERSISTENCE LANDSCAPE ---
    # (vettorizzazione stabile: la Wasserstein a forza bruta su ~10^4 punti e'
    #  O(n^3) e impraticabile; il landscape e' la metrica scalabile canonica.)
    pd = PairwiseDistance(metric="landscape",
                          metric_params={"n_layers": 2, "n_bins": 100}, n_jobs=1)
    Dl = pd.fit_transform(diagrams)
    D = Dl.sum(axis=2) if Dl.ndim == 3 else Dl   # (n, n), H0+H1
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)

    labels = [f"{fam}" for fam in fams]
    np.savez(os.path.join(args.outdir, "full_features.npz"),
             files=np.array(files), families=np.array(fams),
             groups=np.array(groups), features=feat, distance=D)
    _save_distance_csv(D, files, os.path.join(args.outdir, "full_distance.csv"))

    # --- figure ---
    order = _dendrogram(D, fams, groups, args.outdir)
    _heatmap_ordered(D, fams, groups, order, args.outdir)
    _feature_scatter(feat, fams, groups, args.outdir)
    _betti_by_family(diagrams, fams, groups, args.outdir)

    # --- validazione ---
    _validate(D, fams, groups, args.outdir)
    log.info("Output in %s", args.outdir)


def _save_distance_csv(D, files, path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + list(files))
        for name, row in zip(files, D):
            w.writerow([name] + [f"{v:.4f}" for v in row])
    log.info("Salvato: %s", path)


def _dendrogram(D, fams, groups, outdir):
    Z = linkage(squareform(D, checks=False), method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    lbls = [f"{fam}" for fam in fams]
    dn = dendrogram(Z, labels=lbls, ax=ax, leaf_rotation=90, leaf_font_size=8)
    # colour the labels by group
    for tick, idx in zip(ax.get_xticklabels(), dn["leaves"]):
        tick.set_color(GROUP_COLOR.get(groups[idx], "black"))
    ax.set_title("Hierarchical clustering (Ward on persistence-landscape "
                 "distance) — blue = control, red = pathological")
    ax.set_ylabel("distance")
    fig.tight_layout()
    p = os.path.join(outdir, "full_dendrogram.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    log.info("Salvato: %s", p)
    return dn["leaves"]


def _heatmap_ordered(D, fams, groups, order, outdir):
    M = D[np.ix_(order, order)]
    labs = [fams[i] for i in order]
    cols = [GROUP_COLOR.get(groups[i], "black") for i in order]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(M, cmap="viridis")
    fig.colorbar(im, fraction=0.046)
    ax.set_xticks(range(len(labs)))
    ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_yticklabels(labs, fontsize=6)
    for t, c in zip(ax.get_xticklabels(), cols):
        t.set_color(c)
    for t, c in zip(ax.get_yticklabels(), cols):
        t.set_color(c)
    ax.set_title("Persistence-landscape distance, 36x36 (dendrogram order)")
    fig.tight_layout()
    p = os.path.join(outdir, "full_heatmap.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    log.info("Salvato: %s", p)


def _feature_scatter(feat, fams, groups, outdir):
    # feat = [entropia H0, entropia H1, ampiezza W H0, ampiezza W H1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for gr in ["control", "patho", "unknown"]:
        idx = [i for i, g in enumerate(groups) if g == gr]
        if not idx:
            continue
        c = GROUP_COLOR.get(gr, "gray")
        ax[0].scatter(feat[idx, 0], feat[idx, 1], c=c, label=gr, s=45,
                      edgecolor="k", linewidth=0.4)
        ax[1].scatter(feat[idx, 2], feat[idx, 3], c=c, label=gr, s=45,
                      edgecolor="k", linewidth=0.4)
    for i, fam in enumerate(fams):
        ax[0].annotate(fam, (feat[i, 0], feat[i, 1]), fontsize=5, alpha=0.7)
        ax[1].annotate(fam, (feat[i, 2], feat[i, 3]), fontsize=5, alpha=0.7)
    ax[0].set_xlabel("persistent entropy H0"); ax[0].set_ylabel("persistent entropy H1")
    ax[1].set_xlabel("Wasserstein amplitude H0"); ax[1].set_ylabel("Wasserstein amplitude H1")
    ax[0].set_title("Persistence descriptors — entropy")
    ax[1].set_title("Persistence descriptors — amplitude")
    for a in ax:
        a.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "full_feature_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    log.info("Salvato: %s", p)


def _betti_by_family(diagrams, fams, groups, outdir):
    bc = BettiCurve(n_bins=100)
    curves = bc.fit_transform(diagrams)        # (n, 2, n_bins)
    samp = bc.samplings_
    fam_set = sorted(set(fams))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab20")
    for k, fam in enumerate(fam_set):
        idx = [i for i, f in enumerate(fams) if f == fam]
        for dim in (0, 1):
            mean = curves[idx, dim].mean(axis=0)
            ax[dim].plot(samp[dim], mean, color=cmap(k % 20),
                         label=fam if dim == 0 else None, lw=1.3)
    ax[0].set_title("Mean Betti curve H0 per family (tissue components)")
    ax[1].set_title("Mean Betti curve H1 per family (adipocytic voids)")
    for a in ax:
        a.set_xlabel(r"filtration  $\alpha = +\mathrm{SDT}$")
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p = os.path.join(outdir, "full_betti_by_family.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    log.info("Salvato: %s", p)


def _validate(D, fams, groups, outdir):
    """Leave-one-replica-out: per ogni replica, la classifico col vicino piu'
    vicino (1-NN) sulla distanza di persistence landscape, escludendo se
    stessa, e verifico se ritrovo la famiglia e il gruppo."""
    n = len(fams)
    fam_hit = grp_hit = grp_total = 0
    rows = []
    for i in range(n):
        d = D[i].copy()
        d[i] = np.inf
        j = int(np.argmin(d))
        fok = fams[j] == fams[i]
        fam_hit += fok
        gok = None
        if groups[i] != "unknown" and groups[j] != "unknown":
            grp_total += 1
            gok = groups[j] == groups[i]
            grp_hit += gok
        rows.append((fams[i], fams[j], fok, gok, d[j]))

    fam_acc = fam_hit / n
    grp_acc = grp_hit / grp_total if grp_total else float("nan")
    log.info("VALIDAZIONE leave-one-replica-out (1-NN su persistence landscape):")
    log.info("  accuratezza FAMIGLIA : %d/%d = %.2f", fam_hit, n, fam_acc)
    log.info("  accuratezza GRUPPO   : %d/%d = %.2f", grp_hit, grp_total, grp_acc)

    import csv
    p = os.path.join(outdir, "full_validation.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["replica_family", "nn_family", "family_match",
                    "group_match", "nn_distance"])
        for r in rows:
            w.writerow([r[0], r[1], int(r[2]),
                        "" if r[3] is None else int(r[3]), f"{r[4]:.4f}"])
    log.info("Salvato: %s", p)
    with open(os.path.join(outdir, "full_validation_summary.txt"), "w") as fh:
        fh.write(f"family_accuracy {fam_hit}/{n} = {fam_acc:.3f}\n")
        fh.write(f"group_accuracy  {grp_hit}/{grp_total} = {grp_acc:.3f}\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TDA SDT+persistenza su tutto il dataset")
    p.add_argument("--smoke", type=int, default=None,
                   help="usa solo le prime N immagini (test rapido)")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="soglia di persistenza per il Filtering del rumore")
    p.add_argument("--outdir", default=OUT_DIR)
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stdout)
    import warnings
    warnings.filterwarnings("ignore")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
