#!/usr/bin/env python3
# diag_color.py -- verifica che il COLORE separi adipociti (bianchi, tondi =
# osteoporosi) dai vasi (rosa, allungati/striati). La conversione a grigi li
# confonde entrambi in "chiaro". Qui guardiamo saturazione/luminosita' e una
# maschera di BIANCO VERO (bassa S, alta V), poi la rotondita' delle componenti.
import os, sys, logging
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from full_persistence import IMG_DIR, OUT_DIR, WHITE_S_MAX, WHITE_V_MIN, load_rgb as _load_rgb

log = logging.getLogger("color")
FILES = ["WT 8m III.tif", "OVX 3m.tif", "RANKL_Tg_I.tif", "WT 3m.tif"]
OUT = os.path.join(OUT_DIR, "diag_color.png")   # Figure 4
S_THR, V_THR = WHITE_S_MAX, WHITE_V_MIN          # same thresholds as the paper mask


def load_rgb(path):
    return _load_rgb(path)


def roundness(mask):
    """circolarita' media (4*pi*area/perim^2) delle componenti grandi."""
    lbl, n = ndi.label(mask)
    if n == 0:
        return np.nan, 0
    vals = []
    for k in range(1, n + 1):
        comp = lbl == k
        area = comp.sum()
        if area < 30:
            continue
        per = comp.sum() - ndi.binary_erosion(comp).sum()  # ~perimetro
        if per > 0:
            vals.append(min(1.0, 4 * np.pi * area / (per * per)))
    return (np.mean(vals) if vals else np.nan), len(vals)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    import warnings; warnings.filterwarnings("ignore")
    rgbs = [load_rgb(os.path.join(IMG_DIR, f)) for f in FILES]
    H = min(r.shape[0] for r in rgbs); W = min(r.shape[1] for r in rgbs)
    rgbs = [r[:H, :W] for r in rgbs]

    fig, ax = plt.subplots(len(FILES), 4, figsize=(15, 3.5 * len(FILES)))
    log.info("%-18s %8s %8s %8s %8s", "file", "gray>thr", "white%", "round_w", "round_g")
    for i, (f, rgb) in enumerate(zip(FILES, rgbs)):
        hsv = mcolors.rgb_to_hsv(rgb)
        S, V = hsv[..., 1], hsv[..., 2]
        gray = rgb.mean(-1)
        # maschera vecchia (grigio) vs nuova (bianco vero)
        from full_persistence import otsu
        gmask = gray > otsu(gray)
        wmask = (S < S_THR) & (V > V_THR)
        rw, nw = roundness(wmask)
        rg, ng = roundness(gmask)
        log.info("%-18s %8.2f %8.2f %8.2f %8.2f", f, gmask.mean(), wmask.mean(),
                 rw, rg)

        ax[i, 0].imshow(rgb); ax[i, 0].set_ylabel(f, fontsize=9)
        ax[i, 0].set_title("RGB" if i == 0 else "", fontsize=9)
        im = ax[i, 1].imshow(S, cmap="magma")
        ax[i, 1].set_title("saturation (pink vessels = high)" if i == 0 else "",
                           fontsize=9)
        plt.colorbar(im, ax=ax[i, 1], fraction=0.046)
        ax[i, 2].imshow(gmask, cmap="gray")
        ax[i, 2].set_title("OLD mask: gray>Otsu\n(white+pink)" if i == 0 else
                           f"frac={gmask.mean():.2f}", fontsize=8)
        ax[i, 3].imshow(wmask, cmap="gray")
        ax[i, 3].set_title("NEW mask: TRUE WHITE\n(low S, high V)" if i == 0 else
                           f"frac={wmask.mean():.2f}", fontsize=8)
        for k in range(4):
            ax[i, k].set_xticks([]); ax[i, k].set_yticks([])
    fig.suptitle("Color separates white adipocytes (osteoporosis) from pink "
                 "vessels — grayscale conflates them", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120); plt.close(fig)
    log.info("Salvato: %s", OUT)


if __name__ == "__main__":
    main()
