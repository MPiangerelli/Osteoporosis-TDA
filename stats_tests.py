#!/usr/bin/env python3
# =============================================================================
# stats_tests.py -- robustezza statistica dell'analisi (control vs pathological)
#
#  1) Mann-Whitney U su ogni descrittore scalare, a livello IMMAGINE (n=36) e
#     FAMIGLIA (n=12, repliche mediate -> niente pseudo-replicazione), con
#     effect size di Cliff (delta) e correzione FDR (Benjamini-Hochberg).
#  2) PERMANOVA (Anderson 2001) sulla matrice di distanze persistence-landscape:
#     pseudo-F e p da permutazione delle etichette di gruppo.
#  3) Test di permutazione sull'accuratezza leave-one-replica-out 1-NN, con
#     permutazione a livello di FAMIGLIA (rispetta la gerarchia 5 ctrl / 7 patho).
#
# Output: <results>/stats_report.txt , <results>/stats_boxplots.png  (Table 2, Figure 10)
# =============================================================================
import os, sys, logging
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gtda.homology import CubicalPersistence
from full_persistence import (white_mask, signed_distance, IMG_DIR, OUT_DIR,
                              discover, GROUP_COLOR)

log = logging.getLogger("stats")
RES = OUT_DIR
RNG = np.random.default_rng(0)
NPERM = 9999


# ---------------------------------------------------------------- descrittori
def descriptors():
    files, fams, groups = discover(None)
    masks = [white_mask(os.path.join(IMG_DIR, f)) for f in files]
    H = min(m.shape[0] for m in masks); W = min(m.shape[1] for m in masks)
    masks = [m[:H, :W] for m in masks]
    sdts = np.stack([signed_distance(m) for m in masks]).astype(float)
    diags = CubicalPersistence(homology_dimensions=(0, 1),   # +SDT: H0 tessuto, H1 vuoti
                               n_jobs=1).fit_transform(sdts)
    white_frac = np.array([m.mean() for m in masks])
    max_void = sdts.reshape(len(files), -1).max(axis=1)
    dom_tissue = np.array([(d[d[:, 2] == 0][:, 1] - d[d[:, 2] == 0][:, 0]).max()
                           for d in diags])
    dom_void = np.array([(d[d[:, 2] == 1][:, 1] - d[d[:, 2] == 1][:, 0]).max()
                         if (d[:, 2] == 1).any() else 0.0 for d in diags])
    D = np.load(os.path.join(RES, "full_features.npz"))
    feat = D["features"]      # [entropy H0 tissue, H1 void | amplitude H0 tissue, H1 void]
    desc = {
        "adipocyte fraction": white_frac,
        "largest-void radius (px)": max_void,
        "dominant void persistence (H1)": dom_void,
        "dominant tissue persistence (H0)": dom_tissue,
        "persistent entropy tissue (H0)": feat[:, 0],
        "persistent entropy void (H1)": feat[:, 1],
        "Wasserstein amplitude tissue (H0)": feat[:, 2],
        "Wasserstein amplitude void (H1)": feat[:, 3],
    }
    return files, np.array(fams), np.array(groups), desc


def cliffs_delta(x, y):
    x = np.asarray(x)[:, None]; y = np.asarray(y)[None, :]
    return (np.sign(x - y).sum()) / (x.size * y.size)


def bh_fdr(pvals):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p)
    ranked = np.empty(n); q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    ranked[order] = np.clip(q, 0, 1)
    return ranked


# ------------------------------------------------------------------ PERMANOVA
def permanova(D, labels, nperm=NPERM):
    labels = np.asarray(labels); n = len(labels)
    D2 = D ** 2
    def ss(lab):
        sst = D2[np.triu_indices(n, 1)].sum() / n
        ssw = 0.0
        for g in np.unique(lab):
            idx = np.where(lab == g)[0]
            if len(idx) > 1:
                ssw += D2[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].sum() / len(idx)
        a = len(np.unique(lab))
        ssa = sst - ssw
        return (ssa / (a - 1)) / (ssw / (n - a))
    F = ss(labels)
    perm = np.array([ss(RNG.permutation(labels)) for _ in range(nperm)])
    p = (np.sum(perm >= F) + 1) / (nperm + 1)
    return F, p


# ------------------------------------- leave-one-replica-out 1-NN + permutazione
def loo_group_acc(D, groups):
    n = len(groups); hit = tot = 0
    for i in range(n):
        d = D[i].copy(); d[i] = np.inf; j = int(np.argmin(d))
        if groups[i] != "unknown":
            tot += 1; hit += groups[j] == groups[i]
    return hit / tot


def perm_test_acc(D, fams, groups, nperm=NPERM):
    obs = loo_group_acc(D, groups)
    # mappa famiglia -> gruppo, poi permuto a livello di FAMIGLIA
    fam_list = list(dict.fromkeys(fams))
    fam_grp = {f: groups[list(fams).index(f)] for f in fam_list}
    base = np.array([fam_grp[f] for f in fam_list])
    null = []
    for _ in range(nperm):
        permmap = dict(zip(fam_list, RNG.permutation(base)))
        g2 = np.array([permmap[f] for f in fams])
        null.append(loo_group_acc(D, g2))
    null = np.array(null)
    p = (np.sum(null >= obs) + 1) / (nperm + 1)
    return obs, p, null


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    import warnings; warnings.filterwarnings("ignore")
    files, fams, groups, desc = descriptors()
    ctrl = groups == "control"; pat = groups == "patho"
    lines = ["STATISTICAL ROBUSTNESS — control (n=%d) vs pathological (n=%d)\n"
             % (ctrl.sum(), pat.sum())]

    # --- 1) Mann-Whitney, image-level e family-level ---
    names = list(desc)
    p_img, p_fam, rows = [], [], []
    for k in names:
        v = desc[k]
        # image level
        u, p = mannwhitneyu(v[ctrl], v[pat], alternative="two-sided")
        d = cliffs_delta(v[pat], v[ctrl])
        p_img.append(p)
        # family level (media per famiglia)
        fmean, fgrp = [], []
        for f in dict.fromkeys(fams):
            m = fams == f; fmean.append(v[m].mean()); fgrp.append(groups[m][0])
        fmean = np.array(fmean); fgrp = np.array(fgrp)
        uf, pf = mannwhitneyu(fmean[fgrp == "control"], fmean[fgrp == "patho"],
                              alternative="two-sided")
        p_fam.append(pf)
        rows.append((k, np.median(v[ctrl]), np.median(v[pat]), d, p, pf))
    q_img = bh_fdr(p_img); q_fam = bh_fdr(p_fam)

    lines.append("1) MANN-WHITNEY U  (median ctrl | median patho | Cliff d | "
                 "p_img | q_img(FDR) | p_fam | q_fam(FDR))")
    for (k, mc, mp, d, p, pf), qi, qf in zip(rows, q_img, q_fam):
        lines.append(f"   {k:<28} {mc:9.3f} | {mp:9.3f} | d={d:+.2f} | "
                     f"p={p:.1e} q={qi:.1e} | pfam={pf:.3f} q={qf:.3f}")

    # --- 2) PERMANOVA ---
    D = np.load(os.path.join(RES, "full_features.npz"))["distance"]
    F, pP = permanova(D, groups)
    lines.append(f"\n2) PERMANOVA on landscape distance (group factor): "
                 f"pseudo-F={F:.2f}, p={pP:.4f}  ({NPERM} perms)")

    # --- 3) permutation test on LOO accuracy ---
    acc, pa, null = perm_test_acc(D, fams, groups)
    lines.append(f"\n3) Leave-one-replica-out 1-NN group accuracy = {acc:.3f}; "
                 f"family-level permutation p={pa:.4f} "
                 f"(null mean={null.mean():.3f}, 95th pct={np.quantile(null,.95):.3f})")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RES, "stats_report.txt"), "w") as fh:
        fh.write(report + "\n")

    # --- boxplot dei 4 descrittori chiave ---
    key = ["adipocyte fraction", "largest-void radius (px)",
           "dominant void persistence (H1)", "Wasserstein amplitude void (H1)"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, k in zip(ax, key):
        v = desc[k]
        bp = a.boxplot([v[ctrl], v[pat]], labels=["control", "patho"],
                       patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], [GROUP_COLOR["control"], GROUP_COLOR["patho"]]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        for grp, col, xx in [(ctrl, GROUP_COLOR["control"], 1),
                             (pat, GROUP_COLOR["patho"], 2)]:
            a.scatter(np.full(grp.sum(), xx) + RNG.uniform(-.08, .08, grp.sum()),
                      v[grp], color=col, s=18, edgecolor="k", linewidth=0.3, zorder=3)
        pim = p_img[names.index(k)]
        a.set_title(f"{k}\nMann-Whitney p={pim:.1e}", fontsize=9)
    fig.suptitle("Key descriptors: control vs pathological", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "stats_boxplots.png"), dpi=140); plt.close(fig)
    log.info("Salvato: %s", os.path.join(RES, "stats_boxplots.png"))


if __name__ == "__main__":
    main()
