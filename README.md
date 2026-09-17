# Topological analysis of bone-marrow adiposity in murine osteoporosis

Code accompanying *Unveiling Osteoporosis and Bone Marrow Adiposity Through
Topological Data Analysis* (Piangerelli et al.). It reproduces the
computational Materials and Methods, Table 1, Table 2 and Figures 4-11 from
the 36 H&E micrographs (12 experimental groups x 3 replicates).

## Method in one paragraph

Each micrograph is segmented in HSV colour space: a pixel is an adipocytic
void when its saturation is below 0.15 and its value above 0.82 (true white;
pink eosinophilic vessels and stroma are excluded). The signed Euclidean
distance transform of the mask (positive inside voids, negative inside tissue)
is used as a sublevel-set filtration of the cubical complex of the image, so
that H0 counts tissue components and H1 counts the voids enclosed by tissue;
the persistence of a void is its radius. Diagrams are denoised (persistence
below 1 pixel discarded) and summarised by persistent entropy, Wasserstein
amplitude and persistence landscapes (2 layers, 100 bins). Samples are
compared by the L2 distance between landscapes (H0 + H1), clustered with Ward
linkage and validated with a leave-one-replicate-out nearest-neighbour rule.
Group differences are tested with Mann-Whitney U (image and family level,
Cliff's delta, Benjamini-Hochberg), PERMANOVA and a family-level permutation
test of the leave-one-out accuracy.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the 36 images in data/images/  (see data/images/README.md)
python run_all.py
```

Runtime: about 65 s on a laptop (Apple M-series, Python 3.9). Outputs go to
`results/`; the reference outputs used in the paper are in `results_expected/`
and must be reproduced exactly (`diff results/stats_report.txt
results_expected/stats_report.txt`).

The image folder and the output folder can also be set with the environment
variables `OSTEO_TDA_IMAGES` and `OSTEO_TDA_RESULTS`.

## What each script produces

| Script | Output | Paper |
|---|---|---|
| `full_persistence.py` | `full_features.npz`, `full_distance.csv`, `full_validation.csv`, `full_dendrogram.png`, `full_heatmap.png`, `full_feature_scatter.png` | Table 1, Figures 8, 9, 11 |
| `stats_tests.py` | `stats_report.txt`, `stats_boxplots.png` | Table 2, Figure 10 |
| `void_size_spectrum.py` | `void_size_spectrum.png` | Figure 5 |
| `diag_largescale.py` | `diag_largescale.png` | Figure 6 |
| `state_diagram.py` | `state_diagram.png` | Figure 7 |
| `diag_color.py` | `diag_color.png` | Figure 4 |

`run_all.py` runs them in this order.

## Expected numbers

| Quantity | Value |
|---|---|
| Leave-one-replicate-out, control vs pathological | 36/36 = 1.00 |
| Leave-one-replicate-out, experimental family (12 classes) | 12/36 = 0.33 |
| Largest-void radius, median control / pathological | 2.2 px / 32.8 px (14.6x) |
| Adipocyte area fraction, median control / pathological | 0.097 / 0.562 |
| Cliff's delta (adipocyte fraction, largest-void radius) | +1.00, family-level q = 0.005 |
| Dominant void persistence H1, Cliff's delta | +0.97, q = 0.005 |
| Wasserstein amplitude H1 (void), Cliff's delta | -0.95, q = 0.005 |
| Persistent entropy H0 / H1, Cliff's delta | +0.86 / -0.73, q = 0.012 |
| Wasserstein amplitude H0 (tissue), Cliff's delta | +0.68, q = 0.012 |
| PERMANOVA on landscape distances | pseudo-F = 35.6, p = 0.0001 (9,999 permutations) |
| Family-level permutation test of the accuracy | p = 0.0022 |

## Groups

Control: WT 3m, WT 6m, WT 8m, SO 3m (sham), RANKL Ctrl (WT littermates). Pathological: OVX 3m, Aged 1y, Aged 2y, p62-3m, p62-1y, p62-2y,
RANKL Tg. The mapping is the `GROUP` dictionary in `full_persistence.py`.

## Notes

* `OSTEO_TDA_MASK=hue` switches to a stain-robust segmentation (white balance
  plus a hue rule) developed for prospective screening of slides with a
  different stain cast. It is **not** the configuration of the paper and gives
  different numbers.
* Persistent homology is computed with giotto-tda (`CubicalPersistence`);
  use `n_jobs=1`, parallel workers can deadlock on macOS.
* The six WT 6m and WT 8m micrographs were acquired at 2088 x 1550 px over the same
  field of view as the ~508 x 376 px sections of the other groups; they are
  resampled to 508 x 377 px (`OSTEO_TDA_RESCALE`, on by default) so that all
  images share one pixel scale. All images are then cropped to the common grid
  (375 x 506 px).

## License

MIT. Please cite the paper if you use this code.
