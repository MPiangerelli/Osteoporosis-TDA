# Changelog

## 1.0 (2026-09-17)

First public version of the pipeline used in the paper.

- Fixed HSV segmentation of adipocytic voids (saturation < 0.15, value > 0.82), no white balance.
- Sublevel-set filtration on the signed distance transform (+SDT): H0 counts tissue components, H1 the enclosed voids.
- High-resolution acquisitions (WT 6m, WT 8m: 2088 x 1550 px over the same field of view as the other 508 x 376 px sections)
  are resampled to 508 x 377 px before segmentation, so that all images share one pixel scale (`OSTEO_TDA_RESCALE`).
- Group naming: WT 6m and WT 8m (formerly labelled WT Sinus8 and WT Sinus16 in earlier drafts).
- Reference outputs in `results_expected/`: control vs pathological 36/36, family 12/36, PERMANOVA pseudo-F 35.6,
  seven of eight descriptors significant at the family level.
- `diag_color.py` uses the same thresholds as the pipeline mask.
- `state_diagram.py` shows six representative families plus the WT 6m and WT 8m controls.
