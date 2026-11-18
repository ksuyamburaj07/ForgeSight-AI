# ForgeSight AI — M3.14C-2 Four-PCB Visual Audit

Date: 19 November 2026
Status: Visual export verified; research limitations retained.

## Scope

Four reused M3.13 diagnostic cases were processed through
the Dell CPU inference engine. Three-panel visualizations
showed original image, interpolated patch distances and
source-aligned overlay.

This is not a new independent benchmark, defect-segmentation
evaluation or manufacturing-validation exercise.

## Findings

| Case | Category | Label | Review flag | Observation |
|---|---|---|---|---|
| B3-01 | PCB1 | Anomaly | No | Dispersed PCB and background responses |
| B3-09 | PCB2 | Anomaly | No | Strong right-board response, below image threshold |
| B3-17 | PCB3 | Normal | Yes | Dominant upper-left background diagonal |
| B3-23 | PCB4 | Normal | Yes | Dominant response near upper board boundary |

## Engineering interpretation

- All four source-image identities passed SHA-256 checks.
- Original-size heatmaps and overlays were generated.
- Visible alignment appears plausible in the four triptychs.
- Interpolated heatmaps are not pixel-level defect masks.
- Per-image color scaling prevents absolute cross-image
  interpretation of the color intensities.
- Background and PCB boundary sensitivity remain important
  research limitations.
- Four frozen review flags were reproduced.
- Strict CPU-versus-Colab numerical reproduction remains
  3/4 within the original 5e-5 tolerance.

## Interface requirements

1. Display the input image, heatmap and overlay.
2. Expose the raw 48x48 patch map as research evidence.
3. Disclose the per-image visualization color scale.
4. Show score and frozen threshold without recalibration.
5. Label threshold exceedance as a research review flag.
6. Do not claim causal localization or defect segmentation.
7. Make background/boundary limitations accessible.
8. Preserve model, bank and runtime provenance.

## Next phase

Implement a validated upload inference service, followed
by the interactive four-category inspection application.

No change to the frozen research protocol is authorized.
