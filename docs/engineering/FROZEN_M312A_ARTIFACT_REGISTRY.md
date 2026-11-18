# ForgeSight AI — Frozen M3.12A Artifact Registry

**Status:** Recovered and SHA-256 verified locally, 19 November 2026.

**Research experiment:** `M3_12A_DINOv2S_672_8shot_v1`

**Evidence status:** Reused retrospective benchmark, not independent
external validation.

## Local Artifact Location

`models/frozen/m3_12a_dinov2s_672_v1/`

This directory is intentionally excluded from Git.

## Frozen Artifact Fingerprints

| File | SHA-256 |
|---|---|
| `m312_image_pixel_retro_lock_v1.json` | `2b4872f4115f3a683ad1d6e98934a9c6d6ce41bf0a63535e75704b861217940b` |
| `m312_image_pixel_retro_results_v1.json` | `0d33b11936c95241c503a318c447d6cad3105c5e0a93d2ec3e5068a8e9d4f35b` |
| `m312_pcb1_reference_bank_v1.npz` | `922ece4f2df36f33a59218f2953729dc19423decac3dff85b27af9b95064f411` |
| `m312_pcb2_reference_bank_v1.npz` | `f6e020f8fdc2ba036f5cd77568c529da5f8a3c8188949d9b881693e66b8a4617` |
| `m312_pcb3_reference_bank_v1.npz` | `910eac6c4d8bd807a6c8f1e62dc55fe57dd0153db9b937e938ea86ea0ec98a82` |
| `m312_pcb4_reference_bank_v1.npz` | `20b7e8e58e3a3f11651c9b7793797570c5b7bd6c32e5f9cb44494ee6760bc3fc` |

All six hashes passed verification against the original frozen
research contract.

## Frozen Image-Scoring Contract

- Backbone: DINOv2-Small
- Input resolution: 672 x 672
- Patch size: 14 x 14
- Reference images: 8 normal images per PCB category
- Similarity: Cosine nearest-neighbor embedding similarity
- Image score: Mean of the highest 1% of valid patch distances
- Four category-specific reference banks

## Frozen Research Thresholds

| PCB | Threshold |
|---|---:|
| PCB1 | 0.15704087913036346 |
| PCB2 | 0.14679642021656036 |
| PCB3 | 0.18991123139858246 |
| PCB4 | 0.23230013251304626 |

These thresholds belong to the original frozen scoring procedure.

They must not be reused with a changed scoring rule without a
separate, properly designed calibration and evaluation protocol.

## Model Weights

Original model-weight SHA-256:

`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`

The six-file artifact recovery does not itself establish that the
pretrained model weights have been recovered and independently
verified in the local deployment environment.

## Deployment Gate

Before declaring deployment ready:

1. Verify pretrained model weights and their provenance.
2. Verify exact preprocessing and scoring reproduction.
3. Verify inference on all four PCB categories.
4. Benchmark CPU and GPU memory and latency.
5. Audit applicable source, dataset, and model licenses.
6. Implement appropriate human-review and provenance contracts.

No M3.13 foreground eligibility modification is approved
for deployment.
