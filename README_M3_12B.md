# ForgeSight AI — M3.12B-1 Paired Failure Analysis (CPU-only)

**Status:** descriptive forensic examination of two previously frozen DINOv2 experiments, not independent model validation. This package does NOT train, load model weights, decode images, alter frozen thresholds, or evaluate a new image dataset.

## Why this follows M3.12A

M3.12A (672px, DINOv2-S, eight normal references) caught 255/400 defects and falsely flagged 59/402 normals. Earlier M3.8 tiled DINOv2 caught 185/400 defects and falsely flagged 15/402 normals. Its macro image AUROC (0.9018) remains slightly higher than M3.12A (0.8927). A paired audit is more informative than comparing these aggregates: it reveals where decisions overlap and which images both missed.

## Input evidence / immutable SHA256 contracts

- Earlier M3.8: `reports/m3_official_holdout_results_v1.json` SHA `ea0354d971d52ad1ef1d3adea9eb9fdd36849f19318bf266eb4e97ac32f11f72`
- M3.12A: `reports/m312_image_pixel_retro_results_v1.json` SHA `0d33b11936c95241c503a318c447d6cad3105c5e0a93d2ec3e5068a8e9d4f35b`

The script fails closed if source file hashes disagree, images are missing, labels conflict, duplicates exist, or derived confusion counts differ from source-reported values.

## Files

- `scripts/m3_12b_paired_failure_audit_v1.py`: deterministic stdlib Python audit. Does not modify inputs.
- `tests/test_m3_12b_paired_failure_audit_v1.py`: five unit tests.
- `reports/eda/m3_12b_paired_failure_audit_v1.json`: computed paired outcome counts.
- `reports/eda/m3_12b_paired_case_manifest_v1.csv`: 802 rows, one per image. Contains only image paths, ground-truth labels, existing scores and flags; *not* image pixels.

The reports in this ZIP were independently computed from the two verified uploaded JSON files and may be kept without rerunning. If rerunning, use a new temporary output directory or leave the included files in place; the script refuses to overwrite artifacts.

## Run locally (optional verification)

Use the Dell's local Python (no PyTorch needed):

```bash
cd /data/Projects/Project_05
python -m unittest discover -s tests -p 'test_m3_12b_paired_failure_audit_v1.py' -v
python scripts/m3_12b_paired_failure_audit_v1.py \
  --tiled reports/m3_official_holdout_results_v1.json \
  --m312a reports/m312_image_pixel_retro_results_v1.json \
  --outdir /tmp/forgesight_m312b_verification
```

If the local M3.8 report lives at another path, supply that existing path. Put the downloaded M3.12A report in `reports/` using the canonical filename (the uploaded file was called `Copy of ...`; the **bytes**, not its filename, are what the SHA checks).

## Findings on the 802 matched paths

| Category | Both catch defect | M3.12A only catch | M3.8 tiled only catch | Both miss defect | Both false alarm | M3.12A only false alarm | Tiled only false alarm | Both correctly normal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCB1 | 46 | 49 | 1 | 4 | 0 | 31 | 0 | 69 |
| PCB2 | 48 | 15 | 3 | 34 | 7 | 10 | 1 | 82 |
| PCB3 | 29 | 21 | 9 | 41 | 1 | 4 | 1 | 95 |
| PCB4 | 37 | 10 | 12 | 41 | 2 | 4 | 3 | 92 |
| TOTAL | **160** | **95** | **25** | **120** | **10** | **49** | **5** | **338** |

## Forensics, not test tuning

1. Inspect examples missed by both, particularly PCB2/3/4. Use the official defect masks to distinguish subtle board defects from high response on the background. These cases are deliberately selected from *already revealed* benchmark labels.
2. Inspect M3.12A-only false alarms, particularly PCB1. The predeclared first-case heatmaps visibly show some response outside the board on normal images, making background/geometry sensitivity worth testing. This remains a hypothesis, not a proven cause.
3. Do **not** select new thresholds, reference sets, ensemble weights, foreground masks, or feature rules using these 802 labeled image outcomes and then claim an independent improvement. Any method improvement would require a new development cohort and separately sequestered assessment data.
4. Full CPU/GPU deployment speed, lighting shifts, domain diversity, and independently acquired PCB images remain unmeasured.

**No new GPU Colab notebook is required for this paired-score audit.** Keep M3.12A as the single original GPU experiment notebook, and this separate CPU-only analysis as versioned Project 05 evidence.
