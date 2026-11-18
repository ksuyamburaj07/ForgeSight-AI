# ForgeSight AI — M3.8 Official VisA holdout evaluation

## Purpose
This package adds a **one-time**, **two-stage** evaluation protocol to the existing M3.7 frozen repository. It does not edit previous model banks, normal calibration thresholds, or original dataset files.

## Preconditions
- Linux project root `/data/Projects/Project_05`, active `.venv`.
- Existing source code through M3.7 and `99` passing tests.
- VisA official split metadata `data/metadata/visa/1cls.csv` (SHA-256 `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`).
- Local data extracted for pcb1–pcb4, including official test images and anomalous masks.
- Frozen protocol `reports/m3_final_evaluation_protocol_v1.json` (SHA-256 `a80efd6a6cb25f7bd21526e054d40212ac69caa52918ad1017d1b365d04eecd4`).
- The two separate model banks from M3.3 and M3.4, plus the cached DINOv2-small checkpoint.

## Commands
Execute from `/data/Projects/Project_05`:

```bash
source .venv/bin/activate
python -m pytest -q -W error::DeprecationWarning
python scripts/m3_final_holdout.py --preflight
OPENBLAS_NUM_THREADS=2 python scripts/m3_final_holdout.py --evaluate
```

`--preflight` validates the CSV, required file presence, and frozen model fingerprints but **does not decode the official test images or masks**. It locks the exact evaluation source code and chosen metric implementation under `reports/m3_metric_implementation_v1.json`. It refuses to overwrite an existing lock.

`--evaluate` checks the lock, opens the official test images, scores both fixed candidates, and writes `reports/m3_official_holdout_results_v1.json` only after all 802 examples are processed. The result file cannot be overwritten silently. Only progress counts, not intermediate performance scores, are printed while processing.

## Metrics
- Image AUROC and average precision: exact, noninterpolated, ties grouped. Score: frozen mean of the five highest patch cosine distances.
- TP/FP/TN/FN, recall, specificity, false-positive rate: frozen category-specific thresholds, strictly-greater comparison.
- Pixel AUROC and average precision: **approximate**, computed from 4096 fixed uniform bins spanning cosine distance `[0,2]` with pooled pixels per category. No learned pixel threshold and no binary segmentation IoU.
- Full-frame maps: interpolate valid patch scores to 224x224, crop letterbox content, reproject to original dimensions.
- Tiled maps: interpolate each tile map to original tile coordinates, average overlapping predictions.
- Anomaly annotations: any nonzero mask value is positive; normal photographs get all-zero masks.

The official test has not been evaluated before this run. Once executed, this is no longer an untouched holdout. Do not change thresholds, model choice, tiling or postprocessing based on the result and then publish another score on the same test as if it were independent.

## Limitations
Benchmark is not a safety/production certification. Physical unit independence is unknown. The pixel metrics use a fixed-bin approximation; visible artifacts may arise from padding, interpolation, tile seams, or normal background variation. Image-level and pixel-level metrics are different tasks.
