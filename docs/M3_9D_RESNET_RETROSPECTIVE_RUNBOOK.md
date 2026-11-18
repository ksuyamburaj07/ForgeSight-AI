# ForgeSight AI — M3.9D ResNet retrospective evaluation

## Scientific status

The 802 official VisA test images have **already been evaluated** in M3.8. This is a **retrospective, test-informed comparison**, not a fresh independent holdout. Do not report model selection based on M3.9D as if it had been independently validated. ResNet feature bank and category thresholds are fixed from M3.9B/C. No threshold retuning on test labels.

## Installation (Linux repo `/data/Projects/Project_05`)

The archive contains paths prefixed by `Project_05/`. Extract from `/data/Projects` and do not overwrite historical files:

```bash
cd /data/Projects
unzip -n ~/Downloads/ForgeSight_AI_M3_9D_ResNet_Retrospective.zip
cd Project_05
source .venv/bin/activate
python -m pytest -q -W error::DeprecationWarning
```

Expect 130 passed if previous 124 tests remain unchanged (six additional tests).

## Step 1 — Implemented-code and evidence preflight

```bash
python scripts/m3_9_resnet_retro.py --preflight
```

This checks known SHA-256 fingerprints of the original benchmark, the frozen 256-vector ResNet coreset, and the M3.9C normal-only calibration. It also checks publisher metadata/path inventory and records source-file hashes in `reports/m3_9_resnet_retro_lock_v1.json`. No test image contents are decoded during this step, although the M3.8 test set has been unsealed historically.

## Step 2 — Retrospective evaluation

```bash
OPENBLAS_NUM_THREADS=2 python scripts/m3_9_resnet_retro.py --evaluate
```

It computes real ResNet image scores and original-resolution novelty maps for the 802 official VisA examples, comparing with publisher masks. It reuses the exact M3.8 helper metrics:
- Exact image AUROC and noninterpolated average precision
- TP/FP/TN/FN at previously calibrated M3.9C thresholds
- Pixel AUROC and AP as **4,096-bin approximations** on `[0, 2]`
- Observed CPU processing times

Outputs `reports/m3_9_resnet_retro_results_v1.json` and SHA-256, without modifying or rerunning M3.8 and without overwriting completed results.

## Review

Share the printed report output and JSON. We will compare ResNet's image AUROC, AP, recall, specificity, false-positive count, pixel AP and latency against the M3.8 reference, accounting for already-used test-set limitations.

## If a check fails

Do not delete the original M3.8 report or silently edit the new lock. Share the full traceback. A corrected evaluation implementation should receive a new version label and a clear changelog.
