# ForgeSight AI — M3.11C-1: Student–teacher normal-only calibration

This package implements a two-scale STFPM-lite feature-discrepancy map and calibrates category-specific decision thresholds using previously reserved **normal** development images. It **does not** load any official VisA test images, measure defect recall, retrain models or modify historical artifacts.

## Exact installation

From `/data/Projects/Project_05` with `.venv` activated:

```bash
unzip -n ~/Downloads/ForgeSight_M3_11C_Student_Normal_Calibration_v1.zip
python -m pytest -q -W error::DeprecationWarning
```

Expected tests: **171 passed** (163 existing plus eight added), assuming the original suite is unchanged. Treat a test failure as a stop condition; do not create a lock.

## Two distinct run commands

```bash
python scripts/m3_11_student_normal_pilot_v1.py --preflight
OPENBLAS_NUM_THREADS=2 python scripts/m3_11_student_normal_pilot_v1.py --evaluate
```

The first command validates report/checkpoint SHA-256 hashes, metadata consistency, exact category cohort separation and source-code hashes **without decoding images**. Then it creates an immutable implementation lock.

The second uses the unchanged ResNet-18 teacher and four separately trained students, each on 24 training-partition normal calibration images and 12 reserved (previously inspected) normal validation images. It creates a separately named result report; thresholds are based strictly on normal calibration images, never defect labels.

Outputs:

- `reports/m3_11_student_normal_pilot_lock_v1.json`
- `reports/eda/m3_11_student_normal_pilot_v1.json`

Both are created only if absent. Never delete or overwrite frozen evidence to fix an issue; if the implementation needs changes after preflight, preserve the original and version the new implementation.

## Method

Teacher and student each produce 128×28×28 and 256×14×14 features. Per-scale unit-normalized channelwise squared difference is computed, the coarse map is bilinearly upsampled to 28×28, and both scales are averaged. Zeroed letterbox padding is excluded. The image score is the mean of the five highest valid spatial discrepancies. Thresholds use the same 95%-normal quantile implementation used in the prior ResNet and PaDiM pilots.

These scores are on their **own scale** and must never inherit thresholds from ResNet coreset or PaDiM. Their per-image inference timing measures the teacher/student forward passes and anomaly-map computation, **excluding image loading/preprocessing**.

## Scientific interpretation

- Training: 256 normal examples per category, with 32 disjoint normal monitoring images (already completed M3.11B).
- Calibration: 24 other training-partition normal images per category, previously reserved from M3.9C.
- Development validation: 12 normal images per category, **previously inspected** in earlier projects. These are not an independent final validation.
- Official VisA test images: **not accessed**.
- No defect recall, pixel localization AP/AUROC, or real-world manufacturing safety claim is supported by M3.11C-1.

A later separately locked retrospective experiment can evaluate image detection and pixel localization, but must report reused-benchmark status.
