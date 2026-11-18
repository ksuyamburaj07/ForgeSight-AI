# ForgeSight AI — M3.11B CPU Student–Teacher Training

Research stage: normal-only model training, **not** defect-validated.

This package adds exactly three files to an existing Project_05 checkout:

- `src/forgesight/student_protocol.py`
- `scripts/m3_11_student_train_v1.py`
- `tests/test_student_protocol.py`

It does **not** replace any M0–M3.11A model, code, or research artifact.

## Experiment contract

- Frozen teacher: existing pinned ResNet-18 model (verified by existing project loader).
- Student: existing `CompactStudent` (117,752 trainable parameters).
- Initial run: `pcb1` only; subsequent categories use separate files.
- Samples: 256 normal training + 32 disjoint normal monitoring examples per PCB, deterministically selected from the existing frozen *training* partition.
- Excluded: prior PaDiM normal-fit images, ResNet reference-memory images, and normal calibration images. The existing frozen normal-validation and VisA official-test images are not used.
- Training: CPU mini-batches of 8, AdamW, 10 epochs, two-scale normalized feature matching. The checkpoint is selected by normal monitoring loss. Teacher weights remain frozen.
- New checkpoint: `models/m3_11_stfpm_lite_pcb1_v1.pt` (weight tensors only).
- New report: `reports/eda/m3_11_stfpm_lite_pcb1_train_v1.json`.
- Training does not establish anomaly recall, false alarms, or production readiness.

## Install in existing checkout

```bash
cd /data/Projects/Project_05
source .venv/bin/activate
unzip -n ~/Downloads/ForgeSight_M3_11B_Student_Training_v1.zip
python -m pytest -q -W error::DeprecationWarning
```

The six new tests give an expected total of **163** if the previous 157 are retained and the environment remains compatible.

## Run one category first

```bash
OPENBLAS_NUM_THREADS=2 python scripts/m3_11_student_train_v1.py --category pcb1
```

Do **not** rerun after successfully saving the checkpoint and report. The code refuses to overwrite existing outputs. Do not run the official holdout.

Once PCB1 report has been reviewed, later training can use `--category pcb2`, `--category pcb3`, or `--category pcb4`, with distinct artifacts for each category.

## Review and scope

The existing `m3_11_student_training_smoke.py` supplies the already-tested teacher feature extraction function, and the existing `m3_9_resnet_feature_probe.py` supplies the frozen data loader and preprocessing. The script fingerprints its own code and the student/protocol source in the result report.

Built and checked with Python syntax compilation; the six new tests were exercised against the compact student and passed in a separate CPU test environment. **The full Project_05 integration and fitting cannot be claimed as tested here**; run the full suite in your checkout before training.
