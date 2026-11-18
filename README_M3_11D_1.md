# ForgeSight AI — M3.11D-1 Student–teacher image retrospective

## Purpose
Evaluate the *already-fitted and normal-calibrated* PCB1–PCB4 student–teacher detectors on the same 802 VisA test photographs previously examined in M3.8/M3.9/M3.10. **This is a repeated retrospective development benchmark, NOT an independent final test.** No model or decision-threshold changes are allowed here.

## Included
- `src/forgesight/student_retro_contract.py`: pure-data checks for path/label/count alignment with the paired PaDiM benchmark and a normal-score regression guard.
- `scripts/m3_11_student_image_retro_v1.py`: metadata-only preflight lock, then separate image-only evaluation.
- `tests/test_student_retro_contract.py`: six tests for manifest/score integrity.

## Execution (Project_05 root)

```bash
cd /data/Projects/Project_05
source .venv/bin/activate
unzip -n ~/Downloads/ForgeSight_M3_11D_1_Student_Image_Retrospective_v1.zip
python -m pytest -q -W error::DeprecationWarning
python scripts/m3_11_student_image_retro_v1.py --preflight
OPENBLAS_NUM_THREADS=2 python scripts/m3_11_student_image_retro_v1.py --evaluate
```

Expected suite count is **177** if the previous 171 are intact; the user must verify this locally. Preflight reads official CSV metadata and verifies file existence, **without opening test image contents or test masks**. It also verifies student/teacher provenance and calibrations. Evaluation decodes only test *images* (not annotation masks), checks first normal validation score for each student against the frozen M3.11C-1 report before any test-image decoding, and records exact image AUROC/AP, confusion matrices and CPU timing.

Output files (new, never overwrite):
- `reports/m3_11_student_image_retro_lock_v1.json`
- `reports/m3_11_student_image_retro_results_v1.json`

## Important limitations
The same official VisA photographs have been assessed repeatedly and are not an untouched final holdout. Category decision thresholds were calibrated from 24 normal examples each; the 12 normal development validation examples per PCB had been inspected previously. Do not optimize selection of model/checkpoint, loss, score map, threshold, or fusion based on these retrospective anomaly labels while presenting outcomes as independent test performance. M3.11D-2 will measure pixel localization separately using predeclared binning/projection and source-mask population checks.

Do not rerun a completed evaluation; both stages protect their outputs against overwrites. If `--preflight` fails, do not manually change old artifacts or hashes. Report the traceback and reconcile the schema.
