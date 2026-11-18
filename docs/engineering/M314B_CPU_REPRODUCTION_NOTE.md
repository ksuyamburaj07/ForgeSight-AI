# M3.14B — CPU Cross-Platform Score Reproduction

Date: 19 November 2026

Status: OPEN — STRICT FOUR-CASE SCORE GATE NOT PASSED

## Scope

This is a diagnostic comparison against previously frozen M3.12A
research scores and reused M3.13 cases. It is not an independent
benchmark, recalibration study, or new scientific evaluation.

## Verified identities

- Four original input-image SHA-256 fingerprints: passed.
- Frozen model-weight SHA-256: passed.
- Frozen metadata and four category banks: verified.
- Actual DINOv2 input resolution: 672 × 672.
- Output patch grid: 48 × 48 with 384-dimensional features.

## Original Dell SDPA reproduction

Absolute acceptance tolerance: 0.00005.

| Case | Frozen score | Dell SDPA score | Absolute delta | Gate |
|---|---:|---:|---:|---|
| B3-01 | 0.1453762800 | 0.1453241557 | 0.0000521243 | FAIL |
| B3-09 | 0.1426969320 | 0.1427421272 | 0.0000451952 | PASS |
| B3-17 | 0.2334015220 | 0.2334035933 | 0.0000020713 | PASS |
| B3-23 | 0.2991513014 | 0.2991603315 | 0.0000090301 | PASS |

Strict numerical reproduction: 3/4 cases.
Frozen review-flag agreement: 4/4 cases.

## B3-01 patch-level diagnostic

- GPU reference evidence SHA-256 verified.
- Valid patches: 1,728.
- Top-1% count: 18.
- CPU/GPU top-18 patch overlap: 18/18.
- Mean absolute patch-distance difference: 0.0000182308.
- Maximum absolute patch-distance difference: 0.0012263656.

## Controlled CPU attention comparison

| Backend | Score | Absolute frozen-score delta | One-run elapsed |
|---|---:|---:|---:|
| SDPA | 0.1453241557 | 0.0000521243 | 3.304 s |
| Eager | 0.1453240961 | 0.0000521839 | 6.970 s |

Neither backend met the original tolerance. The two CPU
scores are nearly identical, so selecting eager attention
does not resolve this mismatch.

The exact cause of CPU-versus-GPU numerical differences
has not been isolated.

## Engineering decision

Use the existing CPU SDPA runtime as the current application
implementation, without changing frozen scores, reference
banks, thresholds, or numerical acceptance criteria.

This is a documented cross-platform reproduction exception.
No claim of complete numerical reproduction is authorized.

All displayed anomaly flags are research review indicators,
not manufacturing rejection decisions.

Real inference functionality can be developed and tested
while the strict reproduction issue remains open.
