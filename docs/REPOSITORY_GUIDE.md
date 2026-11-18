# ForgeSight AI — Repository Guide

## Purpose

ForgeSight AI is an industrial visual anomaly-inspection research
and application engineering project using four PCB categories from
the VisA dataset.

This repository separates scientific evidence, experimental methods,
application software, and deployment engineering.

## Source Code

| Location | Responsibility |
|---|---|
| `src/forgesight/` | Python package and existing research modules |
| `tests/` | Unit tests and regression tests |
| `scripts/` | Reproducible research and engineering utilities |
| `pyproject.toml` | Package metadata and local dependencies |

## Scientific Research

| Location | Responsibility |
|---|---|
| `notebooks/` | Original Colab research notebooks |
| `splits/` | Dataset split specifications |
| `reports/` | Local research findings and experiment evidence |
| `docs/research/` | Research documentation and interpretation |
| `docs/DATA_PROVENANCE.md` | Existing data-provenance documentation |

Research artifacts are not automatically publishable merely because
their filenames appear in this project.

## Model Artifacts

The selected M3.12A frozen DINOv2-Small research configuration uses
four category-specific reference banks and a locked image-score
contract.

See `docs/engineering/FROZEN_M312A_ARTIFACT_REGISTRY.md`.

Large model assets remain outside Git until licensing, provenance,
and publication requirements are verified.

## Application Engineering

| Location | Planned Responsibility |
|---|---|
| `docs/engineering/` | Inference specifications, architecture, artifact contracts |
| `docs/deployment/` | Deployment and operations runbooks |
| `src/forgesight/inference/` | Future reproducible inference implementation |
| `src/forgesight/inspection/` | Future inspection and audit contracts |
| `web/` | Future application interface |

Planned directories are not claims of completed implementation.

## Scientific Boundaries

- The 802-image M3.12A benchmark was reused retrospectively.
- Four manually annotated M3.13 cases are diagnostic examples.
- M3.13 foreground and boundary scoring alternatives are not
  approved deployment models or calibrated inspection policies.
- Automated research scores must not be represented as certified
  manufacturing quality decisions.

## Preserved Baseline

Git tag: `m3.14a-pre-app-baseline-v1`

Commit: `4c78c67`

Prior to application engineering, 194 existing tests passed.
Six frozen M3.12A artifacts were verified using SHA-256.

The Git tag preserves committed source files only, not ignored
datasets, notebooks, model weights, or research reports.
