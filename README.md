# ForgeSight AI

### Industrial Visual Quality Inspection & Anomaly Localization

**An evidence-driven computer vision research and engineering project for visual anomaly inspection of printed circuit boards.**

ForgeSight AI investigates industrial visual inspection using reproducible computer vision experiments, frozen deep-learning representations, nearest-neighbor anomaly scoring, controlled evaluation, and human-supervised inspection principles.

The project now includes a working local CPU inference service, camera-enabled React inspection studio, interactive research explorers, and preserved retrospective evidence. Public deployment remains a separate, unverified release milestone.

> **Current status — November 2026**
>
> Real four-category CPU inference, the FastAPI service, and the React Inspection Studio are implemented and tested locally. The public application and its runtime artifact delivery have not yet been deployed or independently verified.
>
> This is not a certified manufacturing inspection system.

---

## Project at a glance

| Component | Status |
|---|---|
| Image validation and metadata extraction | Implemented |
| Four-category VisA PCB research | Completed research stages |
| DINOv2-Small frozen research scorer | Implemented in research and local CPU inference; cross-platform numerical gate open |
| Category-specific frozen reference banks | Recovered and SHA-256 verified |
| Retrospective image-level evaluation | Completed |
| Spatial failure and boundary diagnostics | Completed |
| Automated regression tests | 227 Python and 67 frontend tests passed on November 20, 2026 |
| Unified local inference engine | Implemented for all four PCB categories |
| Human-supervised inspection API | Implemented locally with FastAPI |
| Interactive industrial dashboard | Implemented in React, including camera/video inputs and research explorers |
| Public deployment | Not yet available |

## 1. Research motivation

Industrial visual quality inspection must handle unusual components, subtle surface differences, complex board geometry, variable backgrounds, and uncertainty about whether unfamiliar visual evidence indicates an actual defect.

A useful research detector must do more than produce visually compelling heatmaps.

ForgeSight emphasizes:

- Reproducible image preprocessing and model scoring
- Clear separation of training, validation and evaluation
- Frozen model configurations and reference artifacts
- Explicit false-alarm and missed-anomaly analysis
- Evidence preservation and independent integrity checks
- Transparent distinctions between anomaly detection and physical defect diagnosis
- Human review for consequential inspection decisions

Our goal is a credible, open engineering system, not an unsupported claim of factory-grade accuracy.

### Research journey — why the direction changed

ForgeSight AI began with an engineering question: could we
build a credible, end-to-end industrial computer-vision
inspection system using genuinely accessible research data?

The intention was to learn how visual inspection works beyond
a model notebook: dataset auditing, representations, calibration,
false alarms, missed anomalies, artifact integrity, inference,
human review, software tests, and a real application.

We used Amazon Science VisA's four PCB categories and
investigated **six evaluated configurations** across multiple
method families. These are historical experiments, not six
independent external validation studies.

| Configuration | Macro image AUROC | Anomalies flagged / 400 | Normal images flagged / 402 |
|---|---:|---:|---:|
| DINOv2 full-frame | 0.8001 | 114 | 24 |
| DINOv2 tiled | **0.9018** | 185 | 15 |
| ResNet coreset / PatchCore-style | 0.8253 | 152 | 20 |
| PaDiM-style statistical model | 0.8053 | 160 | 34 |
| Student–teacher / STFPM-lite | 0.7664 | 139 | 27 |
| High-resolution DINOv2-Small, 672 px | 0.8927 | **255** | 59 |

The early tiled DINOv2 configuration retained the highest
reported macro image AUROC. The later high-resolution
configuration flagged 70 more anomaly-labelled images at its
own frozen category thresholds, but also flagged 44 additional
normal-labelled images.

**That is a tradeoff, not a universal victory.** AUROC measures
image ranking across thresholds, whereas the counts above
describe distinct, previously frozen operating thresholds.
The models were not compared at a common false-review rate.

Our experiments repeatedly exposed missed anomalies and normal
images flagged for review. The follow-up spatial investigations
examined four selected cases, including background-associated
evidence and PCB boundary sensitivity. They produced useful
descriptive diagnostics, not a causal explanation of every
failure or an independent validation of a proposed remedy.

The engineering direction therefore changed. Rather than
present a model as an autonomous pass/fail authority, ForgeSight
became an **evidence-first, human-supervised research inspection
studio**. Its interface exposes source-aligned views, raw patch
distances, selected measurements, spatial visualizations,
provenance, and explicit unresolved limitations.

The active 672-pixel DINOv2-Small configuration is the
implemented inspection workflow, **not a claim that it is
statistically or operationally superior to every alternative**.

Scientific boundaries:

- The 802-image cohort contains 400 anomaly-labelled and
  402 normal-labelled VisA images and was reused in later
  retrospective experiments.
- The original four-PCB inventory contains 4,416 images:
  3,614 normal-training, 402 evaluation-normal, and
  400 evaluation-anomaly.
- The active score is the mean of the top 1% of eligible
  48×48 patch-grid cosine nearest-neighbour distances.
- The CPU cross-platform numerical score gate remains open:
  three of four audited cases met the stated tolerance;
  one narrowly did not. All four review flags were unchanged.
- Heatmaps are interpolated model evidence, not pixel-level
  defect segmentation. The 3D surface depicts relative
  distances, not physical PCB height.
- Camera and video frames may differ markedly from the
  VisA benchmark domain. Strong scores do not confirm
  physical defects.
- The system does not authorize autonomous manufacturing
  acceptance or rejection.

The public research aggregates are available in
[`model-registry.json`](frontend/public/benchmark/model-registry.json)
and
[`paired-outcomes.json`](frontend/public/benchmark/paired-outcomes.json).
Their source references preserve the identities of earlier
research records; publication of aggregates alone does not
redistribute or independently reproduce every original
research input.

## 2. Dataset and research scope

The research uses the publicly documented **VisA (Visual Anomaly)** dataset, originally released by Amazon Science.

Publisher and dataset references:

- https://github.com/amazon-science/spot-diff
- https://registry.opendata.aws/visa/

Four PCB categories are included in the project's research scope:

`pcb1` · `pcb2` · `pcb3` · `pcb4`

Original images, masks, archives and intermediate research assets are stored separately from version-controlled source code.

The right to use a dataset for research does not automatically imply that every related artifact can be redistributed without additional checks.

## 3. Research pipeline

The selected frozen high-resolution research configuration follows this conceptual sequence:

```text
Original PCB image
       |
       v
Verified image decoding / EXIF handling
       |
       v
Deterministic 672 x 672 letterbox preprocessing
       |
       v
Frozen DINOv2-Small feature extraction
       |
       v
Valid patch embeddings (48 x 48 patch grid)
       |
       v
Category-specific normal reference bank
       |
       v
Cosine nearest-neighbor patch distances
       |
       v
Mean of highest 1% valid patch distances
       |
       v
Frozen research score and threshold comparison
       |
       v
Human interpretation and research audit
```

Each PCB category has its own reference bank constructed from eight normal reference images.

The research visualization pipeline can project patch distances back onto the original image. Visualization smoothing is not part of the frozen raw image-scoring operation.

The local service implements this inspection contract. The separate CPU cross-platform numerical score reproduction gate remains open, so this must not be represented as fully closed reproducibility.

## 4. Frozen research results

### M3.12A — High-resolution DINOv2-Small

The frozen retrospective image-level evaluation covered **802 reused VisA PCB images** across the four selected categories.

Recorded results:

| Metric | M3.12A retrospective result |
|---|---:|
| Image-level AUROC | Approximately 0.8927 |
| True positives | 255 |
| False negatives | 145 |
| False positives | 59 |
| True negatives | 343 |
| Total evaluated images | 802 |

**Interpretation:** These results characterize the frozen experiment on a reused retrospective benchmark. They are not an independent external validation result or evidence of certified manufacturing performance.

Scores, category thresholds, reference-bank fingerprints and reproducibility details are documented in:

[The M3.12A Frozen Artifact Registry](docs/engineering/FROZEN_M312A_ARTIFACT_REGISTRY.md)

### M3.13 — Spatial failure investigation

Four purposively selected retrospective cases were independently annotated and investigated using the frozen DINOv2 score pipeline.

The investigation included:

- Reproduction of all four original frozen scores
- Verification of manually annotated PCB and background regions
- Complete valid-patch distance analysis
- Alternative spatial eligibility diagnostics
- Top-k sensitivity analysis
- Original-image boundary visualization
- Controlled mask-registration stress testing
- Export and SHA-256 integrity audits

The investigation identified background-associated high-scoring patches in some cases and strong board-boundary sensitivity in another.

**Scientific limitation:** These four cases cannot establish causation or demonstrate population-level accuracy improvements.

No foreground-only scoring modification has been approved for deployment.

## 5. Software and inspection experiences

ForgeSight provides two distinct Python entry points.

### Validation-only CLI

The original `forgesight inspect` CLI validates input
images, checks metadata and returns structured results.

Its `inference_performed: false` field is intentional.
This CLI must not be confused with the neural-network
inspection service.

### Genuine CPU inference service

The FastAPI application in
[`src/forgesight/api/app.py`](src/forgesight/api/app.py)
implements genuine local PCB anomaly inspection.

The `/api/v1/inspect` endpoint supports the four VisA PCB
categories and uses the verified DINOv2-Small encoder,
frozen category-specific reference banks, 672-pixel
preprocessing and cosine nearest-neighbour scoring.

The service returns image-level anomaly evidence, raw
patch measurements, original-image visualizations,
source SHA-256 and provenance.

A flagged score requests human review. It does not
establish a confirmed physical defect.

### React Inspection Studio

The application in [`frontend/`](frontend/) includes:

- Upload inspection for PCB1 through PCB4
- Browser camera capture and photograph review
- Guided video and paced, serial frame sampling
- Linked original, distance-map and overlay views
- Interactive 2D/3D patch-distance visualizations
- Six-configuration retrospective Model Explorer
- Audited Dataset Explorer with 16 source photographs
- Four-case historical Case Atlas
- Engineering records and unresolved scientific gates

The application has been exercised locally with genuine
CPU inference. Public deployment has not yet been verified.

## 6. Local development

Requirements:

- Python 3.11 or newer
- A compatible Python virtual environment
- Local disk space for optional research artifacts

Install the existing Python package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the test suite:

```bash
python -m pytest -q
```

Try the existing image-validation CLI:

```bash
python -m forgesight inspect /absolute/path/to/image.png
```

This command does not perform neural-network inference.

The minimal installation above supports the
validation-oriented CLI and core tests. It does not
provision the frozen neural-network runtime.

For genuine CPU inference, a separate optional dependency
group is declared:

```bash
python -m pip install -e '.[dev,service,inference]'
```

For CPU-only deployments, install a compatible PyTorch CPU
wheel using the official platform-specific guidance before
installing the other package dependencies. A clean-install
test remains pending.

**Installing dependencies is not sufficient to run inference.**
The service additionally requires the exact frozen DINOv2
snapshot and the four original SHA-256-verified reference
banks, which are intentionally excluded from this repository.

See the
[CPU runtime and artifact guide](docs/engineering/RELEASE_CPU_RUNTIME.md)
before running or deploying the service.


## 7. Repository organization

| Path | Responsibility |
|---|---|
| `src/forgesight/` | Python package and existing research modules |
| `tests/` | Unit and regression tests |
| `scripts/` | Reproducible research utilities |
| `splits/` | Dataset development protocols |
| `notebooks/` | Local original experiment notebooks |
| `models/` | Local model artifacts and frozen banks |
| `reports/` | Local experiment outputs and evidence |
| `docs/` | Architecture, scientific and engineering documentation |
| `data/` | Local dataset files and archives |

Large or unreviewed research artifacts are intentionally excluded from the current Git source baseline.

See the [Repository Guide](docs/REPOSITORY_GUIDE.md) for the complete organization plan.

## 8. Engineering status and release gates

### Implemented and locally verified

- [x] Audited the four VisA PCB categories
- [x] Evaluated six historical model configurations
- [x] Frozen the high-resolution DINOv2 research method
- [x] Investigated selected false alarms and missed anomalies
- [x] Built the four-category CPU inference service
- [x] Built camera and paced-video inspection
- [x] Added linked 2D/3D evidence visualization
- [x] Added model, dataset and historical-case explorers
- [x] Passed 67 frontend tests at the M3.20A checkpoint
- [x] Passed 227 Python tests at the November 20 release-prep checkpoint

### Open scientific and publication requirements

- [ ] Resolve or retain the remaining numerical reproduction
      discrepancy with explicit disclosure
- [ ] Verify all Python inference dependencies in a clean environment
- [ ] Document frozen model and reference-bank provisioning
- [ ] Complete repository history, licensing and security audits
- [ ] Deploy the real FastAPI CPU inference service
- [ ] Verify the public HTTPS application end to end
- [ ] Complete responsive, keyboard and camera acceptance testing

The reused 802-image retrospective evaluation does not
constitute independent external validation. This is a
human-supervised research system, not a certified autonomous
manufacturing inspection product.

## 9. Responsible interpretation

ForgeSight produces or investigates visual anomaly evidence. An anomaly score is not, by itself, a physical defect diagnosis.

Research acceptance thresholds must not be described as manufacturing approval decisions without appropriate qualification, calibration and operational validation.

The project does not use Tesla proprietary data, does not represent Tesla research, and has not been qualified for any production inspection line.

## 10. Reproducibility and provenance

The source baseline is preserved in Git:

- Baseline commit: `4c78c67`
- Baseline tag: `m3.14a-pre-app-baseline-v1`
- Earlier M3.14 baseline: 194 Python tests; current Python release suite pending rerun. Frontend: 67 passing tests at M3.20A.
- Frozen M3.12A artifacts: six SHA-256 verification checks passed

The tag preserves committed source code, not ignored datasets, notebooks, pretrained weights, reference banks or output reports.

Additional documentation:

- [Repository Guide](docs/REPOSITORY_GUIDE.md)
- [Frozen Artifact Registry](docs/engineering/FROZEN_M312A_ARTIFACT_REGISTRY.md)
- [Original Architecture Decision Record — M0](docs/ARCHITECTURE.md)
- [Original Dataset Provenance Record — M0](docs/DATA_PROVENANCE.md)

The M0 records are retained as historical documents and may contain statements that were correct at the time but have since been superseded by later research.

## 11. Licensing

ForgeSight's original source code is distributed under the [MIT License](LICENSE).

Dataset licensing, pretrained model-weight rights, third-party dependencies and any redistribution obligations must be checked independently before releasing a bundled application or research artifact.

## Project philosophy

**Reproducible evidence before polished predictions.**

The intended outcome is an industrial inspection system whose behavior can be inspected, tested, explained and audited—not merely demonstrated.
