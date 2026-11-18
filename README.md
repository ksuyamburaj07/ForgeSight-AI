# ForgeSight AI

### Industrial Visual Quality Inspection & Anomaly Localization

**An evidence-driven computer vision research and engineering project for visual anomaly inspection of printed circuit boards.**

ForgeSight AI investigates industrial visual inspection using reproducible computer vision experiments, frozen deep-learning representations, nearest-neighbor anomaly scoring, controlled evaluation, and human-supervised inspection principles.

The current project has completed multiple research milestones and is transitioning into a deployable inspection application.

> **Current status — November 2026**
>
> The research foundation and experimental DINOv2 inference workflow exist. The local command-line package currently performs real input validation, not model inference. A unified four-PCB inference service and public web application are under development.
>
> This is not a certified manufacturing inspection system.

---

## Project at a glance

| Component | Status |
|---|---|
| Image validation and metadata extraction | Implemented |
| Four-category VisA PCB research | Completed research stages |
| DINOv2-Small frozen research scorer | Implemented in Colab research |
| Category-specific frozen reference banks | Recovered and SHA-256 verified |
| Retrospective image-level evaluation | Completed |
| Spatial failure and boundary diagnostics | Completed |
| Source-code regression tests | 194 passing at M3.14 baseline |
| Unified local inference engine | Planned |
| Human-supervised inspection API | Planned |
| Interactive industrial dashboard | Planned |
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

The future application must reproduce this contract before exposing model-generated results to users.

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

## 5. Existing software

ForgeSight already contains a Python package with independently tested modules covering image processing, experimental representations, geometric analysis, data protocols, evaluation, and several anomaly-detection research methods.

The currently exposed CLI provides an image-validation workflow.

It can:

- Decode supported JPEG, PNG and WebP inputs
- Validate size and image dimensions
- Handle EXIF orientation
- Report source format, dimensions and SHA-256
- Return structured JSON and explicit input errors

**It does not yet execute the frozen DINOv2 detector.**

Its existing results correctly indicate:

`inference_performed: false`

This behavior must remain explicit until real model inference is implemented and tested.

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

Research notebooks and frozen model assets have additional runtime and dependency requirements. They are not installed by the minimal package configuration above.

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

## 8. Engineering roadmap

### Completed foundation

- [x] Implement deterministic input-validation boundaries
- [x] Develop PCB research dataset protocols
- [x] Explore multiple visual anomaly-detection approaches
- [x] Freeze and evaluate the M3.12A DINOv2 research configuration
- [x] Complete M3.13 spatial and boundary diagnostics
- [x] Verify the six essential frozen M3.12A artifacts
- [x] Establish a local, version-controlled source baseline

### Application development

- [ ] Implement one reproducible inference interface for PCB1–PCB4
- [ ] Verify inference scores against the frozen reference pipeline
- [ ] Benchmark CPU and GPU runtime behavior
- [ ] Build a human-supervised inspection service
- [ ] Implement traceable inspection and model-version records
- [ ] Integrate original-image, novelty-map and overlay views
- [ ] Build the industrial web dashboard
- [ ] Audit deployment licensing and packaging
- [ ] Publish and validate a functioning public application

The roadmap describes planned work, not already implemented capabilities.

## 9. Responsible interpretation

ForgeSight produces or investigates visual anomaly evidence. An anomaly score is not, by itself, a physical defect diagnosis.

Research acceptance thresholds must not be described as manufacturing approval decisions without appropriate qualification, calibration and operational validation.

The project does not use Tesla proprietary data, does not represent Tesla research, and has not been qualified for any production inspection line.

## 10. Reproducibility and provenance

The source baseline is preserved in Git:

- Baseline commit: `4c78c67`
- Baseline tag: `m3.14a-pre-app-baseline-v1`
- Baseline verification: 194 passing existing tests
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
