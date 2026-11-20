# Clean-checkout Python testing and historical evidence

Status: public source-release verification, 20 November 2026.

## Verified fresh source checkout

The GitHub main commit verified by both source-CI workflows was:

`d68672e893fbe2a3934962c11f199d074079905b`

The Python workflow used Ubuntu 24.04, Python 3.12,
and the declared CPU inference requirements. An earlier
separate Linux cleanroom also verified source installation.

Research dependencies additionally verified:

- opencv-python-headless 4.14.0.94
- scikit-learn 1.9.1
- scipy 1.18.1

With those dependencies installed, 225 self-contained
Python tests passed without the original data, frozen banks,
or historical reports. The frontend workflow passed 68 Node
tests and the TypeScript/Vite production build using Node 24
and `npm ci`.

Workflow definitions:

- [Python source CI](../../.github/workflows/python-ci.yml)
- [Frontend CI](../../.github/workflows/frontend-ci.yml)

The source-CI and cleanroom tests exercise source-code contracts and
mocked service boundaries; they do not constitute real
inference or independent scientific evaluation.

The `research` extra supplies the additional source-CI dependencies.
Legacy ResNet scripts separately require `timm`, and historical
visual-atlas plotting requires `matplotlib` plus the original
research context. The CI result does not establish reproduction
of every historical script or experiment.

## Historical evidence tests

`tests/test_m312b3_evidence_review.py` contains six
additional integrity assertions that depend on original
ignored research evidence:

`reports/eda/m312b3_provisional_reviewer_worksheet_v1.csv`

SHA-256:
`1519b714b2a7dddc7088a34e8f8b1df7a49f14cdb5312ec0297476fa96e8949e`

`reports/eda/m312b3_evidence_summary_v1.json`

SHA-256:
`78431b505c586b4eabdb4e60d89e3cb2ad4c402976b0b9d09519851952f7b704`

All six passed in the original research environment.

These files are not included in the source repository.
A missing-file failure is not an algorithm regression.

The original worksheet contains 24 provisional research
cases with human review still pending. An integrity pass
must not be represented as human approval.

No replacement or synthetic worksheet is authorized.
Publishing those artifacts requires a separate provenance,
privacy, and redistribution review.

## Explicit CI boundary

GitHub Actions runs:

`python -m pytest -q --ignore=tests/test_m312b3_evidence_review.py`

This intentionally excludes only the six tests requiring
original local historical evidence.

The six assertions remain unchanged and executable when
the original verified files are available.

Do not describe source-only CI as a full frozen-inference
reproduction, benchmark rerun, or independent validation.

## Hosted checks and the scientific boundary

The separate private Cloud Run release context contains the
authorized original encoder and frozen banks; those artifacts
are not in this source repository. Genuine hosted CPU inference
was checked with synthetic inputs for all four PCB categories,
public HTTPS checks passed, and the project owner confirmed
browser testing of the [live application](https://forgesight-ai-6d5yatbeia-el.a.run.app/).

Hosted integration checks are separate from source CI. Synthetic
inputs demonstrate that the system executes; they do not measure
real-world anomaly-detection accuracy. The historical 802-case
evaluation is retrospective and non-independent.

The strict CPU cross-platform numerical score gate remains open:
3 of 4 audited cases met the original tolerance, and all four
review flags agreed. The system remains human supervised and
is not certified for autonomous manufacturing decisions.
