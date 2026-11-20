# Clean-checkout Python testing and historical evidence

Status: November 2026, M3.21C.

## Verified fresh source checkout

The GitHub main commit tested was:

`67460701a7c896b906e617070bec45df83a65659`

The independent Linux cleanroom used Python 3.12 and
the declared CPU inference requirements.

Research dependencies additionally verified:

- opencv-python-headless 4.14.0.94
- scikit-learn 1.9.1
- scipy 1.18.1

With those dependencies installed, 225 self-contained
Python tests passed without the original data, frozen banks,
or historical reports.

The cleanroom tests exercise source-code contracts and
mocked service boundaries; they do not constitute real
inference or independent scientific evaluation.

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

## Remaining release gates

- Verify the new Python CI on a fresh GitHub Actions runner
- Perform genuine inference on the target Linux server
- Verify the exact original frozen banks on that server
- Retain the open 3-of-4 numerical reproduction disclosure
- Test public HTTPS routing and CPU-service safeguards
