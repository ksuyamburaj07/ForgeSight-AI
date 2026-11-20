# Contributing

Please open an issue describing the problem, evidence, and proposed scope before a substantial change. Small documentation fixes can go directly to a pull request.

Use the [README installation](README.md#installation) and run the source Python suite, frontend tests, and production build for changes affecting those areas. Historical evidence tests require separately authorized local inputs; do not upload those inputs to make a test pass.

Keep research changes explicit: describe the data split, calibration, evaluation independence, failure cases, and any altered preprocessing or scoring rule. A new method must not silently replace frozen thresholds, reference banks, hashes, or acceptance tolerances. Preserve the open numerical reproduction disclosure.

Commit source, tests, and appropriately attributed public documentation. Do not commit credentials, original datasets, model weights/banks, private research reports, downloaded archives, or unreviewed media. Existing curated public derivatives are listed in [third-party notices](THIRD_PARTY_NOTICES.md).

Make human-review semantics clear. An anomaly flag is not a confirmed physical defect or manufacturing acceptance/rejection authorization.
