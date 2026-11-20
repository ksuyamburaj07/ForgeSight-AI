# Public source-release audit

Date: 20 November 2026. Audited baseline: `d68672e893fbe2a3934962c11f199d074079905b` on main, following merged PRs #1–#3.

**Finding:** no material publication blocker was found in the reviewed source/history and GitHub surfaces. This is a scoped release audit, not a guarantee that no vulnerability or undetected secret exists.

## Git and media scope

- Fresh authenticated remote clone; nonshallow history. All advertised branches and PR heads resolved within the reviewed commit set. No remote tags existed before this release.
- 23 reachable commits, 242 historical paths, and 261 unique blobs (6,960,721 uncompressed blob bytes).
- Secret-pattern review of reachable blob and commit/tag content: no credential matches found. Reviewed long encoded payloads rather than assuming text-only files were safe.
- No tracked original `data/`, `models/`, or `reports/` files in any reviewed commit; no serialized model weights/banks, dataset archives, private report documents, notebooks, or Git LFS pointers found.
- Exactly 20 tracked image binaries: 16 gallery JPEGs and four historical atlas triptychs. Source/derivative hashes match the retained manifests and locally retained source/export files. No copied EXIF or embedded image text metadata was identified.
- One compressed 24-case selection manifest in a historical script was decoded and reviewed. It contains scientific paths, labels, scores, thresholds, and provenance metadata, not images, weights, or credentials.

## Intentionally public scientific derivatives

The release includes source-path/label/hash split protocols, selected benchmark aggregates, four-case atlas records and patch-distance fields, and the curated media above. The 16 gallery photos are normal-training examples, not confirmed frozen-bank members. Atlas cases are reused retrospective diagnostics.

These derivatives are explicitly documented; the source is not described as containing no research evidence. Original frozen datasets, model binaries/banks, metadata report files, evidence worksheets, research reports/notebooks, and the artifact-bearing cloud build context remain separately held and were not added or modified by release preparation.

## Rights and licenses

Original software remains MIT licensed. The [third-party notices](../../THIRD_PARTY_NOTICES.md), gallery attribution, and atlas attribution retain the VisA publisher, copyright, CC BY 4.0 link, research citation, and descriptions of modifications. Upstream DINOv2-Small is identified as Apache 2.0; its binaries are not redistributed in this source release. Dependencies retain their own licenses.

Primary records checked on the audit date: [VisA registry](https://registry.opendata.aws/visa/), [upstream dataset license](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET), [upstream NOTICE](https://github.com/amazon-science/spot-diff/blob/main/NOTICE), and [DINOv2 model card](https://huggingface.co/facebook/dinov2-small).

## GitHub surfaces

Before the visibility change:

- Three merged pull requests; no issue comments, inline PR comments, submitted PR reviews, or commit comments. PR bodies and commit metadata were pattern-scanned; no credential matches or attachment URLs found.
- All eight available completed workflow-run logs were downloaded and pattern-scanned; no credential matches found.
- No existing GitHub releases/assets or Actions artifacts.
- Wiki and Discussions were disabled.

## Software verification and boundaries

At the audited baseline, both GitHub Actions workflows succeeded: [225 self-contained Python tests](https://github.com/ksuyamburaj07/ForgeSight-AI/actions/runs/44666228142) and [68 frontend tests plus TypeScript/production build](https://github.com/ksuyamburaj07/ForgeSight-AI/actions/runs/44666228260). The source install includes checked CPU wheels and `pip check`. Six original historical evidence tests were verified separately in the local research environment and are excluded from source CI.

Release preparation changes presentation, notices, guidance, and package metadata. It does not alter inference code, thresholds, model/reference-bank bytes, public scientific assets, or scientific acceptance criteria. Subsequent CI status is visible through the README badges and Actions history.

Hosted four-category synthetic-input execution and owner browser acceptance are documented separately in the [deployment record](PUBLIC_DEPLOYMENT.md). Neither closes the 3/4 numerical reproduction gate nor converts the 802-case retrospective evaluation into independent validation. The system remains human supervised and noncertified.
