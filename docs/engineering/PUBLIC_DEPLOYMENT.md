# Public application deployment record

Recorded: 20 November 2026. This records engineering acceptance, not new scientific validation.

**Application:** [ForgeSight AI on Google Cloud Run](https://forgesight-ai-6d5yatbeia-el.a.run.app/)

The source baseline was main `d68672e893fbe2a3934962c11f199d074079905b`, following merged PRs #1–#3. The owner deployed a separate private build context containing the original authorized encoder snapshot, frozen metadata, four reference banks, a React production build, and a small Cloud Run entrypoint. The Python API source in that context matches the source baseline. Private model assets and the artifact-bearing build context are not published with this source release.

## Recorded acceptance

The owner supplied these direct HTTPS results before the public source release:

| Check | Recorded result |
|---|---|
| Public homepage `/` | HTTP 200 |
| `/api/v1/health` | HTTP 200; CPU runtime; scientific gate open |
| `/api/v1/capabilities` | HTTP 200 |
| Public `/api/v1/ready` | HTTP 404, intentionally hidden |
| PCB1 synthetic inspection | HTTP 200; 5.13 s |
| PCB2 synthetic inspection | HTTP 200; 3.99 s |
| PCB3 synthetic inspection | HTTP 200; 3.28 s |
| PCB4 synthetic inspection | HTTP 200; 3.69 s |

All four inspection responses contained `inspection`, `patch_evidence`, and `visualizations`. These were genuine DINOv2 CPU inference requests on synthetic inputs. They establish execution across all categories, not benchmark accuracy. The timings are single sequential observations, not throughput guarantees or a controlled performance study.

The owner confirmed browser testing of the public application. During source-release preparation, a separate unauthenticated health request also returned HTTP 200 with `runtime_device: cpu` and `scientific_status: cross_platform_score_gate_open`. No new browser screenshot or browser acceptance is claimed by the release audit.

## Packaging and operational scope

The private entrypoint verifies original frozen banks at startup, mounts the built frontend at `/`, and hides the internal readiness route. The backend uses the same frozen inference code, scores, bank identities, and thresholds as the source baseline. Encoder initialization remains lazy until valid inspection.

The historical Nginx/systemd templates in `deployment/` are an alternative VPS plan. They are not the active Cloud Run ingress. In particular, their per-IP Nginx rate limits must not be claimed as installed on the public cloud service. This record does not establish load, abuse-resistance, worst-case resource, or broad security acceptance.

A fresh source clone can run the self-contained software tests. Genuine local inference additionally requires the exact authorized artifacts described in the [CPU runtime guide](RELEASE_CPU_RUNTIME.md). There is no public bank bundle, model package, or private research report attached to this release.

## Scientific acceptance remains separate

The [strict score reproduction gate](M314B_CPU_REPRODUCTION_NOTE.md) remains open: 3/4 selected historical cases met tolerance `0.00005`; all four review flags agreed. The 802-case evaluation is retrospective and non-independent. Heatmaps are model evidence, not defect segmentation, and the application supports human-supervised research review rather than certified manufacturing acceptance/rejection.
