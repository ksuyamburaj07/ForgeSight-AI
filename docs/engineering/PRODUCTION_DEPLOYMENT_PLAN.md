
# ForgeSight AI — Production deployment plan

Status: uninstalled deployment templates, November 2026.

Historical VPS option: the active application now runs on Google Cloud Run.
See the [public deployment record](PUBLIC_DEPLOYMENT.md) and
[current runtime guide](RELEASE_CPU_RUNTIME.md). The Nginx limits and systemd
topology below were not installed on Cloud Run and must not be presented
as protections verified on the live service. The remaining gates below
belong to this proposed VPS deployment.

These files define the proposed deployment topology.
They are not proof of a successful cloud installation.

## Architecture

One public HTTPS domain:

- `/`: React production build served by Nginx
- `/api/v1/health`: lightweight process liveness
- `/api/v1/capabilities`: inspection contract
- `/api/v1/inspect`: rate-limited CPU inference
- `/api/v1/ready`: blocked at public Nginx ingress

FastAPI listens on 127.0.0.1:8000, with one worker.
No raw uploaded images are intentionally persisted by
the application endpoint.

## Proposed initial machine

- Linux x86-64
- 2 vCPUs
- 4 GiB RAM
- At least 40 GB disk
- Dedicated unprivileged `forgesight` service user

This is a conservative starting configuration, not a
verified production capacity or minimum requirement.

## Measured local benchmark

Machine: Intel Core i3-1005G1, Linux, CPU inference.

One deterministic synthetic 672x672 JPEG was submitted
sequentially through the actual FastAPI inspection endpoint.

| Request | Seconds |
|---|---:|
| PCB1 cold | 6.965 |
| PCB1 warm | 2.251 |
| PCB2 | 2.237 |
| PCB3 | 2.897 |
| PCB4 | 2.707 |

All five requests returned HTTP 200 and evidence payloads.

Highest measured Python process RSS: 747.9 MiB.
This is not a system-wide or worst-case cloud memory figure.

## Required application layout

Proposed checkout:

`/srv/forgesight/app/`

Frontend build:

`/srv/forgesight/app/frontend/dist/`

Frozen artifacts:

`/srv/forgesight/app/models/frozen/m3_12a_dinov2s_672_v1/`

Hugging Face cache for service user's home:

`/var/lib/forgesight/.cache/huggingface/hub/`

The existing artifact resolver assumes a source-checkout
layout. Use an editable Python installation until packaging
and path portability have been independently verified.

The exact frozen weights, two metadata files, and four
reference banks must pass the existing SHA-256 checks.

The frozen banks are not downloaded from GitHub.
No public artifact-distribution location is approved yet.

## Nginx files

`deployment/nginx/forgesight-limits.conf` must be loaded
in the `http` context, typically through a conf.d include.

`deployment/nginx/forgesight.conf.template` is a
server/site template, not an immediately activatable file.

Replace `__FORGESIGHT_DOMAIN__` consistently and provision
valid certificates before activating the TLS site.

Certificate issuance may require a temporary HTTP-only
ACME bootstrap configuration. Do not activate the final
TLS configuration until certificate files exist.

Run `sudo nginx -t` before each reload on the server.

The proxy's 9 MiB inspection body ceiling is distinct
from FastAPI's 8 MiB uploaded-file limit. Both must
remain enforced.

The per-IP request policy is initially 10 requests
per minute, with a burst of 3, and 2 concurrent
connections per IP. These are tunable operational
settings, not validated production throughput.

Do not add a CDN or another reverse proxy without
revalidating client-IP identification and rate limiting.

## Systemd runtime

Install the supplied unit only after creating the
dedicated service user and provisioning the checkout,
Python environment, frozen banks, and encoder cache.

The systemd service intentionally:

- Binds Uvicorn to loopback only
- Uses one inference worker
- Uses an unprivileged account
- Disables model network fetching
- Restricts filesystem writes
- Restarts on process failure

Check that the service user can read the model cache,
bank files, source checkout, and installed packages.

The model cache is tied to the service user's HOME.
Changing HF_HOME alone does not relocate the existing
ForgeSight encoder's Path.home()-based lookup.

## Readiness and verification

On the VPS, after starting Uvicorn, call:

`curl --fail-with-body http://127.0.0.1:8000/api/v1/ready`

A successful response requires model initialization
and loading all four original reference banks.

Public HTTPS `/api/v1/ready` should return 404.

Check public `/api/v1/health` and
`/api/v1/capabilities`, followed by genuine uploads to
PCB1, PCB2, PCB3, and PCB4.

Verify upload rejection, rate limiting, concurrency
behavior, front-end camera permission and mobile UX.

## Security and scientific restrictions

This application is for human-supervised research review,
not autonomous manufacturing acceptance or rejection.

The CPU cross-platform numerical reproduction gate is
still open (3 of 4 historical cases passed tolerance).

Do not silently change the encoder, bank data,
preprocessing, or frozen category thresholds.

Do not expose the service's TCP port directly to the
internet. Do not put credentials into Git or the
public frontend.

Nginx limits are not a substitute for a firewall,
monitoring, patching, and cloud budget controls.

## Remaining acceptance gates

1. Clean CPU installation of declared dependencies
2. Verified secure provisioning of exact frozen artifacts
3. Nginx syntax test on the actual target server
4. HTTPS certificate issuance and renewal verification
5. Real readiness and four-PCB public smoke test
6. Rejection and rate-limit tests through public ingress
7. Privacy, licensing, cost, and rollback review
