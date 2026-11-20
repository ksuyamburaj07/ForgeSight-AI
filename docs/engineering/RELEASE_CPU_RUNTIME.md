# ForgeSight AI — CPU Runtime and Frozen Artifacts

Status: public application and source-release documentation, 20 November 2026.

The [live application](https://forgesight-ai-6d5yatbeia-el.a.run.app/)
runs genuine DINOv2 CPU inference on Google Cloud Run. Hosted
synthetic-input checks succeeded for PCB1 through PCB4, and the
project owner confirmed browser testing. These checks establish
working inference and delivery, not real-world defect accuracy.

The deployment uses a separate, private release context containing
the authorized frozen runtime artifacts. Its Python source matches
the released source baseline. The artifact-bearing deployment
context, model binaries and original banks are not included in this
repository. A source clone alone cannot reproduce the hosted service.

## Observed environment

| Package | Observed version |
|---|---|
| Python | 3.12.3, Linux x86_64 |
| PyTorch | 2.15.0+cpu |
| TorchVision | 0.29.2+cpu |
| Transformers | 4.57.6 |
| NumPy | 2.5.4 |
| Safetensors | 0.8.0 |
| Hugging Face Hub | 0.36.2 |
| Tokenizers | 0.22.2 |
| Pillow | 12.3.0 |
| FastAPI | 0.165.0 |
| Uvicorn | 0.54.0 |
| python-multipart | 0.0.34 |
| Pydantic | 2.14.3 |
| httpx | 0.28.1 |
| pytest | 9.2.0 |

At main commit `d68672e893fbe2a3934962c11f199d074079905b`, GitHub
Actions passed 225 self-contained Python tests, 68 frontend tests,
and the TypeScript production build. Six additional historical
evidence integrity tests passed separately in the original local
research environment. See the [test policy](CLEAN_CHECKOUT_TEST_POLICY.md).
None of these checks closes the numerical cross-platform
reproduction discrepancy or independently validates model accuracy.

## Install the source and CPU requirements

The verified source-CI setup uses Python 3.12 on Linux x86_64.
From a fresh repository clone:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.15.0+cpu' 'torchvision==0.29.2+cpu'
python -m pip install -e '.[dev,service,inference,research]'
python -m pip check
python -m pytest -q --ignore=tests/test_m312b3_evidence_review.py
```

The `research` extra covers the additional dependencies used by
source CI and current image-processing/evaluation utilities. Legacy
ResNet experiments also require `timm` (observed locally: 1.0.31);
the historical visual-atlas plotting script requires `matplotlib`.
Those historical workflows also depend on their original research
inputs and notebook state. Installing the source-CI extras is not a
claim that every historical experiment has been cleanly reproduced.

## Original DINOv2-Small encoder

Source repository:

https://huggingface.co/facebook/dinov2-small

Frozen research revision:

`150c8e7bb7cef2d30ec31b13a517af14840ee3f7`

Expected `model.safetensors` SHA-256:

`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`

The upstream model repository identifies its license as
Apache License 2.0. Its license and attribution are separate
from ForgeSight's own MIT software license.

The existing encoder resolves the standard local Hugging Face
cache and requires an exact weight SHA-256 match before loading.

To provision the original upstream revision in a clean
environment, after installing compatible dependencies:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/dinov2-small",
    revision=(
        "150c8e7bb7cef2d30ec31b13a517af14840ee3f7"
    ),
    allow_patterns=[
        "config.json",
        "model.safetensors",
    ],
)
```

This command contacts Hugging Face and populates its default
cache. It does not need to be run on a machine where the
original verified snapshot is already present.

A successful download alone is NOT sufficient: the existing
ForgeSight loader must still validate the exact frozen weight
hash and required model configuration.

Do not silently substitute the latest DINOv2 checkpoint.

## Four frozen reference banks

Required artifact directory:

`models/frozen/m3_12a_dinov2s_672_v1/`

Required metadata:

- `m312_image_pixel_retro_lock_v1.json`
- `m312_image_pixel_retro_results_v1.json`

Required reference-bank files:

- `m312_pcb1_reference_bank_v1.npz`
- `m312_pcb2_reference_bank_v1.npz`
- `m312_pcb3_reference_bank_v1.npz`
- `m312_pcb4_reference_bank_v1.npz`

Expected reference-vector counts:

| Category | Reference vectors |
|---|---:|
| PCB1 | 13,824 |
| PCB2 | 13,824 |
| PCB3 | 11,520 |
| PCB4 | 14,976 |

The bank archive must contain a float32 `vectors` array
with 384 feature dimensions. Original bank bytes and
normalization must not be changed.

Expected metadata SHA-256:

Lock:
`2b4872f4115f3a683ad1d6e98934a9c6d6ce41bf0a63535e75704b861217940b`

Report:
`0d33b11936c95241c503a318c447d6cad3105c5e0a93d2ec3e5068a8e9d4f35b`

Individual bank hashes are contained in the frozen lock and
checked by the existing loader.

These files are NOT included in the public source baseline.
There is currently no verified public provisioning location for
the four original banks. A fresh clone cannot perform genuine
inference until these exact artifacts have been supplied through
an authorized private provisioning process.

Do not manufacture substitute banks or rebuild them from the
16 public training-gallery photographs.

## Verify the local frozen artifacts

Run from the project root after installing the package:

```bash
python - <<'PYCODE'
from forgesight.inference.encoder import find_verified_snapshot
from forgesight.inference.artifacts import (
    verify_frozen_metadata,
    load_frozen_bank,
)

snapshot = find_verified_snapshot()
lock = verify_frozen_metadata()

print("Verified encoder snapshot:", snapshot.name)
print("Frozen resolution:", lock["resolution"])

for pcb in ("pcb1", "pcb2", "pcb3", "pcb4"):
    bank = load_frozen_bank(pcb)
    print(pcb, tuple(bank.shape))

print("Frozen artifact verification passed.")
PYCODE
```

The verification is read-only. It does not run inference or
prove that GPU/CPU image scores are numerically identical.

## Scientific and production limitations

- The CPU/GPU score reproduction gate remains open: 3 of
  4 audited cases met the frozen numerical tolerance.
- Evaluation uses reused retrospective VisA images, not
  independent external validation.
- An anomaly flag is a request for human review.
- Camera/video inputs can be outside the dataset domain.
- Heatmaps are interpolated feature evidence, not defect masks.
- The spatial surface does not reconstruct physical PCB height.
- This is not a certified production quality-control system.

## Hosted verification and reproducibility boundary

- Source installation and the declared CPU requirements passed on
  a fresh GitHub Actions runner; source CI does not load real banks.
- The separate Cloud Run release uses the original verified encoder
  and all four frozen category banks. Real synthetic-input requests
  returned inspection records, patch evidence and visualizations.
- Public HTTPS, health and capabilities checks passed. The hosted
  deployment hides `/api/v1/ready` and returns HTTP 404 there; the
  source API retains a local readiness endpoint for operators.
- The frontend build requires same-origin `/api` routing to FastAPI.
  The live deployment provides this through its private packaging.
- A publicly downloadable bank bundle and a source-only recreation
  of the artifact-bearing Cloud Run context are not provided.

The existing numerical gate remains open: 3 of 4 audited scores
met the original tolerance, while all four review flags agreed.
No thresholds, bank bytes or scientific acceptance criteria were
changed to obtain the hosted deployment.
