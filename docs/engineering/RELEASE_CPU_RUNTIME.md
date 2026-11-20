# ForgeSight AI — CPU Runtime and Frozen Artifacts

Status: release candidate documentation, 20 November 2026.

This guide describes the locally tested inference environment.
It is not evidence of a clean-install reproduction or a public
deployment.

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

The Python suite passed 227 tests and the frontend suite
passed 67 tests on this environment. These do not establish
clean-environment compatibility or close the numerical
cross-platform reproduction discrepancy.

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
inference until these exact artifacts have been supplied.

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

## Deployment requirements still pending

- Clean Linux CPU installation and dependency-resolution test
- Verified and authorized provisioning of frozen banks
- Runtime memory and inference-latency measurement
- HTTPS routing for frontend and FastAPI under one origin
- Upload limits, concurrency, health checks and error handling
- Public end-to-end acceptance test across PCB1 through PCB4
