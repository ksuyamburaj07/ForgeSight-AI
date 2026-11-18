# ForgeSight AI — M0 starter

An open industrial visual-inspection engineering project. The goal is a functional, portable, auditable inspection application based on independently licensed data, code and weights. **This starter is not an anomaly detector or a deployed application.**

## What works right now

A command-line input-validation component that:
- accepts an existing local JPEG, PNG, or WebP image by inspecting the file contents (not trusting its extension);
- enforces a maximum input size of 16 MiB and 20 million pixels before full decoding;
- fully decodes the image, applies EXIF orientation when computing displayed dimensions;
- records image format, dimensions, color mode, file size, and SHA-256 hash;
- rejects invalid, unsupported or over-limit files with a clear error;
- returns JSON with `inference_performed: false` — **no simulated defect predictions**.

It does not upload or retain the image, train a model, determine product acceptability, or make defect/safety claims.

## Install (Linux / Python 3.11+)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Try a real image

```bash
python -m forgesight inspect /absolute/path/to/your/image.png
```

The image stays on the local machine. No cloud use occurs. Alternatively run `forgesight inspect ...` after installing.

## Planned architecture

```text
src/forgesight/image_io.py     Validate and identify inputs (implemented)
src/forgesight/cli.py          Thin command-line adapter (implemented)
src/forgesight/__main__.py     `python -m forgesight` (implemented)
tests/                        Synthetic-image regression tests (implemented)
docs/                         Data/engineering decisions (started)
data/                         Local data only, never commit raw downloads
models/                       Future audited, portable weights
reports/                      Future reproducible experiment outputs
```

**Separation principle:** image validation must not depend on the future neural-network architecture or web framework. The same core can later be called by a FastAPI server or another runtime adapter.

## Next gated steps

1. Inspect VisA's official file layout and split CSV without contaminating the held-out test protocol.
2. Confirm licensing separately for dataset, training code and pretrained weights.
3. Inspect sample images and establish an input/color-space contract.
4. Build a classical, explicitly named baseline and evaluate it before a deep-learning model.
5. Select a deployable model using measured defect localization, latency and resource use.

No Tesla proprietary imagery, no Vultr dependency, no claim of factory qualification.
