# Architecture decision record 0001 — a narrow trustworthy first slice

**Decision:** Implement an image input validator before any model, UI or cloud resources.

**Why:** A model is useless if an input cannot be decoded predictably, and inference cannot be measured or reproduced without input identity, format and dimensions. This boundary can be tested with synthetic files and does not depend on the eventual architecture.

**Modules:**
- `image_io.py` owns content validation and returns immutable `ImageRecord` metadata.
- `cli.py` handles user interaction and formats an output as JSON. It contains no image decoding or ML logic.
- `tests/` verifies the boundary with generated images and malformed files.

**Risks not yet solved:** adversarial file parsing, concurrent upload limits, web security, model correctness, image-mode normalization, color management, calibration, inference reproducibility, operational audit storage, privacy and deployment availability. The CLI is not a production upload endpoint.

**Conventions:** No manufactured inference outputs. Explicit failures. No cloud services. Preserve original inputs without modifying them.
