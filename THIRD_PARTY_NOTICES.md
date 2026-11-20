# Third-party notices

ForgeSight's [MIT License](LICENSE) covers original project software. It does not replace the licenses of dataset material, models, or dependencies. No endorsement by the upstream authors or publishers is implied.

## Visual Anomaly (VisA)

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

The upstream [NOTICE](https://github.com/amazon-science/spot-diff/blob/main/NOTICE) and [dataset license](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET) identify the dataset rights. VisA dataset material is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Retain attribution, a license link, and an indication of changes when redistributing these derivatives. The license includes warranty and liability disclaimers; see the linked legal text.

Source: [Amazon Science SPot-the-Difference / VisA](https://github.com/amazon-science/spot-diff), also documented in the [AWS Registry of Open Data](https://registry.opendata.aws/visa/).

Citation: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*. 2022. [arXiv:2207.14315](https://arxiv.org/abs/2207.14315).

Public derivatives in this source repository:

- **16 normal-training photographs** in `frontend/public/dataset/gallery/`: downscaled with aspect ratio preserved, JPEG recompressed, original metadata omitted. Source paths, source hashes, and derivative hashes remain in the gallery manifest. [Gallery attribution](frontend/public/dataset/gallery/ATTRIBUTION.md)
- **Four historical triptychs** in `frontend/public/research-atlas/`: dataset-derived images combined with ForgeSight interpolated patch-distance maps and overlays. [Atlas attribution](frontend/public/research-atlas/ATTRIBUTION.md)
- Selected source-path, label, hash, and aggregate/diagnostic metadata supporting the scientific records. These do not provide the original dataset, masks, frozen banks, or private report files.

The README's historical preview reuses one of the attributed atlas triptychs. No new live-application screenshot is included in this release. The gallery examples are not confirmed members of the eight-image-per-category frozen banks.

## DINOv2-Small

Upstream model: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small). Upstream implementation: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2). The model repository identifies the model as [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Reference: Maxime Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193).

ForgeSight uses frozen features; it does not claim to have trained DINOv2. The encoder snapshot, original license material, and any applicable upstream notices must accompany separately authorized redistribution of the model. Model binaries are not included in this repository or this source release. The exact revision and weight fingerprint are documented in the [CPU runtime guide](docs/engineering/RELEASE_CPU_RUNTIME.md).

## Software dependencies

Python and frontend dependencies retain their own licenses. Direct dependencies are declared in `pyproject.toml` and `frontend/package.json`; the npm dependency graph is recorded in `frontend/package-lock.json`. Installing dependencies does not grant rights to redistribute unrelated research data or model assets. A separately distributed application/container must retain applicable dependency, dataset, and model notices for what it actually bundles.
