# ForgeSight AI — M3.12B-2 · Fixed-case visual failure audit

**Version:** 1 · 12 November 2026  
**Scientific designation:** Retrospective *post-hoc qualitative failure analysis*, **not an independent test**.

## Why this stage exists

M3.12A improved detection at its frozen thresholds (255/400 defects detected) but did **not** beat the earlier tiled DINOv2 configuration in macro image AUROC. The M3.12B-1 paired audit found 120 anomalies missed by both detectors and 49 normal images falsely flagged only by M3.12A. Before discussing new architecture or threshold policy, we need to examine *actual failure cases with registered ground-truth masks*.

**This package does not change or optimize any model.** It only recreates M3.12A anomaly maps for a few predetermined cases and compares their old scores/decisions with tiled DINOv2. Tiled heatmaps are **not** recreated in this stage.

### Case selection, fixed before visualization

The M3.12B-1 802-row paired case manifest was used to select the **first lexicographic image path per PCB per outcome**. Exactly six cases per PCB, **24 total**:

1. Defect missed by both detectors.
2. Defect detected by M3.12A only.
3. Defect detected by tiled DINOv2 only.
4. Defect detected by both detectors.
5. Normal image falsely flagged by M3.12A only.
6. Normal image correctly accepted by both (control).

This deterministic method is *not representative sampling*; the selected cases were selected using already-inspected benchmark labels and decisions, so **no thresholds, architecture choices, or claims of independent performance may be based on them**.

Manifest SHA-256: `eb24292440134b06c155a25972fd32ca17c0fc0566b916715bc184b3b80b892b`.

## First: install the research script and run CPU-only tests on Dell

From `/data/Projects/Project_05`:

```bash
unzip -n ~/Downloads/ForgeSight_M3_12B_2_Visual_Failure_Audit_v1.zip
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_m3_12b_2_visual_audit_contract.py' -v
```

The six bundled unit tests run on CPU without opening VisA images. They validate selection/flag logic, synthetic ground-truth statistics, and immutable report behavior. Their success **does not** validate the live Colab inference pipeline.

## Then: use the same existing M3.12A Colab notebook

**Do not create another model-training notebook, and do not rerun the completed 802-image benchmark cell.** Use the same M3.12A notebook in which you completed the research and append two code cells at the end.

1. In Colab, open the **Files** sidebar on the left and upload `scripts/m3_12b_2_visual_failure_audit_v1.py` from your Dell. It will normally be accessible as `/content/m3_12b_2_visual_failure_audit_v1.py`.
2. If your **original Colab runtime is still connected with all M3.12A Python variables**, proceed. You may use the A100, L4 or T4. This stage processes just 24 images; there is no training or full-benchmark replay.
3. If the Colab session was restarted, restore the original notebook's **non-benchmark setup cells only**: GPU environment, imports, Drive mount, historical cohort audit, existing data extraction/checks, frozen model, preprocessing, saved reference banks and saved normal pilot (those cells reuse or verify existing artifacts). **Do not execute the pre-existing benchmarking cell**; it refuses overwrites. The exact cell indices in the saved notebook are `3, 7, 9, 11, 13, 15, 17, 18, 20, 22` (zero-based; adjust for any additions; install dependencies only if needed). Verify the notebook's prior fixed model/bank hashes before proceeding.
4. Add and run the following **new Colab code cell**:

```python
from pathlib import Path
script = Path('/content/m3_12b_2_visual_failure_audit_v1.py')
assert script.is_file(), f'Upload the script into Colab Files first: {script}'
exec(compile(script.read_text(encoding='utf8'), str(script), 'exec'), globals())
m312b2_run(preflight_only=True)
```

Expected: `M3.12B-2 PREFLIGHT PASS — 24 selected cases, 4 PCB categories` and `Official benchmark images decoded during preflight: NO`. Preflight validates the original 802-result SHA, normal pilot SHA, lock SHA, pinned model weights, original VisA archive SHA, all four reference-bank SHAs, the frozen case manifest and label/score integrity.

5. **Only if preflight passes**, append and execute the second **new code cell**:

```python
m312b2_run(preflight_only=False)
```

This does frozen M3.12A inference on **24 selected images**, checks that each recomputed score matches the original per-image score within `5e-4` (a numerical drift tolerance, *not* a detection threshold), and saves four images + per-case numeric records. It never touches training, frozen thresholds or the original benchmark report.

### Results, saved to Google Drive

`MyDrive/ForgeSight_AI/M3_12A_DINOv2S_672_v1/m312b_2_visual_failure_audit_v1/`

- `m312b_2_visual_preflight_v1.json`
- `m312b_2_fixed_case_selection_v1.json`
- `m312b_2_pcb1_visual_atlas_v1.png` through `pcb4` plus corresponding JSON files
- `m312b_2_visual_audit_report_v1.json`
- `m312b_2_reviewer_worksheet_v1.csv`

Existing output files are checked, reused or protected against overwrite. Do not manually delete or overwrite finished artifacts if a run stops. If it reports a mismatch or a `.partial.png`, send that exact message for a targeted repair.

### How to read the four pages

Each PCB page has six predetermined rows and four views: original image; **ground-truth anomaly mask** over the image (red); M3.12A anomaly heatmap; and heatmap over the image with a cyan ground-truth contour. The heatmap uses a **common fixed display scale of 0–0.5** across every category and example. Values above 0.5 are clipped for display *only*; the original numerical scores are not changed.

- `M3.12A score / threshold` shows the original frozen high-resolution decision; `Tiled score / threshold` shows the frozen tiled-model decision. They **cannot** be directly compared as raw absolute scores because these methods have different score scales.
- `ground_truth_area_fraction` describes annotated defect size as a fraction of the image.
- `fraction_of_top_1pct_heatmap_pixels_inside_gt` is **descriptive evidence**, not pixel AP or an operating threshold. A small defect naturally occupies a small fraction of pixels.
- The VisA ground-truth masks annotate the **defect**, not the complete circuit board. Heatmap intensity outside the defect **does not prove** the model is focusing on the photographic background instead of PCB structure. To prove that requires a separate predeclared foreground/background annotation or masking method.
- Human observation columns are deliberately blank. Inspect images and document exactly what is supported before proposing a cause such as viewpoint, lighting, missing components or background distractors.

### Stop rule

After this visual audit, do **not** try another model or change thresholds based on selected test cases. Summarize the visible failure types, identify what *additional independent data or geometry controls* would be needed to test each hypothesis, and then make a justified M3.13 engineering decision.

### Integrity, evidence and redistribution

Source VisA archive must remain the original publicly distributed asset with attribution; do not package raw dataset images for redistribution in this ZIP. The Colab notebook and Google Drive artifacts remain user-owned. Download the four generated atlasses and JSON report into Project_05 after reviewing them, preserving their output SHA-256 strings.
