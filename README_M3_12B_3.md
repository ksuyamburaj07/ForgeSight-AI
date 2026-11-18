# ForgeSight AI — M3.12B-3 | Evidence-led visual failure review

## Aim and status

This is a **CPU-only, no-new-model, no-new-inference** desk review of the 24 deterministic M3.12B-2 image cases. The generated worksheet has *preliminary visual observations* for discussion, not a completed human review. It intentionally leaves the four human-review columns blank.

**Scientific status:** post-hoc descriptive investigation on a repeatedly reused benchmark. No model weights, reference banks, thresholds or performance claims have changed. The original six M3.12B-2 cases per PCB remain selected by their earlier outcomes. No new final-test conclusions are permitted.

## Directly verified facts

- Previous B2 report SHA-256: `0ab5d57553835f149ee22bdc6fde4b3323693e494beab698a335bf49ea923701`
- Fixed-selection manifest SHA-256: `eb24292440134b06c155a25972fd32ca17c0fc0566b916715bc184b3b80b892b`
- 24 sampled images: 16 GT-labelled defects + 8 labelled normals.
- Across 16 selected defective examples, mean map value inside GT is greater than mean outside GT. This is **not** proof that the hottest patch corresponds to the GT or that small defect size caused a miss.
- Reproduced scores differed from the recorded M3.12A values by at most ~3.73e-7 in the B2 report.
- The four example atlases use a fixed display scale [0, 0.5] but do not preserve full original anomaly-map arrays.

## What this DOES NOT establish

The GT mask is a defect annotation, **not** a complete PCB foreground mask. The area outside GT includes sound components, PCB substrate, pins, and the photographic support surface. Thus `mean_map_outside_gt` is **not** a "background" measurement. It would be misleading to calculate a numeric foreground/background attribution percentage from exported RGB atlas PNGs. The B2 package also does not contain the tiled detector's pixel maps.

## Main case-level leads (not conclusions)

- PCB1: examples localize around lower connector pins; even an image with faint pin-region response can fall below the frozen decision threshold.
- PCB2: `017.JPG` (detected only by tiled) has ~28.3% of the hottest 1% of M3.12A-map pixels inside GT, while GT covers ~0.31% of the image; M3.12A still accepts it.
- PCB3: the diagonal green-support-surface feature in normal `0237.JPG` visibly overlaps a prominent response. This is a **visual** scene-sensitivity hypothesis, not an attribution percentage.
- PCB4: normal `0552.JPG` shows a conspicuous response near/above the board's upper edge. A measured board+pin foreground ROI would be needed to determine whether it is actually outside the product.

## How to use the package

1. Keep the immutable M3.12B-2 ZIP and the completed M3.12A notebooks unchanged.
2. Extract this ZIP into `/data/Projects/Project_05` using `unzip -n`.
3. Run `python -m unittest discover -s tests -p 'test_m312b3_evidence_review.py' -v` (checks the review data contract).
4. **Optional source verification** (if you have the B2 Google Drive ZIP downloaded to your Dell):

   ```bash
   python scripts/m312b3_verify_evidence_v1.py \
     --source-zip ~/Downloads/m312b_2_visual_failure_audit_v1-20261112T164158Z-1-001.zip \
     --worksheet reports/eda/m312b3_provisional_reviewer_worksheet_v1.csv \
     --summary reports/eda/m312b3_evidence_summary_v1.json
   ```

   This only reads and verifies. It never modifies model artifacts, originals, or the report.
5. Open the four B2 atlas images in the order PCB1-4; for each of the 24 cases, review `preliminary_visual_observation`, then fill the blank `human_*` fields in **a new file** `reports/eda/m312b3_human_review_v1.csv`. Keep the distributed provisional worksheet immutable.
6. Mark every unresolved distinction clearly. A bright region inside a GT contour is not equivalent to a successful image-level decision, and a bright area outside GT is not necessarily outside the PCB.

## Next research gate (not executed now)

To measure PCB-versus-background contribution, define an independently checked **foreground mask for board, connector pins, and components** on a fixed small diagnostic sample, and preserve raw anomaly maps as floats (not colormapped screenshots). Define a region-level metric *before* examining its values. Do not retune the frozen M3.12A detector using labels from these reused cases. Any altered model/policy needs a separate development study and genuinely new validation to support future generalization claims.

## Integrity and privacy

The package contains no raw VisA photo, new checkpoint, or trained weights. The annotation CSV is qualitative and can be shared only with correct attribution and an explicit retrospective qualifier. This is not a Tesla or production QA system.
