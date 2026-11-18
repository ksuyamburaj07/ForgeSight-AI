"""M3.12B-2: append-only, paired visual audit in the EXISTING M3.12A Colab notebook.

Load this source in an existing running M3.12A notebook with:
  exec(compile(open('/content/m3_12b_2_visual_failure_audit_v1.py').read(),
               'm3_12b_2_visual_failure_audit_v1.py', 'exec'), globals())
  m312b2_run(preflight_only=True)
  m312b2_run(preflight_only=False)

No training, no model selection, no threshold changes, no new benchmark sweep.
"""
import base64
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

EXPECTED = {
    "benchmark_sha256": "0d33b11936c95241c503a318c447d6cad3105c5e0a93d2ec3e5068a8e9d4f35b",
    "pilot_sha256": "ba94ee6105b0089eed87b0e76d78f1d85697e258e66046925892790994b8ae45",
    "lock_sha256": "2b4872f4115f3a683ad1d6e98934a9c6d6ce41bf0a63535e75704b861217940b",
    "protocol_sha256": "4b7ef5a06923925922909039c4ee6f4185bc44c95a1e8a047681f6e5d6446a11",
    "weights_sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "fixed_cases_sha256": "eb24292440134b06c155a25972fd32ca17c0fc0566b916715bc184b3b80b892b",
    "reference_banks": {
        "pcb1": "922ece4f2df36f33a59218f2953729dc19423decac3dff85b27af9b95064f411",
        "pcb2": "f6e020f8fdc2ba036f5cd77568c529da5f8a3c8188949d9b881693e66b8a4617",
        "pcb3": "910eac6c4d8bd807a6c8f1e62dc55fe57dd0153db9b937e938ea86ea0ec98a82",
        "pcb4": "20b7e8e58e3a3f11651c9b7793797570c5b7bd6c32e5f9cb44494ee6760bc3fc",
    },
}
SELECTION_GZIP_B64 = "H4sIAAAAAAAC/82Z3W7jNhCF7/MUQq7jhBxyhuT2qtsCRQu02PuiMGSZjoXKliEpaYLFvnsPlSabXXjtJGu6Ngw7kfgjfRydOUN/PCuK83i3iV29iuvh/F1x/ru51PR+QpNFfRfn+Oz6YdLEu7oqmwkaTqpyiNdtdz9pb4aqXcXzizRIH5tYDXW7nnY3TUwDjT2LsWd73ZWbZV0V9aq8jsWmHJZFvS76+q7YdHEeq6bEV/HfgP1F0cdN2WGe5r7AlMWHn95fFNWy7eO6WHTtKvW6rdubvrmf1Ot+g5nRvYs3Pb5mcV0tV2X39w/FPPZVV2+G+jYW7bq5f7jUTVljtml5M6+HadXfTvtlSSzjNYuEGKP3wuQpBs8sPGeSmcaBGKpq7pS35YwXVZhL5ebEUYm2frawGmOME1RlH3sM9yf+KYqP4+d4+AFcmmhTzfTYeDwzYnk8fPVzOZRXv6ZD/dWP63ZVNvdXitzlbx9++dylKWexQRf9dORxOTDMPC6AZDprh+V0Vff9525D3eDW+6rtUkN1SY6DZqdJa4+78F81HJZd7JdtMx8bGy1GvBYSo4P3/FXjRVNeo92ibPr4dGplND2bT1s2TsiFwEpbHb5s9+V0uC5llXdBG2XEWPmy8fPpxhOfLg7IW6nX817Hf6ZPYbYVN67bWOcpGKu99zY7bkn8SAUB9YPgHrqbHLTp9bQfKOzhTUG0D0YBo7A6BO8E4Ju4WbNRloIjgPRyutHNb1STeRzw9w7gzgolPRFjNOibzMRJW2c5GO8c4tywP7EI/6PtQDwFuN1OXG0jvh57PenJdIyCaYksudoh5dYwkYWwgjxLfimXQMSWLDIj8kfgU0Wv9evRj8FeVlXcwFl8m7n2CD+IrIaqI/woP3TNHm88YgKOSKZHlBjaTp0Ol0BfYFiQNin4wHgFZxTvFhgio8ZEK+w9Fsm/BTmxUsYRU1CY2u5WdSsuiCVFWjgt0VGR6xyehbRLuuIgtUY7Q5SfOFBD0AXPE+OJCd8P/KXC8jre2uVxLZBV5byx1lhWgZw7BPGdvsWSBAlQJahLKnpON8Qpk3EBZbEKb4fcSUK5kaPgMt4ifUDICX7sxGL8MXlq5bL7FlLWqJBeFqsQOL++iHjIiklDMWlzouzhKrIZFyJYdI1Uqo3zuLcjiDqKUCUeC426X6MkPaLGmO3YzTdk3ebZatFOyAQbFERdGUP7vKJYp+D0JGXe4N6CHIWATpWRC4n9sxm3IfdYGU3oggTgyR4VubJZdltGX87eMB51Y2x+4EGcCoy8LfK8Ansz7pfqyitp59pt0cGh9hfUoVb22ZYX8t5pW0gsVthCKVihED3h+Fa59luYHLKmSXu4iX5u5KSVFYtUrVxAcUmnFuOPuZNMft/iSeG2lCSZFQrZ1QUaZpNhgV9CRnR7fMv/xz7nhgtZ7QUwtSG2bF1+SfdasWen8JiJcvqYGmO3U7cH3tPds+ESmAHSa+0hY2rPfguSrXesRKn0cxK9yZxjrVD1K5ag0r7pLuCUPLxKwaANhImOC9xk2W7BzUBfyUPNEXp7dOUAwEkQ2t6ycyZtAnw/7peKyutov8WVv2izxTkLR87GOJd+vzkA7p0JVBmPxIECNLCgzKHTDe9cngVFpwg4uLSp6oPOTdwEp1N4u2CV4xOL78ekyUzZDQtE1TgbyDEBBGUXFiTDBFIbBDtE3Z0o+qw/EDlU3vDMJhVE0Jjs2RNjcFphK0idYHpQecHnX2efzv4F8tFSGTYkAAA="
OUTCOMES = (
    "defect_both_miss", "defect_new_only", "defect_tiled_only",
    "defect_both_detect", "normal_new_only_false_alarm", "normal_both_accepted",
)
SCORE_TOLERANCE = 5e-4  # recomputation check; NOT an operating threshold
DISPLAY_VMIN, DISPLAY_VMAX = 0.0, 0.5  # common scale across every PCB/sample; display only


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cases():
    raw = gzip.decompress(base64.b64decode(SELECTION_GZIP_B64))
    # Selection manifest supplied with this package has canonical pretty-printed bytes.
    assert hashlib.sha256(raw).hexdigest() == EXPECTED["fixed_cases_sha256"]
    data = json.loads(raw)
    cases = data["cases"]
    assert len(cases) == 24
    assert tuple(x["outcome"] for x in cases[:6]) == OUTCOMES
    for cat in ("pcb1", "pcb2", "pcb3", "pcb4"):
        selected = [c for c in cases if c["category"] == cat]
        assert len(selected) == 6 and tuple(x["outcome"] for x in selected) == OUTCOMES
        assert all(c["image"].startswith(cat + "/Data/Images/") for c in selected)
        assert len({c["image"] for c in selected}) == 6
        for r in selected:
            assert r["label"] == int("/Anomaly/" in r["image"])
            assert r["m312_flag"] == (r["m312_score"] > r["m312_threshold"])
            assert r["tiled_flag"] == (r["tiled_score"] > r["tiled_threshold"])
            if r["outcome"] == "defect_both_miss": assert r["label"] == 1 and not r["m312_flag"] and not r["tiled_flag"]
            if r["outcome"] == "defect_new_only": assert r["label"] == 1 and r["m312_flag"] and not r["tiled_flag"]
            if r["outcome"] == "defect_tiled_only": assert r["label"] == 1 and not r["m312_flag"] and r["tiled_flag"]
            if r["outcome"] == "defect_both_detect": assert r["label"] == 1 and r["m312_flag"] and r["tiled_flag"]
            if r["outcome"] == "normal_new_only_false_alarm": assert r["label"] == 0 and r["m312_flag"] and not r["tiled_flag"]
            if r["outcome"] == "normal_both_accepted": assert r["label"] == 0 and not r["m312_flag"] and not r["tiled_flag"]
    return data


def write_immutable_json(path, value):
    """Creates only; on repeat verifies semantic equality, never overwrites."""
    dest = Path(path)
    if dest.exists():
        assert json.loads(dest.read_text()) == value, f"Immutable artifact mismatch: {dest}"
        return
    with dest.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify_report_and_cohort(cases, report, pilot, lock, protocol):
    assert report["lock_sha256"] == EXPECTED["lock_sha256"]
    assert report["protocol_sha256"] == EXPECTED["protocol_sha256"]
    assert report["thresholds_changed_using_test_labels"] is False
    assert report["model_weights_updated"] is False
    assert lock["protocol_sha256"] == EXPECTED["protocol_sha256"]
    assert pilot["protocol_sha256"] == EXPECTED["protocol_sha256"]
    assert tuple(report["categories"]) == ("pcb1", "pcb2", "pcb3", "pcb4")
    assert sum(len(report["categories"][cat]["per_image"]) for cat in report["categories"]) == 802
    for row in cases:
        cat, img = row["category"], row["image"]
        reference = {r["image"]: r for r in report["categories"][cat]["per_image"]}
        assert img in reference and reference[img]["label"] == row["label"]
        assert abs(float(reference[img]["score"]) - row["m312_score"]) <= 1e-10
        assert abs(pilot["categories"][cat]["threshold"] - row["m312_threshold"]) <= 1e-10
        assert abs(lock["thresholds"][cat] - row["m312_threshold"]) <= 1e-10
        manifest = {r["image"]: r["label"] for r in protocol["categories"][cat]["test"]}
        assert manifest[img] == row["label"]


def analysis_of_map(heatmap, mask):
    """Mask-based DESCRIPTIVE pixel diagnostics; does not infer physical PCB boundaries."""
    assert heatmap.shape == mask.shape
    assert heatmap.dtype == np.float32 and np.isfinite(heatmap).all()
    assert np.min(heatmap) >= 0
    total = mask.size
    top_n = max(1, math.ceil(total * 0.01))
    cutoff_indices = np.argpartition(heatmap.ravel(), -top_n)[-top_n:]
    top_fraction = float(np.mean(mask.ravel()[cutoff_indices]))
    fraction = float(mask.mean())
    return {
        "ground_truth_area_fraction": fraction,
        "mean_heatmap_inside_gt": float(heatmap[mask].mean()) if mask.any() else None,
        "mean_heatmap_outside_gt": float(heatmap[~mask].mean()) if (~mask).any() else None,
        "fraction_of_top_1pct_heatmap_pixels_inside_gt": top_fraction if mask.any() else None,
        "display_heatmap_fraction_above_fixed_vmax": float(np.mean(heatmap > DISPLAY_VMAX)),
    }


def get_mask(mask_fn, image, row):
    mask = mask_fn(row["image"], image.size, row["label"])
    assert mask.shape == (image.height, image.width)
    assert mask.dtype == bool
    assert bool(mask.any()) == bool(row["label"])
    return mask


def plot_category(cat, selected, output_path):
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(6, 4, figsize=(17, 20), constrained_layout=True)
    titles = ("Original image", "Ground truth (red)", "Fixed-scale heatmap", "Heatmap + GT outline")
    for j, title in enumerate(titles): axes[0, j].set_title(title, fontsize=13)
    for i, (row, image, mask, heat, stats) in enumerate(selected):
        # Downsample for illustration only; scientific calculations used original resolution.
        orig = np.asarray(image)
        w, h = image.size
        disp_w = min(w, 650)
        disp_h = max(1, round(h * disp_w / w))
        rgb = cv2.resize(orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        field = cv2.resize(heat, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(mask.astype(np.uint8), (disp_w, disp_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ax0, ax1, ax2, ax3 = axes[i]
        ax0.imshow(rgb)
        ax1.imshow(rgb)
        if gt.any():
            ax1.imshow(np.ma.masked_where(~gt, gt.astype(float)), cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        ax2.imshow(field, cmap="magma", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX)
        ax3.imshow(rgb)
        ax3.imshow(field, cmap="magma", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX, alpha=0.55)
        if gt.any(): ax3.contour(gt.astype(float), levels=[0.5], colors="cyan", linewidths=1.15)
        label = row["outcome"].replace("_", " ").upper()
        status_n = "REVIEW" if row["m312_flag"] else "ACCEPT"
        status_t = "REVIEW" if row["tiled_flag"] else "ACCEPT"
        ax0.set_ylabel(
            f"{label}\n{row['image'].split('/')[-1]}\n"
            f"M3.12A {row['m312_score']:.4f}/{row['m312_threshold']:.4f} {status_n}\n"
            f"Tiled {row['tiled_score']:.4f}/{row['tiled_threshold']:.4f} {status_t}",
            fontsize=9, rotation=0, ha="right", va="center", labelpad=9)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(DISPLAY_VMIN, DISPLAY_VMAX), cmap="magma"),
                      ax=axes[:, 2:], orientation="horizontal", fraction=0.012, pad=0.015)
    cb.set_label("Original-space cosine NN distance after fixed smoothing; display clipped to [0, 0.5]")
    fig.suptitle(
        f"ForgeSight M3.12B-2 · {cat.upper()} · 6 deterministic cases · REUSED retrospective set\n"
        "Ground truth ≠ model decision; a bright background is not automatically a PCB defect",
        fontsize=14)
    target = Path(output_path)
    assert not target.exists(), f"Refusing to overwrite image: {target}"
    tmp = target.with_name(target.stem + ".partial.png")
    assert not tmp.exists(), f"Partial image from earlier attempt: {tmp}; inspect before retry"
    fig.savefig(tmp, dpi=115, bbox_inches="tight")
    plt.close(fig)
    assert not target.exists()
    tmp.rename(target)


def m312b2_run(preflight_only=True):
    """Run first with preflight_only=True, then False. Uses current M3.12A notebook state."""
    g = globals()
    needed = ("DRIVE_OUT", "PROTOCOL", "CATEGORIES", "VISA_ROOT", "absolute_image",
              "absolute_mask", "open_rgb", "REFERENCE_BANKS", "patches", "score_from_embeddings",
              "MODEL_WEIGHTS_SHA", "model_file", "model", "RESOLUTION", "LOCK_PATH", "PILOT_PATH")
    absent = [name for name in needed if name not in g]
    if absent:
        raise RuntimeError(
            "M3.12A notebook functions are missing (possibly a new Colab session): " + ", ".join(absent)
            + ". See README: restore ONLY non-benchmark setup cells, never rerun completed benchmark."
        )
    import cv2
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    import torch

    assert tuple(CATEGORIES) == ("pcb1", "pcb2", "pcb3", "pcb4")
    assert RESOLUTION == 672
    assert MODEL_WEIGHTS_SHA == EXPECTED["weights_sha256"]
    assert sha256_file(model_file) == EXPECTED["weights_sha256"]
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert Path(VISA_ROOT).exists() and Path(DRIVE_OUT).exists()
    old_report = Path(DRIVE_OUT) / "m312_image_pixel_retro_results_v1.json"
    assert sha256_file(old_report) == EXPECTED["benchmark_sha256"], "Frozen M3.12A report mismatch"
    assert sha256_file(PILOT_PATH) == EXPECTED["pilot_sha256"]
    assert sha256_file(LOCK_PATH) == EXPECTED["lock_sha256"]
    assert sha256_file(Path(DRIVE_OUT) / "dataset_cache/VisA_20220922.tar") == (
        "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
    ), "Official VisA archive missing/changed; keep the original data integrity requirement"
    for cat in CATEGORIES:
        bank_path = Path(DRIVE_OUT) / f"m312_{cat}_reference_bank_v1.npz"
        assert sha256_file(bank_path) == EXPECTED["reference_banks"][cat]
        assert torch.is_tensor(REFERENCE_BANKS[cat]) and REFERENCE_BANKS[cat].shape[1] == 384
        with np.load(bank_path, allow_pickle=False) as saved:
            loaded = saved["vectors"]
            assert np.allclose(loaded, REFERENCE_BANKS[cat].detach().cpu().numpy(), rtol=1e-6, atol=1e-6)

    cases_data = load_cases()
    cases = cases_data["cases"]
    report = json.loads(old_report.read_text())
    pilot_data = json.loads(Path(PILOT_PATH).read_text())
    lock = json.loads(Path(LOCK_PATH).read_text())
    verify_report_and_cohort(cases, report, pilot_data, lock, PROTOCOL)

    out = Path(DRIVE_OUT) / "m312b_2_visual_failure_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    selection_path = out / "m312b_2_fixed_case_selection_v1.json"
    if selection_path.exists():
        assert sha256_file(selection_path) == EXPECTED["fixed_cases_sha256"], "Frozen selection changed"
    else:
        with selection_path.open("xb") as f:
            f.write(gzip.decompress(base64.b64decode(SELECTION_GZIP_B64)))
    assert sha256_file(selection_path) == EXPECTED["fixed_cases_sha256"], "Case selection must remain identical"
    preflight = {
        "stage": "M3.12B-2-reused-benchmark-visual-failure-audit",
        "scientific_status": "DESCRIPTIVE_POST_HOC_NOT_INDEPENDENT",
        "case_selection_rule": cases_data["selection_rule"],
        "selected_cases": len(cases),
        "source_benchmark_sha256": EXPECTED["benchmark_sha256"],
        "source_normal_pilot_sha256": EXPECTED["pilot_sha256"],
        "source_benchmark_lock_sha256": EXPECTED["lock_sha256"],
        "case_selection_sha256": EXPECTED["fixed_cases_sha256"],
        "reference_bank_sha256": EXPECTED["reference_banks"],
        "model_weights_sha256": EXPECTED["weights_sha256"],
        "display_colormap_fixed_vmin": DISPLAY_VMIN,
        "display_colormap_fixed_vmax": DISPLAY_VMAX,
        "benchmark_images_decoded_during_preflight": False,
        "thresholds_modified": False,
        "weights_updated": False,
        "intended_use": "Human qualitative error diagnosis; no selection or validation claims",
    }
    write_immutable_json(out / "m312b_2_visual_preflight_v1.json", preflight)
    print("M3.12B-2 PREFLIGHT PASS — 24 selected cases, 4 PCB categories")
    print("Frozen benchmark SHA256:", EXPECTED["benchmark_sha256"])
    print("Case-selection SHA256:", EXPECTED["fixed_cases_sha256"])
    print("Official benchmark images decoded during preflight: NO")
    print("Output directory:", out)
    if preflight_only:
        print("STOP at preflight. Next, call m312b2_run(preflight_only=False).")
        return

    def original_map(patch_map, geometry):
        left, top, w, h, old_w, old_h = geometry
        canvas = cv2.resize(patch_map, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
        content = canvas[top:top + h, left:left + w]
        full = cv2.resize(content, (old_w, old_h), interpolation=cv2.INTER_LINEAR)
        return gaussian_filter(full, sigma=4.0).astype(np.float32)

    def original_mask(rel, image_size, label):
        iw, ih = image_size
        if label == 0:
            return np.zeros((ih, iw), dtype=bool)
        with Image.open(absolute_mask(rel)) as file:
            a = np.asarray(file)
        m = (a != 0) if a.ndim == 2 else np.any(a[..., :3] != 0, axis=-1)
        assert m.shape == (ih, iw) and m.any()
        return m.astype(bool)

    all_rows = []
    for cat in CATEGORIES:
        png = out / f"m312b_2_{cat}_visual_atlas_v1.png"
        meta = out / f"m312b_2_{cat}_visual_findings_v1.json"
        if png.exists() or meta.exists():
            assert png.exists() and meta.exists(), f"Only one category artifact exists for {cat}; inspect first"
            previous = json.loads(meta.read_text())
            assert previous["source_benchmark_sha256"] == EXPECTED["benchmark_sha256"]
            assert previous["case_selection_sha256"] == EXPECTED["fixed_cases_sha256"]
            assert previous["atlas_sha256"] == sha256_file(png)
            assert [r["image"] for r in previous["records"]] == [r["image"] for r in cases if r["category"] == cat]
            print(cat, "resumed verified atlas; no inference replayed")
            all_rows.extend(previous["records"])
            continue
        selected = []
        rows = []
        for row in (r for r in cases if r["category"] == cat):
            image = open_rgb(absolute_image(row["image"]))
            feat, valid, geo = patches(image)
            score, grid = score_from_embeddings(feat, valid, REFERENCE_BANKS[cat])
            if abs(score - row["m312_score"]) > SCORE_TOLERANCE:
                raise RuntimeError(f"Score regression detected for {row['image']}: recomputed {score}, frozen {row['m312_score']}")
            heat = original_map(grid, geo)
            mask = get_mask(original_mask, image, row)
            statistics = analysis_of_map(heat, mask)
            result = {
                **row, "recomputed_m312_score": float(score),
                "absolute_score_regression_difference": float(abs(score - row["m312_score"])),
                "visual_diagnostics": statistics,
                "reviewer_observation": "", "possible_explanation": "",
                "evidence_for_explanation": "", "remaining_uncertainty": "",
            }
            rows.append(result)
            selected.append((row, image, mask, heat, statistics))
            print(cat, row["outcome"], "score regression: PASS", flush=True)
        plot_category(cat, selected, png)
        data = {
            "source_benchmark_sha256": EXPECTED["benchmark_sha256"],
            "case_selection_sha256": EXPECTED["fixed_cases_sha256"],
            "display_range": [DISPLAY_VMIN, DISPLAY_VMAX],
            "atlas_sha256": sha256_file(png),
            "records": rows,
            "warning": "Retrospective explanatory samples only. No physical board mask; do not label all out-of-GT heatmap as background."
        }
        write_immutable_json(meta, data)
        all_rows.extend(rows)
        print(cat, "atlas saved:", png, "SHA256", data["atlas_sha256"])

    assert len(all_rows) == 24
    report_path = out / "m312b_2_visual_audit_report_v1.json"
    final = {
        **preflight,
        "visual_atlas_sha256": {cat: sha256_file(out / f"m312b_2_{cat}_visual_atlas_v1.png") for cat in CATEGORIES},
        "reviewed_cases": 24,
        "records": all_rows,
        "limitations": [
            "Only six predetermined examples per category; they are not representative estimates",
            "Qualitative selection uses reused test labels and cannot guide an independent final benchmark claim",
            "GT masks mark defects, not the entire foreground PCB; hotspot outside GT is not necessarily background",
            "Map visualization uses a common fixed [0, 0.5] scale; values above 0.5 are clipped for display only",
            "Only M3.12A maps recomputed; tiled comparisons use frozen image scores, not tiled pixel maps",
            "Human reviewer must document observations before proposing follow-up experiments",
        ],
    }
    write_immutable_json(report_path, final)
    review_csv = out / "m312b_2_reviewer_worksheet_v1.csv"
    if not review_csv.exists():
        with review_csv.open("x", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "image", "frozen_outcome", "m312_flag", "tiled_flag",
                             "gt_area_fraction", "top1pct_inside_gt_fraction", "reviewer_observation",
                             "possible_explanation", "evidence_for_explanation", "remaining_uncertainty"])
            for item in all_rows:
                st = item["visual_diagnostics"]
                writer.writerow([item["category"], item["image"], item["outcome"], int(item["m312_flag"]),
                                 int(item["tiled_flag"]), st["ground_truth_area_fraction"],
                                 st["fraction_of_top_1pct_heatmap_pixels_inside_gt"], "", "", "", ""])
    print("M3.12B-2 VISUAL AUDIT COMPLETE — 24 case maps reproduced, threshold/model unchanged")
    print("Report:", report_path)
    print("Report SHA256:", sha256_file(report_path))
    print("Worksheet:", review_csv)
    print("Next: inspect the 4 atlas images and fill human observations; do not change model settings.")
