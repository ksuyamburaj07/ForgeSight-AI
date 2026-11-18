"""M3.11C-1 normal-only student-teacher anomaly scoring and calibration.

No test-image decoding, defect evaluation, or training occurs in this script.
Preflight is metadata-only and records an immutable implementation lock.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

from forgesight.decision import calibrate_upper_threshold, flag_for_review
from forgesight.spatial_coverage import content_patch_mask
from forgesight.stfpm_lite import CompactStudent
from forgesight.student_discrepancy import (
    feature_discrepancy_map, topk_discrepancy_score,
)
from m3_11_student_training_smoke import teacher_features
from m3_9_resnet_feature_probe import (
    ROOT, CATEGORIES, load_protocol, read_training_image, preprocess, sha256_file,
)
from m3_9_coreset_smoke import load_verified_model

RESNET_PILOT = ROOT / "reports/eda/m3_9_resnet_normal_pilot_v1.json"
PADIM_PILOT = ROOT / "reports/eda/m3_10_padim_normal_pilot_v1.json"
FIT = ROOT / "reports/eda/m3_10_padim_smoke.json"
LOCK = ROOT / "reports/m3_11_student_normal_pilot_lock_v1.json"
OUTPUT = ROOT / "reports/eda/m3_11_student_normal_pilot_v1.json"

RESNET_SHA = "ca43c2547c009f219bb539fcb2f07f4980a9fb17ee0d21b8e680e5ba3a1a75a0"
PADIM_SHA = "0e61cf5a6ad3ea158a0062be96e6d73d2766a0cd02deb4a676fd6ef7340ced64"
FIT_SHA = "b352c560712ab0888d1bb67d174f0ce1d46db78616b460aca8d9b114c052eca1"
PROTOCOL_SHA = "18fbb68e12781d6d3166dac862dcd93cbade52fc1b9ca4713f16def3cc6d594d"
WEIGHTS_SHA = "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612"
REVISION = "491b427b45c94c7fb0e78b5474cc919aff584bbf"
TRAIN_REPORT_SHA = {
    "pcb1": "a533fc54f5dd06e3350b64cc363c0eaeec599c56817b90e4263485df63e9a417",
    "pcb2": "03449c31095837ab1a1c1b029ad0903080bae212d92ebfa8a8eddf9907eb5284",
    "pcb3": "8446ace412ac3cd55c41f39a7c0299bae3b5356f85d9fa899ceb00bc61f92f60",
    "pcb4": "4e4decc069947a3c11e5d6c0353f47a8e8a4dbb960190b3f0d5a992c406f89b3",
}
CHECKPOINT_SHA = {
    "pcb1": "1de08069afcd3c79611bf3a94120ac9689b4a733da1d576718d9542243ea7582",
    "pcb2": "e0d3baaf9f4676c2e66c8edf37dd983a183794cb35302e5e1ea38af2344f6a04",
    "pcb3": "11bee3baf9f6d50f478d0c2ed58fecc5563fd1e638b804c657fb936591efe7f2",
    "pcb4": "d39f344fad23062bc68c8c495587c0970ebed8b8cedba54046bd02b4aaad40fd",
}

SOURCES = (
    "src/forgesight/student_discrepancy.py",
    "src/forgesight/stfpm_lite.py",
    "src/forgesight/student_protocol.py",
    "src/forgesight/spatial_coverage.py",
    "src/forgesight/preprocessing.py",
    "scripts/m3_11_student_training_smoke.py",
    "scripts/m3_11_student_train_v1.py",
    "scripts/m3_11_student_normal_pilot_v1.py",
    "scripts/m3_9_resnet_feature_probe.py",
    "scripts/m3_9_coreset_smoke.py",
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes():
    return {path: sha256_file(ROOT / path) for path in SOURCES}


def check_evidence():
    """Verify immutable artifacts and normal-path exclusions, without images."""
    for path, sha in ((RESNET_PILOT, RESNET_SHA), (PADIM_PILOT, PADIM_SHA), (FIT, FIT_SHA)):
        if sha256_file(path) != sha:
            raise ValueError(f"Historic research artifact changed: {path.name}")

    resnet = _read_json(RESNET_PILOT)
    padim = _read_json(PADIM_PILOT)
    fitting = _read_json(FIT)
    protocol = load_protocol()
    if (
        resnet["protocol_sha256"] != PROTOCOL_SHA
        or padim["protocol_sha256"] != PROTOCOL_SHA
        or fitting["protocol_sha256"] != PROTOCOL_SHA
        or resnet["model_revision"] != REVISION
        or padim["resnet_model_revision"] != REVISION
        or fitting["resnet_revision"] != REVISION
        or resnet["official_test_images_accessed"] is not False
        or padim["official_test_images_accessed"] is not False
        or fitting["official_test_images_accessed"] is not False
        or set(protocol["categories"]) != set(CATEGORIES)
    ):
        raise ValueError("Frozen development protocol/provenance mismatch.")

    records = {}
    for category in CATEGORIES:
        report_path = ROOT / f"reports/eda/m3_11_stfpm_lite_{category}_train_v1.json"
        checkpoint_path = ROOT / f"models/m3_11_stfpm_lite_{category}_v1.pt"
        if sha256_file(report_path) != TRAIN_REPORT_SHA[category]:
            raise ValueError(f"Changed {category} training report.")
        if sha256_file(checkpoint_path) != CHECKPOINT_SHA[category]:
            raise ValueError(f"Changed {category} student checkpoint.")
        record = _read_json(report_path)
        old = resnet["categories"][category]
        paired = padim["categories"][category]
        fit = fitting["categories"][category]
        partition = protocol["categories"][category]

        if (
            record["status"] != "NORMAL_ONLY_TRAINING_NOT_DEFECT_VALIDATED"
            or record["category"] != category
            or record["protocol_sha256"] != PROTOCOL_SHA
            or record["teacher_revision"] != REVISION
            or record["teacher_weights_sha256"] != WEIGHTS_SHA
            or record["checkpoint_sha256"] != CHECKPOINT_SHA[category]
            or record["student_parameter_count"] != 117752
            or record["teacher_frozen"] is not True
            or record["official_test_images_accessed"] is not False
            or record["real_defect_recall_measured"] is not False
            or record["threshold_calibrated"] is not False
            or record["best_epoch"] != 10
            or paired["calibration_paths"] != old["calibration_paths"]
            or paired["validation_paths"] != old["validation_paths"]
        ):
            raise ValueError(f"{category} training/calibration provenance mismatch.")
        train, monitor = record["train_paths"], record["monitor_paths"]
        calibration = old["calibration_paths"]
        validation = old["validation_paths"]
        reference = old["reference_paths"]
        reserved = set(fit["fitting_paths"]) | set(reference) | set(calibration)
        all_groups = [train, monitor, calibration, validation, reference, fit["fitting_paths"]]
        if (
            any(len(g) != len(set(g)) for g in all_groups)
            or (len(train), len(monitor), len(calibration), len(validation)) != (256, 32, 24, 12)
            or set(train) & set(monitor)
            or not set(train + monitor + calibration + reference + fit["fitting_paths"]) <= set(partition["training"])
            or not set(validation) <= set(partition["validation"])
            or set(partition["training"]) & set(partition["validation"])
            or set(train + monitor) & (reserved | set(validation))
            or set(calibration) & (set(reference) | set(fit["fitting_paths"]))
            or set(reference) & set(fit["fitting_paths"])
            or set(record["excluded_reference_and_calibration_paths"]) != reserved
        ):
            raise ValueError(f"{category} normal partition separation mismatch.")
        records[category] = {
            "calibration_paths": calibration,
            "validation_paths": validation,
            "train_report_sha256": TRAIN_REPORT_SHA[category],
            "checkpoint_sha256": CHECKPOINT_SHA[category],
        }
    return records


def preflight():
    if LOCK.exists() or OUTPUT.exists():
        raise FileExistsError("M3.11C-1 already locked or evaluated; no overwrite.")
    records = check_evidence()
    record = {
        "status": "STUDENT_NORMAL_CALIBRATION_LOCKED",
        "method": "STFPM_lite_2scale_normalized_squared_feature_discrepancy",
        "score_rule": "mean_largest_5_valid_28x28_discrepancies",
        "calibration_quantile": 0.95,
        "comparison_rule": "strictly_greater_than_threshold",
        "resnet_pilot_sha256": RESNET_SHA,
        "padim_pilot_sha256": PADIM_SHA,
        "fit_report_sha256": FIT_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "source_sha256": source_hashes(),
        "categories": records,
        "official_test_images_decoded_during_preflight": False,
        "note": "Normal-only reused, previously inspected development validation; not an independent final test.",
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print("M3.11C-1 preflight lock:", LOCK)
    print("Lock SHA256:", sha256_file(LOCK))
    print("Official test images decoded: NO")


@torch.inference_mode()
def score_normal(path, category, teacher, student, mean, std):
    image, _ = read_training_image(path, category)
    tensor, geometry = preprocess(image, mean, std)
    mask = content_patch_mask(geometry, side=224, patch_size=8)
    if tensor.shape != (1, 3, 224, 224) or mask.shape != (28, 28):
        raise ValueError("Unexpected student input geometry.")
    start = time.perf_counter()
    target = teacher_features(teacher, tensor)
    predicted = student(tensor)
    anomaly = feature_discrepancy_map(target, predicted, mask)
    score = topk_discrepancy_score(anomaly, mask, top_k=5)
    elapsed = (time.perf_counter() - start) * 1000
    return {"image": path, "score": score, "processing_ms": round(elapsed, 3)}


def evaluate():
    if not LOCK.exists() or OUTPUT.exists():
        raise ValueError("Missing preflight lock or result already exists.")
    records = check_evidence()
    lock = _read_json(LOCK)
    if (
        lock["status"] != "STUDENT_NORMAL_CALIBRATION_LOCKED"
        or lock["resnet_pilot_sha256"] != RESNET_SHA
        or lock["padim_pilot_sha256"] != PADIM_SHA
        or lock["fit_report_sha256"] != FIT_SHA
        or lock["protocol_sha256"] != PROTOCOL_SHA
        or lock["source_sha256"] != source_hashes()
        or lock["categories"] != records
        or lock["calibration_quantile"] != 0.95
    ):
        raise ValueError("M3.11C-1 implementation lock changed.")

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    teacher, probe = load_verified_model()
    if (
        probe["weights_sha256"] != WEIGHTS_SHA
        or probe["model_revision"] != REVISION
        or probe["protocol_sha256"] != PROTOCOL_SHA
    ):
        raise ValueError("Unexpected pretrained teacher weights/protocol.")
    teacher.eval()
    teacher.requires_grad_(False)
    mean, std = probe["image_normalization_mean"], probe["image_normalization_std"]
    report = {
        "experiment": "M3.11C-1 frozen student teacher normal-only calibration",
        "status": "NORMAL_ONLY_STUDENT_DEVELOPMENT_PILOT",
        "teacher_revision": REVISION,
        "teacher_weights_sha256": WEIGHTS_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "lock_sha256": sha256_file(LOCK),
        "source_sha256": source_hashes(),
        "calibration_quantile": 0.95,
        "image_score_rule": "mean_top_five_valid_two_scale_normalized_squared_discrepancies",
        "validation_images_previously_inspected": True,
        "official_test_images_accessed": False,
        "real_defect_recall_measured": False,
        "model_weights_updated": False,
        "categories": {},
    }
    for category in CATEGORIES:
        student = CompactStudent().cpu()
        state = torch.load(
            ROOT / f"models/m3_11_stfpm_lite_{category}_v1.pt",
            map_location="cpu", weights_only=True,
        )
        if not isinstance(state, dict):
            raise ValueError(f"{category}: student checkpoint is not a state dict.")
        student.load_state_dict(state, strict=True)
        student.eval()
        student.requires_grad_(False)
        previous = records[category]
        calibration, validation = [], []

        print(f"\n{category.upper()} — student calibration (24 normals)", flush=True)
        for index, path in enumerate(previous["calibration_paths"], 1):
            calibration.append(score_normal(path, category, teacher, student, mean, std))
            if index % 8 == 0:
                print(f"  {index}/24 scored", flush=True)
        threshold = float(calibrate_upper_threshold([r["score"] for r in calibration], 0.95))
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid student calibration threshold.")

        print(f"{category.upper()} — student normal validation (12 normals)", flush=True)
        for index, path in enumerate(previous["validation_paths"], 1):
            observation = score_normal(path, category, teacher, student, mean, std)
            observation["flagged_for_review"] = bool(flag_for_review(observation["score"], threshold))
            validation.append(observation)
            if index % 6 == 0:
                print(f"  {index}/12 scored", flush=True)
        flags = sum(int(x["flagged_for_review"]) for x in validation)
        timing = statistics.median(x["processing_ms"] for x in validation)
        report["categories"][category] = {
            "training_report_sha256": previous["train_report_sha256"],
            "student_checkpoint_sha256": previous["checkpoint_sha256"],
            "calibration_paths": previous["calibration_paths"],
            "validation_paths": previous["validation_paths"],
            "frozen_threshold": threshold,
            "calibration": calibration,
            "validation": validation,
            "normal_flags": flags,
            "normal_count": len(validation),
            "normal_flag_fraction": flags / len(validation),
            "median_processing_ms": timing,
        }
        print(
            f"{category.upper()}: threshold={threshold:.6f} "
            f"normal_flags={flags}/12 median={timing:.1f}ms", flush=True,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("\nFORGESIGHT — M3.11C-1 STUDENT NORMAL PILOT")
    print("Report:", OUTPUT)
    print("SHA256:", sha256_file(OUTPUT))
    print("Official test images accessed: NO")
    print("Defect recall measured: NO")
    print("Status: NORMAL-ONLY CALIBRATION COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        evaluate()


if __name__ == "__main__":
    main()
