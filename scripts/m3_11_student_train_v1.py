"""M3.11B: Train a PCB-specific compact student on disjoint normal examples.

Research stage: normal feature-matching training only; no defect-label access.
"""

import argparse
import json
import os
import statistics
import time

import torch

from forgesight.student_protocol import select_normal_cohorts
from forgesight.stfpm_lite import CompactStudent, normalized_feature_loss
from m3_11_student_training_smoke import teacher_features
from m3_9_resnet_feature_probe import (
    ROOT, CATEGORIES, load_protocol, read_training_image, preprocess, sha256_file,
)
from m3_9_coreset_smoke import load_verified_model

FIT = ROOT / "reports/eda/m3_10_padim_smoke.json"
PILOT = ROOT / "reports/eda/m3_10_padim_normal_pilot_v1.json"
SMOKE = ROOT / "reports/eda/m3_11_student_training_smoke_v0.json"
FIT_SHA = "b352c560712ab0888d1bb67d174f0ce1d46db78616b460aca8d9b114c052eca1"
PILOT_SHA = "0e61cf5a6ad3ea158a0062be96e6d73d2766a0cd02deb4a676fd6ef7340ced64"
SMOKE_SHA = "ea5356d56052ef57cc2be266d5f462088d1b28bf79396b68fd5e70715aa3c061"

SEED = 314159
TRAIN_COUNT = 256
MONITOR_COUNT = 32
BATCH_SIZE = 8
EPOCHS = 10
LR = 5e-4


def encode_normals(paths, category, teacher, mean, std):
    images, fine, coarse = [], [], []
    for idx, path in enumerate(paths, 1):
        image, _ = read_training_image(path, category)
        tensor, _ = preprocess(image, mean, std)
        if tuple(tensor.shape) != (1, 3, 224, 224):
            raise ValueError("Unexpected image tensor geometry.")
        teacher_fine, teacher_coarse = teacher_features(teacher, tensor)
        if (
            tuple(teacher_fine.shape) != (1, 128, 28, 28)
            or tuple(teacher_coarse.shape) != (1, 256, 14, 14)
        ):
            raise ValueError("Unexpected teacher feature geometry.")
        images.append(tensor[0].detach().cpu())
        fine.append(teacher_fine[0].detach().cpu())
        coarse.append(teacher_coarse[0].detach().cpu())
        if idx % 64 == 0 or idx == len(paths):
            print(f"  Encoded {idx}/{len(paths)} normal images", flush=True)
    return torch.stack(images), (torch.stack(fine), torch.stack(coarse))


@torch.no_grad()
def evaluation_loss(student, inputs, targets):
    student.eval()
    total = 0.0
    for start in range(0, len(inputs), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(inputs))
        outputs = student(inputs[start:end])
        target_batch = (targets[0][start:end], targets[1][start:end])
        loss = normalized_feature_loss(outputs, target_batch)
        total += float(loss.item()) * (end - start)
    return total / len(inputs)


def main(category):
    checkpoint = ROOT / f"models/m3_11_stfpm_lite_{category}_v1.pt"
    report_file = ROOT / f"reports/eda/m3_11_stfpm_lite_{category}_train_v1.json"
    if checkpoint.exists() or report_file.exists():
        raise FileExistsError("M3.11B output already exists; refusing to overwrite.")

    for path, expected in ((FIT, FIT_SHA), (PILOT, PILOT_SHA), (SMOKE, SMOKE_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"Historical research artifact changed: {path.name}")

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    protocol = load_protocol()
    fitting = json.loads(FIT.read_text(encoding="utf-8"))
    previous = json.loads(PILOT.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    teacher, probe = load_verified_model()
    if (
        fitting["protocol_sha256"] != probe["protocol_sha256"]
        or previous["protocol_sha256"] != probe["protocol_sha256"]
        or smoke["development_protocol_sha256"] != probe["protocol_sha256"]
        or smoke["teacher_weights_sha256"] != probe["weights_sha256"]
        or fitting["resnet_revision"] != probe["model_revision"]
        or previous["resnet_model_revision"] != probe["model_revision"]
        or smoke["teacher_model_revision"] != probe["model_revision"]
        or fitting["official_test_images_accessed"] is not False
        or previous["official_test_images_accessed"] is not False
        or smoke["official_test_images_accessed"] is not False
    ):
        raise ValueError("Research provenance mismatch.")

    partition = protocol["categories"][category]
    training_pool = partition["training"]
    validation_pool = partition["validation"]
    previous_category = previous["categories"][category]
    fitting_category = fitting["categories"][category]
    # M3.9C is the authoritative source for ResNet reference images.
    resnet_pilot_path = ROOT / "reports/eda/m3_9_resnet_normal_pilot_v1.json"
    resnet_pilot_sha = "ca43c2547c009f219bb539fcb2f07f4980a9fb17ee0d21b8e680e5ba3a1a75a0"

    if sha256_file(resnet_pilot_path) != resnet_pilot_sha:
        raise ValueError("Historical M3.9C ResNet normal pilot changed.")

    resnet_pilot = json.loads(
        resnet_pilot_path.read_text(encoding="utf-8")
    )

    if (
        resnet_pilot["protocol_sha256"] != probe["protocol_sha256"]
        or resnet_pilot["model_revision"] != probe["model_revision"]
        or resnet_pilot["official_test_images_accessed"] is not False
    ):
        raise ValueError("ResNet reference provenance mismatch.")

    resnet_category = resnet_pilot["categories"][category]

    if (
        resnet_category["calibration_paths"]
        != previous_category["calibration_paths"]
        or resnet_category["validation_paths"]
        != previous_category["validation_paths"]
        or not resnet_category["reference_paths"]
    ):
        raise ValueError("M3.9C and M3.10B normal cohorts disagree.")

    reserved = (
        fitting_category["fitting_paths"]
        + resnet_category["reference_paths"]
        + previous_category["calibration_paths"]
    )
    if set(training_pool) & set(validation_pool):
        raise ValueError("Frozen development split overlap.")
    train_paths, monitor_paths = select_normal_cohorts(
        training_pool, reserved, category=category, seed=SEED,
        train_count=TRAIN_COUNT, monitor_count=MONITOR_COUNT,
    )
    if set(train_paths + monitor_paths) & set(previous_category["validation_paths"]):
        raise ValueError("Previously inspected validation images reused.")

    teacher.eval()
    teacher.requires_grad_(False)
    print(f"\n{category.upper()} — preparing frozen teacher targets", flush=True)
    started_encoding = time.perf_counter()
    train_x, train_t = encode_normals(
        train_paths, category, teacher, probe["image_normalization_mean"],
        probe["image_normalization_std"],
    )
    monitor_x, monitor_t = encode_normals(
        monitor_paths, category, teacher, probe["image_normalization_mean"],
        probe["image_normalization_std"],
    )
    encoding_seconds = time.perf_counter() - started_encoding

    torch.manual_seed(SEED)
    student = CompactStudent().cpu()
    param_count = sum(p.numel() for p in student.parameters())
    if param_count != smoke["student_parameter_count"]:
        raise ValueError("Student architecture changed since smoke test.")
    initial_weight = student.stem[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    shuffle_rng = torch.Generator().manual_seed(SEED)

    initial_monitor = evaluation_loss(student, monitor_x, monitor_t)
    best_monitor, best_epoch, best_state = float("inf"), None, None
    history = []
    started_training = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        student.train()  # BatchNorm uses mini-batches and updates running statistics.
        indices = torch.randperm(TRAIN_COUNT, generator=shuffle_rng)
        epoch_losses, grad_norms = [], []
        for start in range(0, TRAIN_COUNT, BATCH_SIZE):
            positions = indices[start:start + BATCH_SIZE]
            inputs = train_x.index_select(0, positions)
            targets = (
                train_t[0].index_select(0, positions),
                train_t[1].index_select(0, positions),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_feature_loss(student(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient norm.")
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            grad_norms.append(float(norm.item()))

        observed = evaluation_loss(student, monitor_x, monitor_t)
        if not torch.isfinite(torch.tensor(observed)):
            raise ValueError("Nonfinite monitor loss.")
        history.append({
            "epoch": epoch,
            "mean_train_loss": statistics.mean(epoch_losses),
            "monitor_loss": observed,
            "mean_preclip_gradient_norm": statistics.mean(grad_norms),
        })
        if observed < best_monitor:
            best_monitor, best_epoch = observed, epoch
            best_state = {name: t.detach().cpu().clone() for name, t in student.state_dict().items()}
        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"train={history[-1]['mean_train_loss']:.5f} "
            f"monitor={observed:.5f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    training_seconds = time.perf_counter() - started_training
    delta = float(torch.linalg.vector_norm(student.stem[0].weight.detach() - initial_weight))
    if best_state is None or delta <= 0.0:
        raise ValueError("No valid trained student state.")
    student.load_state_dict(best_state, strict=True)
    check_monitor = evaluation_loss(student, monitor_x, monitor_t)
    if abs(check_monitor - best_monitor) > 1e-5:
        raise ValueError("Restored best-epoch state does not reproduce its monitoring loss.")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as stream:
        torch.save(best_state, stream)

    output = {
        "experiment": "M3.11B compact student full normal training",
        "status": "NORMAL_ONLY_TRAINING_NOT_DEFECT_VALIDATED",
        "category": category,
        "protocol_sha256": probe["protocol_sha256"],
        "teacher_revision": probe["model_revision"],
        "teacher_weights_sha256": probe["weights_sha256"],
        "previous_smoke_sha256": SMOKE_SHA,
        "fit_report_sha256": FIT_SHA,
        "normal_pilot_sha256": PILOT_SHA,
        "script_sha256": sha256_file(__file__),
        "student_source_sha256": sha256_file(ROOT / "src/forgesight/stfpm_lite.py"),
        "protocol_source_sha256": sha256_file(ROOT / "src/forgesight/student_protocol.py"),
        "torch_version": torch.__version__,
        "seed": SEED,
        "train_paths": train_paths,
        "monitor_paths": monitor_paths,
        "excluded_reference_and_calibration_paths": sorted(set(reserved)),
        "student_parameter_count": param_count,
        "mini_batch_size": BATCH_SIZE,
        "epoch_count": EPOCHS,
        "learning_rate": LR,
        "optimizer": "AdamW",
        "feature_loss": "mean_two_scale_channel_normalized_squared_distance",
        "initial_monitor_loss": initial_monitor,
        "best_monitor_loss": best_monitor,
        "best_epoch": best_epoch,
        "monitor_loss_reproduced_after_restore": check_monitor,
        "stem_parameter_change_l2": delta,
        "history": history,
        "encoding_seconds": round(encoding_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "checkpoint_sha256": sha256_file(checkpoint),
        "teacher_frozen": True,
        "official_test_images_accessed": False,
        "real_defect_recall_measured": False,
        "threshold_calibrated": False,
        "monitor_is_disjoint_normal_training_pool": True,
    }
    with report_file.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print("\nFORGESIGHT — M3.11B STUDENT NORMAL TRAINING")
    print("Checkpoint:", checkpoint)
    print("Report:", report_file)
    print("SHA256:", sha256_file(report_file))
    print("Selected epoch:", best_epoch)
    print("Official test accessed: NO")
    print("Status: TRAINING COMPLETE — NO DEFECT CLAIMS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True, choices=CATEGORIES)
    args = p.parse_args()
    main(args.category)
