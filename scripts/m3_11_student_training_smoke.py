"""M3.11A: First real gradient-training feasibility experiment."""

import hashlib
import json
import os
import statistics
import time

import torch

from forgesight.stfpm_lite import (
    CompactStudent,
    normalized_feature_loss,
)

from m3_9_resnet_feature_probe import (
    ROOT,
    load_protocol,
    read_training_image,
    preprocess,
    sha256_file,
)
from m3_9_coreset_smoke import load_verified_model


FIT = ROOT / "reports/eda/m3_10_padim_smoke.json"
PILOT = ROOT / "reports/eda/m3_10_padim_normal_pilot_v1.json"

CHECKPOINT = ROOT / "models/m3_11_student_pcb1_smoke_v0.pt"
REPORT = ROOT / "reports/eda/m3_11_student_training_smoke_v0.json"

CATEGORY = "pcb1"
TRAIN_IMAGES = 12
MONITOR_IMAGES = 4
EPOCHS = 2
SEED = 42

FIT_SHA = "b352c560712ab0888d1bb67d174f0ce1d46db78616b460aca8d9b114c052eca1"
PILOT_SHA = "0e61cf5a6ad3ea158a0062be96e6d73d2766a0cd02deb4a676fd6ef7340ced64"


@torch.no_grad()
def teacher_features(model, image_tensor):
    """Extract frozen ResNet-18 layer2 and layer3 targets."""

    x = model.conv1(image_tensor)
    x = model.bn1(x)
    x = model.act1(x)
    x = model.maxpool(x)
    x = model.layer1(x)

    fine = model.layer2(x)
    coarse = model.layer3(fine)

    return fine.detach(), coarse.detach()


def prepare_examples(paths, teacher, mean, std):
    """Cache CPU inputs and teacher targets for the small pilot."""

    examples = []

    for path in paths:
        image, _ = read_training_image(path, CATEGORY)
        tensor, _ = preprocess(image, mean, std)

        with torch.no_grad():
            targets = teacher_features(teacher, tensor)

        examples.append((tensor.detach(), targets))

    return examples


@torch.no_grad()
def monitor_loss(student, examples):
    student.eval()

    losses = []

    for tensor, targets in examples:
        predictions = student(tensor)
        loss = normalized_feature_loss(predictions, targets)
        losses.append(float(loss.item()))

    return statistics.mean(losses)


def main():
    if CHECKPOINT.exists() or REPORT.exists():
        raise FileExistsError(
            "M3.11A artifacts already exist; refusing to overwrite."
        )

    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, os.cpu_count() or 1))

    protocol = load_protocol()
    teacher, probe = load_verified_model()

    if sha256_file(FIT) != FIT_SHA or sha256_file(PILOT) != PILOT_SHA:
        raise ValueError("Previous research artifact changed.")

    fit = json.loads(FIT.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))

    if (
        fit["protocol_sha256"] != probe["protocol_sha256"]
        or pilot["protocol_sha256"] != probe["protocol_sha256"]
        or fit["resnet_revision"] != probe["model_revision"]
    ):
        raise ValueError("Training provenance mismatch.")

    paths = fit["categories"][CATEGORY]["fitting_paths"]

    if len(paths) != 64 or len(set(paths)) != 64:
        raise ValueError("Unexpected normal fitting manifest.")

    training = paths[:TRAIN_IMAGES]
    monitoring = paths[
        TRAIN_IMAGES:TRAIN_IMAGES + MONITOR_IMAGES
    ]

    allowed = set(protocol["categories"][CATEGORY]["training"])
    calibration = set(pilot["categories"][CATEGORY]["calibration_paths"])
    references = set(pilot["categories"][CATEGORY]["fitting_paths"])

    if (
        not set(training + monitoring) <= allowed
        or set(training) & set(monitoring)
        or set(training + monitoring) & calibration
    ):
        raise ValueError("Training data separation failed.")

    # The 64-image PaDiM fitting list supplies these normal examples.
    if not set(training + monitoring) <= references:
        raise ValueError("Training-selection provenance mismatch.")

    teacher.eval()
    teacher.requires_grad_(False)

    mean = probe["image_normalization_mean"]
    std = probe["image_normalization_std"]

    print("Preparing frozen teacher features...", flush=True)

    train_examples = prepare_examples(
        training, teacher, mean, std
    )
    monitor_examples = prepare_examples(
        monitoring, teacher, mean, std
    )

    student = CompactStudent().cpu()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=0.0007,
        weight_decay=0.0001,
    )

    initial_monitor = monitor_loss(student, monitor_examples)

    best_monitor = float("inf")
    best_state = None
    best_epoch = None
    history = []

    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        student.train()
        losses = []
        gradient_norms = []

        for tensor, targets in train_examples:
            optimizer.zero_grad(set_to_none=True)

            predictions = student(tensor)

            loss = normalized_feature_loss(
                predictions, targets
            )

            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=1.0
            )

            if not torch.isfinite(grad_norm):
                raise ValueError("Nonfinite gradient norm.")

            optimizer.step()

            losses.append(float(loss.item()))
            gradient_norms.append(float(grad_norm.item()))

        observed_monitor = monitor_loss(
            student, monitor_examples
        )

        if not torch.isfinite(torch.tensor(observed_monitor)):
            raise ValueError("Nonfinite monitoring loss.")

        history.append({
            "epoch": epoch,
            "training_loss": statistics.mean(losses),
            "monitor_loss": observed_monitor,
            "mean_preclip_gradient_norm": statistics.mean(
                gradient_norms
            ),
        })

        if observed_monitor < best_monitor:
            best_monitor = observed_monitor
            best_epoch = epoch

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in student.state_dict().items()
            }

        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"training_loss={history[-1]['training_loss']:.5f} "
            f"monitor_loss={observed_monitor:.5f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start

    if best_state is None:
        raise ValueError("No valid training checkpoint selected.")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation protects the checkpoint from overwrites.
    with CHECKPOINT.open("xb") as output:
        torch.save(best_state, output)

    report = {
        "experiment": "M3.11A compact student-teacher gradient smoke",
        "status": "TRAINING_FEASIBILITY_ONLY",
        "category": CATEGORY,
        "seed": SEED,
        "teacher_model_revision": probe["model_revision"],
        "teacher_weights_sha256": probe["weights_sha256"],
        "development_protocol_sha256": probe["protocol_sha256"],
        "normal_fit_report_sha256": FIT_SHA,
        "normal_pilot_sha256": PILOT_SHA,
        "student_parameter_count": sum(
            p.numel() for p in student.parameters()
        ),
        "student_trained_from_random_initialization": True,
        "teacher_frozen": True,
        "train_images": training,
        "monitor_images": monitoring,
        "initial_monitor_loss": initial_monitor,
        "best_monitor_loss": best_monitor,
        "selected_epoch": best_epoch,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": 0.0007,
        "history": history,
        "training_seconds": round(elapsed, 3),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "official_test_images_accessed": False,
        "real_defect_recall_measured": False,
        "decision_threshold_calibrated": False,
    }

    with REPORT.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")

    print("\nFORGESIGHT — M3.11A GRADIENT TRAINING")
    print("=" * 50)
    print("Checkpoint:", CHECKPOINT)
    print("Report:", REPORT)
    print("SHA256:", sha256_file(REPORT))
    print("Official test images accessed: NO")
    print("Status: GRADIENT TRAINING FEASIBILITY VERIFIED")


if __name__ == "__main__":
    main()
