"""Exact image-level metrics for a labelled retrospective benchmark."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_manifest(records, category, expected_count):
    """Validate metadata before decoding any benchmark images."""

    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError("Unexpected benchmark record count.")

    seen = set()
    labels = []

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid benchmark record.")

        path = record.get("image")
        label = record.get("label")

        if type(path) is not str or type(label) is not int:
            raise ValueError("Invalid image path or label.")

        if label not in (0, 1):
            raise ValueError("Benchmark labels must be binary.")

        parts = path.split("/")

        if (
            len(parts) != 5
            or parts[:3] != [category, "Data", "Images"]
            or parts[3] not in ("Normal", "Anomaly")
            or not parts[4].lower().endswith((".jpg", ".jpeg", ".png"))
            or label != int(parts[3] == "Anomaly")
            or path in seen
        ):
            raise ValueError("Invalid or duplicated benchmark image.")

        seen.add(path)
        labels.append(label)

    if labels.count(1) != 100:
        raise ValueError("Expected 100 anomalous images per PCB.")

    if labels.count(0) != expected_count - 100:
        raise ValueError("Unexpected number of normal images.")

    return True


def image_metrics(labels, scores, threshold):
    """Calculate exact image discrimination and frozen operating-point metrics."""

    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)

    if (
        y.ndim != 1
        or s.ndim != 1
        or y.shape != s.shape
        or len(y) == 0
        or not np.isin(y, (0, 1)).all()
        or len(np.unique(y)) != 2
        or not np.isfinite(s).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("Invalid metric inputs.")

    predicted = s > threshold

    tp = int(np.sum((y == 1) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))

    return {
        "image_AUROC": float(roc_auc_score(y, s)),
        "image_average_precision": float(
            average_precision_score(y, s)
        ),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "anomaly_recall_at_frozen_threshold": tp / (tp + fn),
        "normal_specificity_at_frozen_threshold": tn / (tn + fp),
    }
