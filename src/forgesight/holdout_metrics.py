"""Frozen, label-aware image metrics and fixed-bin pixel metrics."""
import numpy as np

PIXEL_BINS = 4096
PIXEL_RANGE = (0.0, 2.0)


def image_metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not y.size:
        raise ValueError("Image labels and scores must be aligned 1-D arrays")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("Invalid image labels or scores")
    if not np.isfinite(threshold):
        raise ValueError("Nonfinite threshold")
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("Both image classes are required")
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = fp = 0
    roc = ap = prev_fpr = prev_tpr = 0.0
    start = 0
    while start < len(ys):
        end = start + 1
        while end < len(ys) and ss[end] == ss[start]:
            end += 1
        group_p = int(ys[start:end].sum())
        group_n = (end - start) - group_p
        tp += group_p
        fp += group_n
        tpr = tp / positives
        fpr = fp / negatives
        roc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        ap += (group_p / positives) * (tp / (tp + fp))
        prev_fpr, prev_tpr = fpr, tpr
        start = end
    predicted = s > threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = positives - tp
    tn = negatives - fp
    return {
        "image_AUROC": float(roc),
        "image_average_precision": float(ap),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "anomaly_recall_at_frozen_threshold": tp / positives,
        "normal_specificity_at_frozen_threshold": tn / negatives,
        "false_positive_rate_at_frozen_threshold": fp / negatives,
    }


def empty_pixel_histograms():
    return np.zeros(PIXEL_BINS, dtype=np.int64), np.zeros(PIXEL_BINS, dtype=np.int64)


def add_pixels(positive_hist, negative_hist, scores, ground_truth):
    arr = np.asarray(scores, dtype=np.float32)
    truth = np.asarray(ground_truth, dtype=bool)
    if arr.ndim != 2 or arr.size == 0 or arr.shape != truth.shape:
        raise ValueError("Pixel scores and masks must have matching nonempty dimensions")
    if not np.isfinite(arr).all() or np.any(arr < -1e-6) or np.any(arr > 2 + 1e-6):
        raise ValueError("Pixel distance outside predeclared cosine range [0,2]")
    indices = np.minimum((np.clip(arr, 0, 2) * (PIXEL_BINS / 2)).astype(np.int32), PIXEL_BINS - 1)
    positive_hist += np.bincount(indices[truth], minlength=PIXEL_BINS)
    negative_hist += np.bincount(indices[~truth], minlength=PIXEL_BINS)


def pixel_metrics(positive_hist, negative_hist):
    positives = int(positive_hist.sum())
    negatives = int(negative_hist.sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("Need both positive and negative annotated pixels")
    positive = positive_hist[::-1].astype(np.float64)
    negative = negative_hist[::-1].astype(np.float64)
    tps = np.cumsum(positive)
    fps = np.cumsum(negative)
    tpr = np.r_[0.0, tps / positives]
    fpr = np.r_[0.0, fps / negatives]
    auc = float(np.trapezoid(tpr, fpr))
    precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) > 0)
    ap = float(np.sum((positive / positives) * precision))
    return {
        "pixel_AUROC_binned": auc,
        "pixel_average_precision_binned": ap,
        "positive_pixel_count": positives,
        "negative_pixel_count": negatives,
        "pixel_histogram_bins": PIXEL_BINS,
        "pixel_distance_range": list(PIXEL_RANGE),
    }
