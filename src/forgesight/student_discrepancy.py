"""STFPM-lite two-scale feature discrepancy; normal-only development stage.

The teacher and student must have identical pretrained-feature geometry.
The anomaly score is the mean of the five highest valid 28x28 discrepancies.
"""

import numpy as np
import torch
from torch.nn import functional as F


def feature_discrepancy_map(teacher, student, valid_mask):
    """Return a (28, 28) finite, nonnegative anomaly map; padding is zero.

    Each scale uses squared Euclidean distance between unit-length feature
    vectors. The coarse map is bilinearly upsampled to 28x28, and the two
    scales are averaged. All four tensors must be single-image feature maps.
    """
    if not isinstance(teacher, (tuple, list)) or not isinstance(student, (tuple, list)):
        raise ValueError("Expected two ordered teacher/student feature stages.")
    if len(teacher) != 2 or len(student) != 2:
        raise ValueError("Expected exactly two feature stages.")
    mask = np.asarray(valid_mask)
    if mask.shape != (28, 28) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Expected a nonempty boolean (28, 28) content mask.")

    shapes = ((1, 128, 28, 28), (1, 256, 14, 14))
    level_maps = []
    for level, (expected, t, s) in enumerate(zip(shapes, teacher, student)):
        if not isinstance(t, torch.Tensor) or not isinstance(s, torch.Tensor):
            raise ValueError("Feature stages must be PyTorch tensors.")
        if tuple(t.shape) != expected or tuple(s.shape) != expected:
            raise ValueError(f"Unexpected feature geometry at level {level}.")
        if not (t.is_floating_point() and s.is_floating_point()):
            raise ValueError("Feature tensors must be floating point.")
        if t.device != s.device or t.dtype != s.dtype:
            raise ValueError("Teacher/student dtype or device mismatch.")
        if not torch.isfinite(t).all().item() or not torch.isfinite(s).all().item():
            raise ValueError("Nonfinite teacher or student features.")
        target = F.normalize(t.detach(), p=2, dim=1, eps=1e-6)
        predicted = F.normalize(s.detach(), p=2, dim=1, eps=1e-6)
        difference = (predicted - target).square().sum(dim=1, keepdim=True)
        if level == 1:
            difference = F.interpolate(
                difference, size=(28, 28), mode="bilinear", align_corners=False
            )
        level_maps.append(difference)

    combined = 0.5 * (level_maps[0] + level_maps[1])
    result = combined[0, 0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Nonfinite or negative student-teacher discrepancy.")
    # Only actual PCB image positions contribute to threshold calibration.
    return np.where(mask, result, 0).astype(np.float32)


def topk_discrepancy_score(anomaly_map, valid_mask, top_k=5):
    """Mean of the strongest valid discrepancy positions; higher is stranger."""
    values = np.asarray(anomaly_map)
    mask = np.asarray(valid_mask)
    if (
        values.shape != (28, 28)
        or mask.shape != (28, 28)
        or mask.dtype != np.bool_
        or type(top_k) is not int
        or top_k < 1
        or int(mask.sum()) < top_k
    ):
        raise ValueError("Invalid student-teacher scoring geometry.")
    selected = values[mask]
    if not np.issubdtype(selected.dtype, np.number):
        raise ValueError("Discrepancy map must be numeric.")
    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("Invalid student-teacher scores.")
    kth = len(selected) - top_k
    strongest = np.partition(selected, kth)[kth:]
    return float(np.mean(strongest, dtype=np.float64))
