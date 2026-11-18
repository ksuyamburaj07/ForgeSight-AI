"""Pure-data checks for a paired, retrospective VisA image benchmark.

This module never opens images, labels from masks, or model checkpoints.
"""

import math
from pathlib import PurePosixPath


def verify_paired_manifest(records, previously_scored, category, expected_normal):
    """Verify stable image ordering and binary labels against prior benchmark."""
    if category not in ("pcb1", "pcb2", "pcb3", "pcb4"):
        raise ValueError("Unknown PCB category")
    if type(expected_normal) is not int or expected_normal not in (100, 101):
        raise ValueError("Unexpected category normal count")
    if not isinstance(records, list) or not isinstance(previously_scored, list):
        raise ValueError("Expected paired lists")
    if len(records) != expected_normal + 100 or len(previously_scored) != len(records):
        raise ValueError("Unexpected paired test population")
    seen = set()
    counts = {0: 0, 1: 0}
    for current, historic in zip(records, previously_scored):
        if not isinstance(current, dict) or not isinstance(historic, dict):
            raise ValueError("Invalid manifest record")
        label = current.get("label")
        name = current.get("image")
        path = PurePosixPath(name) if isinstance(name, str) else None
        if label not in ("normal", "anomaly") or path is None:
            raise ValueError("Invalid image name or label")
        if (
            path.is_absolute() or ".." in path.parts or len(path.parts) != 5
            or tuple(path.parts[:4]) != (
                category, "Data", "Images", "Normal" if label == "normal" else "Anomaly"
            )
            or path.suffix != ".JPG"
            or name in seen
        ):
            raise ValueError("Invalid/duplicated test image")
        binary = int(label == "anomaly")
        if historic.get("image") != name or type(historic.get("label")) is not int or historic["label"] != binary:
            raise ValueError("Historical benchmark manifest differs")
        seen.add(name)
        counts[binary] += 1
    if counts[0] != expected_normal or counts[1] != 100:
        raise ValueError("Incorrect normal/anomaly category composition")
    return True


def verify_normal_regression(observed, expected, *, tolerance=1e-4):
    """Ensure frozen scorer still reproduces a known normal validation image."""
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in (observed, expected)):
        raise ValueError("Invalid normal regression score")
    if not math.isclose(observed, expected, abs_tol=tolerance, rel_tol=tolerance):
        raise ValueError("Student score differs from the normal calibration pilot")
    return True
