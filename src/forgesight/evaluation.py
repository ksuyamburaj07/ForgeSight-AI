"""Utilities for reviewing normal-image inspection results."""

from forgesight.decision import flag_for_review


def collect_normal_false_alarms(report: dict) -> list[dict]:
    """Validate stored decisions and collect flagged normal examples."""

    flags = []

    for category, data in report["categories"].items():
        threshold = float(data["threshold"])
        observations = data["validation_results"]
        seen = set()
        observed_flags = 0

        if len(observations) != data["validation_normal_count"]:
            raise ValueError(f"Validation count mismatch: {category}")

        for observation in observations:
            path = observation["image"]

            if path in seen:
                raise ValueError(f"Duplicate validation image: {path}")

            seen.add(path)

            score = float(observation["score"])
            expected = flag_for_review(score, threshold)

            if expected != observation["flagged_for_review"]:
                raise ValueError(
                    f"Decision inconsistent with threshold: {path}"
                )

            if expected:
                observed_flags += 1
                flags.append({
                    "category": category,
                    "image": path,
                    "score": score,
                    "threshold": threshold,
                    "phase_response": observation["phase_response"],
                    "rotation_degrees": observation["rotation_degrees"],
                })

        if observed_flags != data["validation_false_alarms"]:
            raise ValueError(f"False-alarm count mismatch: {category}")

    return flags
