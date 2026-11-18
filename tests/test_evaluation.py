"""Tests for validation-result consistency checking."""

import copy

import pytest

from forgesight.evaluation import collect_normal_false_alarms


def example_report():
    return {
        "categories": {
            "pcb1": {
                "threshold": 50.0,
                "validation_normal_count": 2,
                "validation_false_alarms": 1,
                "validation_results": [
                    {
                        "image": "pcb1/a.jpg",
                        "score": 40.0,
                        "flagged_for_review": False,
                        "phase_response": 0.9,
                        "rotation_degrees": 0,
                    },
                    {
                        "image": "pcb1/b.jpg",
                        "score": 60.0,
                        "flagged_for_review": True,
                        "phase_response": 0.7,
                        "rotation_degrees": 180,
                    },
                ],
            }
        }
    }


def test_collects_only_flagged_normal_images():
    flags = collect_normal_false_alarms(example_report())

    assert len(flags) == 1
    assert flags[0]["image"] == "pcb1/b.jpg"
    assert flags[0]["score"] == 60.0


def test_inconsistent_decision_is_rejected():
    report = copy.deepcopy(example_report())
    report["categories"]["pcb1"]["validation_results"][0][
        "flagged_for_review"
    ] = True

    with pytest.raises(ValueError):
        collect_normal_false_alarms(report)


def test_duplicate_validation_image_is_rejected():
    report = copy.deepcopy(example_report())
    report["categories"]["pcb1"]["validation_results"][1][
        "image"
    ] = "pcb1/a.jpg"

    with pytest.raises(ValueError):
        collect_normal_false_alarms(report)
