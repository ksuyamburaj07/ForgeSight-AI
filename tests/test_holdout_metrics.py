"""Synthetic tests; no VisA test images are accessed."""
import numpy as np
import pytest

from forgesight.holdout_metrics import (
    PIXEL_BINS, add_pixels, empty_pixel_histograms, image_metrics, pixel_metrics,
)


def test_perfect_image_ranking():
    result = image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert result["image_AUROC"] == pytest.approx(1.0)
    assert result["image_average_precision"] == pytest.approx(1.0)
    assert (result["TP"], result["FP"], result["TN"], result["FN"]) == (2, 0, 2, 0)


def test_tied_scores_are_not_artificially_discriminative():
    result = image_metrics([0, 1], [0.5, 0.5], 0.5)
    assert result["image_AUROC"] == pytest.approx(0.5)
    assert result["image_average_precision"] == pytest.approx(0.5)
    assert result["FN"] == 1


def test_inverted_image_ranking():
    result = image_metrics([0, 1], [0.9, 0.1], 0.5)
    assert result["image_AUROC"] == pytest.approx(0.0)
    assert result["image_average_precision"] == pytest.approx(0.5)


def test_missing_class_rejected():
    with pytest.raises(ValueError):
        image_metrics([0, 0], [0.1, 0.2], 0.5)


def test_pixel_histogram_perfect_separation():
    pos, neg = empty_pixel_histograms()
    map_values = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    mask = np.array([[False, False], [True, True]])
    add_pixels(pos, neg, map_values, mask)
    result = pixel_metrics(pos, neg)
    assert pos.sum() == neg.sum() == 2
    assert result["pixel_AUROC_binned"] == pytest.approx(1.0)
    assert result["pixel_average_precision_binned"] == pytest.approx(1.0)
    assert result["pixel_histogram_bins"] == PIXEL_BINS


def test_pixel_histogram_all_tied():
    pos, neg = empty_pixel_histograms()
    add_pixels(pos, neg, np.full((2, 2), 0.2, dtype=np.float32),
               np.array([[True, False], [True, False]]))
    result = pixel_metrics(pos, neg)
    assert result["pixel_AUROC_binned"] == pytest.approx(0.5)
    assert result["pixel_average_precision_binned"] == pytest.approx(0.5)


def test_nonfinite_pixel_scores_rejected():
    pos, neg = empty_pixel_histograms()
    with pytest.raises(ValueError):
        add_pixels(pos, neg, np.array([[np.nan]], dtype=np.float32), np.array([[True]]))


def test_out_of_range_pixel_scores_rejected():
    pos, neg = empty_pixel_histograms()
    with pytest.raises(ValueError):
        add_pixels(pos, neg, np.array([[2.5]], dtype=np.float32), np.array([[True]]))


def test_mask_score_shape_mismatch_rejected():
    pos, neg = empty_pixel_histograms()
    with pytest.raises(ValueError):
        add_pixels(pos, neg, np.zeros((2, 2), dtype=np.float32), np.ones((2, 3), dtype=bool))
