"""Tests for the image similarity helpers."""

from PIL import Image, ImageDraw

from forgesight.similarity import (
    difference_hash,
    hamming_distance,
    nearest_orientation_distance,
    orientation_hashes,
)


def asymmetric_image():
    image = Image.new("RGB", (48, 32), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 3, 12, 17), fill="white")
    return image


def test_hash_is_deterministic():
    image = asymmetric_image()
    assert difference_hash(image) == difference_hash(image)


def test_hash_is_64_bit():
    result = difference_hash(asymmetric_image())
    assert 0 <= result < 2**64


def test_hamming_distance():
    assert hamming_distance(0b1011, 0b0011) == 1
    assert hamming_distance(42, 42) == 0


def test_rotated_image_is_recognized():
    image = asymmetric_image()
    rotated = image.transpose(Image.Transpose.ROTATE_180)

    distance, _ = nearest_orientation_distance(
        orientation_hashes(image),
        orientation_hashes(rotated),
    )

    assert distance == 0
