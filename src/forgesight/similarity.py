"""Lightweight visual similarity tools for dataset auditing."""

from PIL import Image


def difference_hash(image: Image.Image) -> int:
    """Create a 64-bit horizontal difference hash."""
    small = image.convert("L").resize(
        (9, 8),
        Image.Resampling.BILINEAR,
    )
    pixels = small.get_flattened_data()

    result = 0
    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]
            result = (result << 1) | int(left > right)

    return result


def orientation_hashes(image: Image.Image) -> tuple[int, int]:
    """Represent original and 180-degree orientations."""
    return (
        difference_hash(image),
        difference_hash(
            image.transpose(Image.Transpose.ROTATE_180)
        ),
    )


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two integer hashes."""
    return (a ^ b).bit_count()


def nearest_orientation_distance(
    reference: tuple[int, int],
    candidate: tuple[int, int],
) -> tuple[int, str]:
    """Compare original and 180-degree-aligned orientations."""
    original = hamming_distance(reference[0], candidate[0])
    rotated = hamming_distance(reference[0], candidate[1])

    if rotated < original:
        return rotated, "180_degrees"

    return original, "original"
