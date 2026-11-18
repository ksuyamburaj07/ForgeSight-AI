"""Validate the spatial structure of Vision Transformer features."""


def patch_grid(height: int, width: int, patch_size: int = 14):
    """Calculate the spatial patch grid for a given image size."""

    if (
        type(height) is not int
        or type(width) is not int
        or type(patch_size) is not int
        or min(height, width, patch_size) <= 0
    ):
        raise ValueError("Image and patch dimensions must be positive integers.")

    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch size.")

    return height // patch_size, width // patch_size


def validate_dinov2_tokens(
    token_shape: tuple,
    height: int,
    width: int,
    patch_size: int = 14,
):
    """Check batch, class token, spatial tokens and embedding width."""

    rows, cols = patch_grid(height, width, patch_size)

    if (
        len(token_shape) != 3
        or token_shape[0] != 1
        or token_shape[1] != rows * cols + 1
        or token_shape[2] != 384
    ):
        raise ValueError(
            "Unexpected DINOv2-small token layout."
        )

    return rows, cols
