"""Explicit field-of-view controls for industrial image processing."""

from PIL import Image


def center_crop_retained_fraction(
    width: int,
    height: int,
    shortest: int = 256,
    crop: int = 224,
) -> float:
    """Estimate original-frame area retained by resize + center crop."""

    if any(
        type(v) is not int or v <= 0
        for v in (width, height, shortest, crop)
    ):
        raise ValueError("Dimensions must be positive integers.")

    if crop > shortest:
        raise ValueError("Crop cannot exceed resized shortest edge.")

    scale = shortest / min(width, height)

    resized_width = width * scale
    resized_height = height * scale

    retained_width = min(1.0, crop / resized_width)
    retained_height = min(1.0, crop / resized_height)

    return retained_width * retained_height


def letterbox_rgb(
    image: Image.Image,
    side: int = 224,
    fill: tuple[int, int, int] = (124, 116, 104),
):
    """Preserve the complete RGB frame using aspect-aware padding."""

    if image.mode != "RGB" or min(image.size) == 0:
        raise ValueError("Expected a nonempty RGB image.")

    if type(side) is not int or side < 32:
        raise ValueError("Invalid output dimensions.")

    width, height = image.size
    scale = min(side / width, side / height)

    new_width = max(1, min(side, round(width * scale)))
    new_height = max(1, min(side, round(height * scale)))

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )

    canvas = Image.new("RGB", (side, side), fill)

    left = (side - new_width) // 2
    top = (side - new_height) // 2

    canvas.paste(resized, (left, top))

    metadata = {
        "original_width": width,
        "original_height": height,
        "content_width": new_width,
        "content_height": new_height,
        "padding_left": left,
        "padding_top": top,
        "padding_right": side - left - new_width,
        "padding_bottom": side - top - new_height,
    }

    return canvas, metadata
