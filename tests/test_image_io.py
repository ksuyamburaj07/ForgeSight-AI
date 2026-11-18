from hashlib import sha256

from PIL import Image
import pytest

from forgesight.image_io import ImageInputError, inspect_image


def test_png_validated_without_claiming_inference(tmp_path):
    path = tmp_path / "board.png"
    Image.new("RGB", (96, 64), (14, 36, 64)).save(path)
    result = inspect_image(path)
    assert result.detected_format == "PNG"
    assert (result.width, result.height) == (96, 64)
    assert result.source_mode == "RGB"
    assert result.sha256 == sha256(path.read_bytes()).hexdigest()
    assert result.inference_performed is False


def test_content_format_is_checked_not_extension(tmp_path):
    path = tmp_path / "not_really_a_jpeg.jpg"
    Image.new("RGB", (8, 8)).save(path, format="PNG")
    assert inspect_image(path).detected_format == "PNG"


def test_unsupported_bmp_is_rejected(tmp_path):
    path = tmp_path / "board.bmp"
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(ImageInputError, match="Unsupported"):
        inspect_image(path)


def test_corrupt_file_is_rejected(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an actual image")
    with pytest.raises(ImageInputError, match="decoded"):
        inspect_image(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ImageInputError, match="regular file"):
        inspect_image(tmp_path / "missing.png")


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")
    with pytest.raises(ImageInputError, match="between 1"):
        inspect_image(path)


def test_oversized_file_is_rejected_before_decode(tmp_path, monkeypatch):
    import forgesight.image_io as io
    monkeypatch.setattr(io, "MAX_FILE_BYTES", 10)
    path = tmp_path / "too_big.png"
    path.write_bytes(b"x" * 11)
    with pytest.raises(ImageInputError, match="between 1"):
        inspect_image(path)


def test_image_is_not_modified(tmp_path):
    path = tmp_path / "board.png"
    Image.new("RGB", (20, 10)).save(path)
    before = path.read_bytes()
    inspect_image(path)
    assert path.read_bytes() == before
