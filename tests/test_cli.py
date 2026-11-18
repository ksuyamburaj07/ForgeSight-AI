import json

from PIL import Image

from forgesight.cli import main


def test_cli_reports_real_input_metadata(tmp_path, capsys):
    path = tmp_path / "sample.png"
    Image.new("RGB", (20, 12)).save(path)
    assert main(["inspect", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["width"] == 20
    assert output["height"] == 12
    assert output["inference_performed"] is False
