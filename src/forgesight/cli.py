"""Command-line interface: presentation only, no ML logic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .image_io import ImageInputError, inspect_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forgesight")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect_cmd = subcommands.add_parser("inspect", help="Validate a local image")
    inspect_cmd.add_argument("image", help="Path to a PNG, JPEG or WebP file")
    args = parser.parse_args(argv)

    if args.command == "inspect":
        try:
            record = inspect_image(args.image)
        except ImageInputError as exc:
            parser.exit(status=2, message=f"Input rejected: {exc}\n")
        print(json.dumps(asdict(record), indent=2))
    return 0
