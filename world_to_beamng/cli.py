"""
Command line of world_to_beamng.py.

Parsed before anything else of the package is imported: world_to_beamng.config fixes the log level on import (from the
LOG_LEVEL environment variable), so --loglevel sets that variable first and thereby overrides it.
"""

import argparse
import os
from typing import Optional, Sequence

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="world_to_beamng.py",
        description="Exports elevation data, OSM roads and buildings as a BeamNG.drive level.",
    )
    parser.add_argument(
        "--loglevel",
        type=str.upper,
        choices=LOG_LEVELS,
        metavar="LOGLEVEL",
        help=f"log level ({', '.join(LOG_LEVELS)}, case-insensitive); overrides the LOG_LEVEL environment variable",
    )
    return parser.parse_args(argv)


def apply_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses the command line and applies the options that must be set before the config is imported."""
    args = parse_args(argv)
    if args.loglevel:
        os.environ["LOG_LEVEL"] = args.loglevel
    return args
