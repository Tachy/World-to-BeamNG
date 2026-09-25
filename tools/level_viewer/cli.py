"""Command line: ``python -m tools.level_viewer [options]``."""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER_KEYS = {"terrain": "g", "roads": "a", "markings": "m", "water": "o", "structures": "b", "horizon": "h",
              "forest": "c", "zones": "z", "labels": "n", "debug": "d"}


@dataclass
class ViewerOptions:
    level_dir: Path
    terrain_step: int = 4
    full_photo: bool = False
    debug_network: Optional[Path] = None
    layer_overrides: Dict[str, bool] = field(default_factory=dict)  # layer key -> visible


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.level_viewer", description="Developer viewer for an exported World-to-BeamNG level.")
    parser.add_argument("--level-dir", type=Path, help="level folder (default: config.BEAMNG_DIR)")
    parser.add_argument("--terrain-step", type=int, default=4, help="use every n-th terrain sample (default 4; 1 = full resolution)")
    parser.add_argument("--full-photo", action="store_true", help="texture the terrain with the full aerial photo tiles instead of the minimap")
    parser.add_argument("--show", default="", help=f"comma-separated layers to show ({', '.join(LAYER_KEYS)})")
    parser.add_argument("--hide", default="", help="comma-separated layers to hide")
    parser.add_argument("--debug-network", type=Path, default=REPO_ROOT / "cache" / "debug_network.json", help="debug network JSON")
    parser.add_argument("--screenshot", type=Path, help="render off-screen, save a PNG and exit")
    return parser.parse_args(argv)


def _layer_overrides(show: str, hide: str) -> Dict[str, bool]:
    overrides = {}
    for names, visible in ((show, True), (hide, False)):
        for name in filter(None, (n.strip().lower() for n in names.split(","))):
            if name not in LAYER_KEYS:
                raise SystemExit(f"Unknown layer '{name}' (known: {', '.join(LAYER_KEYS)})")
            overrides[LAYER_KEYS[name]] = visible
    return overrides


def main(argv=None) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    args = parse_args(argv)
    from .level_data import default_level_dir

    level_dir = args.level_dir or default_level_dir()
    if not (level_dir / "main").is_dir():
        print(f"Not an exported level: {level_dir}", file=sys.stderr)
        return 1
    options = ViewerOptions(
        level_dir=level_dir,
        terrain_step=max(1, args.terrain_step),
        full_photo=args.full_photo,
        debug_network=args.debug_network,
        layer_overrides=_layer_overrides(args.show, args.hide),
    )
    from .app import LevelViewer

    viewer = LevelViewer(options, off_screen=args.screenshot is not None)
    if args.screenshot is not None:
        viewer.screenshot(args.screenshot)
        print(f"Screenshot saved: {args.screenshot} (loaded in {viewer.load_seconds:.1f} s)")
    else:
        viewer.show()
    return 0
