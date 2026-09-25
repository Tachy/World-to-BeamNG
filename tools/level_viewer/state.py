"""Window size, camera and layer visibility, stored between runs in tools/level_viewer.cfg (JSON, not committed)."""

import json
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).resolve().parent.parent / "level_viewer.cfg"


def load_state(path: Path = CONFIG_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict, path: Path = CONFIG_PATH) -> None:
    try:
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def camera_to_dict(camera) -> dict:
    return {
        "position": list(camera.position),
        "focal_point": list(camera.focal_point),
        "view_up": list(camera.up),
        "view_angle": camera.view_angle,
    }


def apply_camera(plotter, data: Optional[dict]) -> bool:
    """Restores a saved camera; False if there is none."""
    if not data or "position" not in data:
        return False
    plotter.camera_position = [data["position"], data["focal_point"], data.get("view_up", [0, 0, 1])]
    plotter.camera.view_angle = float(data.get("view_angle", 30.0))
    plotter.renderer.ResetCameraClippingRange()
    return True
