"""Tests für die LevelInfo-Basiszeile (Sichtweite/Nebel) des ItemManagers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng import config
from world_to_beamng.managers.item_manager import ItemManager


def _level_info():
    return next(line for line in ItemManager.OTHER_BASE_LINES if line["class"] == "LevelInfo")


def test_level_info_sets_visible_distance_from_config():
    # Ohne visibleDistance greift der BeamNG-Standard (~1 km): der Rest wird geclippt.
    info = _level_info()

    assert info["visibleDistance"] == config.LEVEL_VISIBLE_DISTANCE
    assert info["visibleDistance"] > 5000  # mindestens so weit wie die kleinsten Original-Level


def test_level_info_sets_fog_density_from_config():
    assert _level_info()["fogDensity"] == config.LEVEL_FOG_DENSITY
