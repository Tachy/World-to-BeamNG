"""
Regression test: io/lod2.py::cache_lod2_buildings() transforms the BBox via the central
source CRS (world_to_beamng.geometry.coordinates) instead of a locally hardcoded, differing
EPSG:32632 (previous bug - everything else in the pipeline uses EPSG:25832).
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.geometry import coordinates
from world_to_beamng.io import lod2
from world_to_beamng.io.lod2 import cache_lod2_buildings


def test_source_no_longer_hardcodes_a_second_epsg_code():
    source = inspect.getsource(lod2)

    assert "32632" not in source
    assert "transformer_to_utm" in inspect.getsource(cache_lod2_buildings)


def test_cache_lod2_buildings_uses_the_central_transformer_and_hits_cache_without_touching_lod2_dir(tmp_path):
    coordinates.set_source_crs(25832)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    height_hash = "regressiontest"
    (cache_dir / f"lod2_{height_hash}.pkl").write_bytes(b"")  # cache hit, no real LoD2 directory needed

    result = cache_lod2_buildings(
        lod2_dir=str(tmp_path / "does_not_exist"),
        bbox=(48.0, 8.0, 48.1, 8.1),
        local_offset=(0.0, 0.0, 0.0),
        cache_dir=str(cache_dir),
        height_hash=height_hash,
    )

    assert Path(result) == cache_dir / f"lod2_{height_hash}.pkl"
