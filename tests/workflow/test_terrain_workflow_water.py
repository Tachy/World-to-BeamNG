"""Tests für TerrainWorkflow.export_water() und _build_water() mit den echten Config-/Vorlagendaten."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from shapely.geometry import box

from world_to_beamng import config
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _RecordingItems:
    def __init__(self):
        self.objects = {}

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), scale=(1, 1, 1), **fields):
        self.objects[name] = {"class": item_class, "position": list(position), "scale": list(scale), **fields}


def _nodes(n=5, z=100.0):
    return [[float(i * 10), 0.0, z - i * 0.2, 2.5, 1.0, 0.0, 0.0, 1.0] for i in range(n)]


def _export(water):
    stub = SimpleNamespace(items=_RecordingItems())
    count = TerrainWorkflow.export_water(stub, {"water": water})
    return count, stub.items.objects


def test_streams_become_river_objects_with_the_template_render_settings():
    count, objects = _export({"rivers": [{"name": "river_0", "waterway": "stream", "nodes": _nodes()}], "ponds": []})

    river = objects["river_0"]
    assert count == 1
    assert river["class"] == "River"
    assert river["position"] == river["nodes"][0][:3]
    assert len(river["nodes"]) == 5
    # nur core-Texturen (immer vorhanden, nichts zu vendoren)
    for key in ("rippleTex", "foamTex", "depthGradientTex"):
        assert river[key].startswith("core/"), f"{key}: {river[key]}"
    assert river["cubemap"] == "GreySkyCubemap"  # von der Engine selbst angelegt


def test_ponds_become_water_blocks_with_an_engine_cubemap():
    block = {"position": [10.0, 20.0, 281.15], "scale": [6.0, 4.0, 3.0], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}

    count, objects = _export({"rivers": [], "ponds": [{"name": "pond_0", "blocks": [block, dict(block)]}]})

    assert count == 2
    assert set(objects) == {"pond_0_0", "pond_0_1"}
    water_block = objects["pond_0_0"]
    assert water_block["class"] == "WaterBlock"
    assert water_block["position"] == [10.0, 20.0, 281.15]
    assert water_block["scale"] == [6.0, 4.0, 3.0]
    # die Cubemap der Vorlage (cubemap_river_reflection) ist level-spezifisch und würde fehlen
    assert water_block["cubemap"] == config.WATER_POND_CUBEMAP == "DefaultSkyCubemap"


def test_nothing_is_exported_when_water_is_disabled_or_empty(monkeypatch):
    assert _export({"rivers": [], "ponds": []})[0] == 0
    monkeypatch.setattr(config, "WATER_ENABLED", False)
    assert _export({"rivers": [{"name": "r", "waterway": "stream", "nodes": _nodes()}], "ponds": []})[0] == 0


def test_build_water_clips_to_the_terrain_and_uses_the_heightmap():
    stub = SimpleNamespace()
    height_at = lambda x, y: np.full_like(np.asarray(x, float), 250.0)
    # Bach quer durch das Terrain (-1000..1000), Teich am Rand; Punkte als lat/lon nahe dem Ursprung
    from world_to_beamng.osm.landuse_polygons import make_local_transform

    offset = (412000.0, 5297000.0)
    to_local = make_local_transform(offset)
    ax, ay = to_local([{"lat": 47.83, "lon": 7.68}])[0]
    osm = [
        {
            "type": "way",
            "id": 1,
            "tags": {"waterway": "stream"},
            "geometry": [{"lat": 47.83, "lon": 7.68}, {"lat": 47.8301, "lon": 7.6803}, {"lat": 47.8302, "lon": 7.6806}],
        },
        {"type": "way", "id": 2, "tags": {"waterway": "stream", "tunnel": "culvert"},
         "geometry": [{"lat": 47.83, "lon": 7.68}, {"lat": 47.8301, "lon": 7.6803}]},
    ]
    pond = {"osm_tags": {"natural": "water"}, "geometry": box(ax - 30, ay - 30, ax - 10, ay - 10)}
    grass = {"osm_tags": {"landuse": "meadow"}, "geometry": box(ax, ay, ax + 10, ay + 10)}
    bounds = (ax - 500, ax + 500, ay - 500, ay + 500)

    water = TerrainWorkflow._build_water(stub, osm, [pond, grass], offset, height_at, bounds)

    assert len(water["rivers"]) == 1  # der Durchlass ist ausgeschlossen
    nodes = water["rivers"][0]["nodes"]
    assert all(abs(n[2] - 250.2) < 1e-6 for n in nodes)  # Rinnenboden 250 + Wasserstand 0,2
    assert len(water["ponds"]) == 1  # nur die Wasserfläche, nicht die Wiese
    assert all(abs(b["position"][2] - 250.0) < 1e-6 for b in water["ponds"][0]["blocks"])  # Randhöhe des Teichs


def test_dry_detention_basins_get_no_water_but_a_small_wet_basin_does():
    stub = SimpleNamespace()
    height_at = lambda x, y: np.full_like(np.asarray(x, float), 250.0)
    detention = {"osm_tags": {"landuse": "basin", "basin": "detention"}, "geometry": box(0, 0, 200, 90)}
    small_basin = {"osm_tags": {"landuse": "basin", "name": "Rückhaltebecken"}, "geometry": box(80, 30, 100, 44)}

    water = TerrainWorkflow._build_water(stub, [], [detention, small_basin], (0.0, 0.0), height_at, (-500, 500, -500, 500))

    assert len(water["ponds"]) == 1
    xs = [b["position"][0] for b in water["ponds"][0]["blocks"]]
    assert 70 < min(xs) and max(xs) < 110  # nur das kleine Becken, nicht die 200 m breite Fläche


def test_streams_end_at_the_pond_shore():
    stub = SimpleNamespace()
    height_at = lambda x, y: np.full_like(np.asarray(x, float), 250.0)
    from world_to_beamng.osm.landuse_polygons import make_local_transform

    offset = (412000.0, 5297000.0)
    to_local = make_local_transform(offset)
    line = [{"lat": 47.83, "lon": 7.68}, {"lat": 47.83, "lon": 7.6812}]  # gut 90 m West-Ost
    (x0, y0), (x1, _) = to_local(line)
    mid = (x0 + x1) / 2
    pond = {"osm_tags": {"natural": "water"}, "geometry": box(mid - 15, y0 - 10, mid + 15, y0 + 10)}
    osm = [{"type": "way", "id": 1, "tags": {"waterway": "stream"}, "geometry": line}]

    water = TerrainWorkflow._build_water(stub, osm, [pond], offset, height_at, (x0 - 500, x1 + 500, y0 - 500, y0 + 500))

    assert len(water["rivers"]) == 2  # oberhalb und unterhalb des Teichs
    for river in water["rivers"]:
        for node in river["nodes"]:
            assert not (mid - 15 + 0.5 < node[0] < mid + 15 - 0.5)  # kein Knoten im Teich (0,5 m Toleranz: seitliches Einrasten)
