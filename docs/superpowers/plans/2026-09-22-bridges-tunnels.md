# Brücken, Tunnel & Galerien Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OSM-Ways mit `bridge=*`/`tunnel=*` bekommen ein plausibles Höhenprofil und werden als eigene 3D-Objekte (Brücken-Deck+Pfeiler, Tunnel-Röhre+Portale, offene Lawinengalerien) statt als normale, ins Terrain eingebettete `DecalRoad`-Straßen exportiert.

**Architecture:** Eine gemeinsame Klassifizierung (`geometry/road_structures.py`) erkennt Brücken/Tunnel/Galerien anhand ihrer OSM-Tags. Das Höhenprofil dieser Straßen wird in `geometry/polygon.py` durch lineare Interpolation zwischen den (unveränderten) Anschlusspunkten ersetzt. In `terrain_workflow.py` werden diese Straßen aus der normalen Terrain-Einbettung und dem `DecalRoad`-Export herausgenommen und stattdessen von drei neuen Mesh-Buildern (`bridges/bridge_mesh.py`, `tunnels/tunnel_mesh.py`, `tunnels/gallery_mesh.py`) als DAE+TSStatic exportiert - alle drei folgen dem Extrusions-Muster aus dem bestehenden `walls/wall_mesh.py`.

**Tech Stack:** Python, NumPy, Shapely (bereits im Projekt), bestehende DAE-Export-/Material-/Textur-Infrastruktur (`managers/dae_exporter.py`, `managers/material_manager.py`, `textures/registry.py`).

**Spec:** `docs/superpowers/specs/2026-09-22-bridges-tunnels-design.md`

## Global Constraints

- Sprache: Chat-Antworten und Code-Kommentare auf Deutsch, Code selbst (Bezeichner, Strings, Commit-Präfixe) auf Englisch (siehe CLAUDE.md).
- Terrain-Heightmap bleibt unter Brücken und über/neben Tunneln/Galerien vollständig unverändert (Design-Spec Nicht-Ziele).
- Keine Stapel-/Kollisionsauflösung für sich kreuzende Brücken; keine Geländer-Geometrie; keine fotobasierte Beton-Textur (prozedural reicht).
- Jede neue Konfigurationskonstante bekommt einen Default, der den Export nicht bricht (`BRIDGES_ENABLED`/`TUNNELS_ENABLED = True`).
- Commit-Messages enden mit den Attribution-Zeilen aus der System-Reminder dieser Session.

**Implementierungshinweis (Abweichung von der Spec-Formulierung, siehe unten):** Die Design-Spec (Abschnitt 4/5) beschreibt `ground_at` als Abtastung "vor jeder Straßen-Änderung". Um konsistent mit dem bestehenden Muster für Mauern (`_build_wall_meshes()`, die dieselbe, zu diesem Zeitpunkt bereits durch Böschung/Straßen-Einbettung/Teichmulden verarbeitete `heights`-Variable verwendet) zu bleiben, verwenden Brücken/Tunnel/Galerien in diesem Plan dieselbe, zum Zeitpunkt des Mauer-Baus bereits fertige `heights`-Variable (nicht eine separate, frühere Momentaufnahme). Das ist semantisch gleichwertig (Brücken-/Tunnel-Straßen selbst verändern `heights` ja nicht mehr) und vermeidet eine zweite Heightmap-Variable nur für diesen einen Zweck.

**Implementierungshinweis (Ausschlusszonen unverändert statt nur Fußabdrücke):** Die Design-Spec (Abschnitt 8) beschreibt, dass Vegetations-Ausschlusszonen nur um Pfeiler-/Portal-Fußabdrücke statt um die ganze Brücken-/Tunnel-Spanne liegen sollen. Dieser Plan lässt `road_surface_union`/`road_shapes` in `process_tile()` bewusst unverändert (weiterhin aus ALLEN Straßen inkl. Brücken/Tunnel/Galerien gebildet) - das ist eine konservative, funktional unkritische Vereinfachung (kein Baumwuchs unter einer Talbrücke ist optisch unauffällig) und spart die zusätzliche Geometrie-Verdrahtung der Pfeiler-/Portal-Fußabdrücke zurück in die Terrain-Workflow-Ausschlusslogik. Eine genauere Fußabdruck-Variante ist eine mögliche spätere Verfeinerung, kein Teil dieses Plans.

**Implementierungshinweis (kreisrundes Tunnelprofil):** Task 8 wurde nach der initialen Planung überarbeitet - die Tunnelröhre ist kreisrund (Standard-Tunnelprofil: 240° Kreisbogen über einer flachen Bodensehne, die restlichen 120° unter der Sehne unmodelliert), nicht rechteckig. Radius (`R = Bodenbreite / sqrt(3)`) und Kronenhöhe (`1.5 * R`) ergeben sich aus der Fahrbahnbreite, `config.TUNNEL_HEIGHT` entfällt zugunsten von `config.TUNNEL_ARC_SEGMENTS`. Galerien (Task 9) bleiben bewusst rechteckig mit fester Höhe (`config.GALLERY_HEIGHT`) und ohne Portal-Rahmen - sie sind offene Schutzbauten, keine gebohrte Röhre.

**Implementierungshinweis (Vereinfachung der Portal-Geometrie):** Die Spec beschreibt, dass die Stirnfläche der Tunnel-Röhre selbst schräg abgeschnitten wird. Dieser Plan setzt das stattdessen im separaten, flachen Portal-Rahmen-Mesh um (die Röhre selbst behält ein rechtwinkliges, nicht sichtbares Ende dahinter) - liefert optisch dasselbe Ergebnis, ohne die Extrusion selbst scheren zu müssen. Siehe Task 8.

---

## Task 1: Klassifizierung von Brücken/Tunneln/Galerien

**Files:**
- Create: `world_to_beamng/geometry/road_structures.py`
- Test: `tests/geometry/test_road_structures.py`

**Interfaces:**
- Produces: `classify_structure(osm_tags: dict) -> str` (Werte: `"bridge"`, `"tunnel"`, `"gallery"`, `"surface"`); `split_by_structure_type(road_slope_polygons_2d: list[dict]) -> tuple[list[dict], list[dict]]` (liest `road["structure_type"]`, Default `"surface"`). Beide werden von Task 2 (`geometry/polygon.py`) und Task 3 (`terrain_workflow.py`) importiert.

- [ ] **Step 1: Failing Test schreiben**

`tests/geometry/test_road_structures.py`:

```python
"""Tests für world_to_beamng.geometry.road_structures: Klassifizierung von Brücken/Tunneln/Galerien anhand OSM-Tags."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.road_structures import classify_structure, split_by_structure_type


def test_bridge_tag_is_classified_as_bridge():
    assert classify_structure({"highway": "primary", "bridge": "yes"}) == "bridge"
    assert classify_structure({"highway": "primary", "bridge": "viaduct"}) == "bridge"


def test_bridge_no_is_not_a_bridge():
    assert classify_structure({"highway": "primary", "bridge": "no"}) == "surface"


def test_avalanche_protector_is_a_gallery_not_a_tunnel():
    assert classify_structure({"highway": "primary", "tunnel": "avalanche_protector"}) == "gallery"


def test_other_tunnel_values_are_classified_as_tunnel():
    assert classify_structure({"highway": "trunk", "tunnel": "yes"}) == "tunnel"
    assert classify_structure({"waterway": "stream", "tunnel": "culvert"}) == "tunnel"
    assert classify_structure({"highway": "path", "tunnel": "building_passage"}) == "tunnel"


def test_tunnel_no_is_not_a_tunnel():
    assert classify_structure({"highway": "primary", "tunnel": "no"}) == "surface"


def test_missing_tags_are_surface():
    assert classify_structure({}) == "surface"
    assert classify_structure(None) == "surface"
    assert classify_structure({"highway": "residential"}) == "surface"


def test_bridge_takes_priority_over_tunnel_if_both_are_present():
    assert classify_structure({"bridge": "yes", "tunnel": "yes"}) == "bridge"


def test_split_by_structure_type_separates_surface_from_structures():
    roads = [
        {"road_id": 1, "structure_type": "surface"},
        {"road_id": 2, "structure_type": "bridge"},
        {"road_id": 3, "structure_type": "tunnel"},
        {"road_id": 4, "structure_type": "gallery"},
        {"road_id": 5},  # fehlendes Feld -> gilt als surface
    ]

    surface, structures = split_by_structure_type(roads)

    assert [r["road_id"] for r in surface] == [1, 5]
    assert [r["road_id"] for r in structures] == [2, 3, 4]
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/geometry/test_road_structures.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.geometry.road_structures'`

- [ ] **Step 3: Implementierung schreiben**

`world_to_beamng/geometry/road_structures.py`:

```python
"""
Klassifizierung von Straßen-Ways als Brücke, Tunnel, Galerie oder normale Fahrbahn anhand ihrer OSM-Tags
(siehe Design-Spec docs/superpowers/specs/2026-09-22-bridges-tunnels-design.md Abschnitt 1).
"""

from typing import Dict, List, Tuple


def classify_structure(osm_tags: Dict) -> str:
    """
    "bridge" | "tunnel" | "gallery" | "surface", anhand von `bridge`/`tunnel`-Tags.

    Reihenfolge: bridge=* (außer "no") -> "bridge"; tunnel=avalanche_protector -> "gallery"; jedes andere
    tunnel=* (außer "no") -> "tunnel"; sonst "surface".
    """
    osm_tags = osm_tags or {}
    bridge = str(osm_tags.get("bridge", "")).strip().lower()
    if bridge and bridge != "no":
        return "bridge"
    tunnel = str(osm_tags.get("tunnel", "")).strip().lower()
    if tunnel == "avalanche_protector":
        return "gallery"
    if tunnel and tunnel != "no":
        return "tunnel"
    return "surface"


def split_by_structure_type(road_slope_polygons_2d: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    (surface_roads, structure_roads) - `structure_roads` sind Brücken/Tunnel/Galerien
    (road["structure_type"] != "surface"; fehlt das Feld, gilt die Straße als "surface").
    """
    surface, structures = [], []
    for road in road_slope_polygons_2d:
        target = surface if road.get("structure_type", "surface") == "surface" else structures
        target.append(road)
    return surface, structures
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/geometry/test_road_structures.py -v`
Expected: PASS (9 Tests)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/geometry/road_structures.py tests/geometry/test_road_structures.py
git commit -m "feat: Classify OSM roads as bridge, tunnel, or gallery structures

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 2: Lineares Höhenprofil für Brücken/Tunnel/Galerien

**Files:**
- Modify: `world_to_beamng/geometry/polygon.py` (Import + zwei neue Funktionen + ein Aufruf in `get_road_polygons()`)
- Test: `tests/geometry/test_structure_elevation_profile.py`

**Interfaces:**
- Consumes: `classify_structure` aus Task 1 (`world_to_beamng.geometry.road_structures`).
- Produces: `apply_structure_elevation_profiles(road_polygons: list[dict]) -> list[dict]` (mutiert `road["coords"]` in-place für Nicht-"surface"-Straßen und gibt dieselbe Liste zurück).

- [ ] **Step 1: Failing Test schreiben**

`tests/geometry/test_structure_elevation_profile.py`:

```python
"""Tests für world_to_beamng.geometry.polygon.apply_structure_elevation_profiles: Brücken/Tunnel/Galerien
bekommen ein lineares Höhenprofil zwischen ihren Endpunkten statt der rohen DGM-Höhe an jedem Punkt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.geometry.polygon import apply_structure_elevation_profiles


def _road(coords, **tags):
    return {"id": 1, "coords": coords, "name": "r", "osm_tags": tags}


def test_surface_roads_keep_their_raw_elevation():
    coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 150.0), (20.0, 0.0, 90.0)]
    roads = [_road(coords, highway="residential")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"] == coords


def test_bridge_gets_a_linear_profile_between_its_endpoints():
    # Rohe DGM-Höhe in der Mitte wäre die Talsohle (50 m) - die Brücke soll stattdessen glatt zwischen den
    # beiden Anschlusspunkten (100 m) interpolieren.
    coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 50.0), (20.0, 0.0, 100.0)]
    roads = [_road(coords, highway="primary", bridge="yes")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"][0] == (0.0, 0.0, 100.0)
    assert result[0]["coords"][-1] == (20.0, 0.0, 100.0)
    assert result[0]["coords"][1][2] == pytest.approx(100.0)  # Mittelpunkt: nicht mehr die Talsohle


def test_tunnel_profile_is_weighted_by_arc_length_not_point_count():
    # Ungleichmäßig verteilte Punkte: bei 0m, 10m, 40m (Gesamtlänge 40m) Endhöhen 100m/180m
    coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 999.0), (40.0, 0.0, 180.0)]
    roads = [_road(coords, highway="trunk", tunnel="yes")]

    result = apply_structure_elevation_profiles(roads)

    # Bei 10m von 40m Gesamtlänge: 100 + (10/40)*(180-100) = 120
    assert result[0]["coords"][1][2] == pytest.approx(120.0)


def test_gallery_profile_also_gets_linearised():
    coords = [(0.0, 0.0, 200.0), (5.0, 0.0, 500.0), (10.0, 0.0, 210.0)]
    roads = [_road(coords, highway="primary", tunnel="avalanche_protector")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"][1][2] == pytest.approx(205.0)


def test_degenerate_zero_length_road_is_left_unchanged():
    coords = [(5.0, 5.0, 100.0), (5.0, 5.0, 100.0)]
    roads = [_road(coords, bridge="yes")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"] == coords
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/geometry/test_structure_elevation_profile.py -v`
Expected: FAIL mit `ImportError: cannot import name 'apply_structure_elevation_profiles'`

- [ ] **Step 3: Implementierung schreiben**

In `world_to_beamng/geometry/polygon.py`, Import-Block (aktuell Zeilen 5-14) erweitern:

```python
import numpy as np
from shapely.geometry import Polygon

from ..terrain.elevation import get_elevations_for_points
from ..geometry.coordinates import transformer_to_utm
from ..config import OSM_MAPPER
from .. import config
from .road_structures import classify_structure
from world_to_beamng.logging_config import LoggerConfig
```

Direkt vor `def get_road_polygons(...)` (aktuell Zeile 217) zwei neue Funktionen einfügen:

```python
def _linear_elevation_profile(coords):
    """Ersetzt die Z-Werte durch lineare Interpolation zwischen Anfangs- und Endpunkt (Bogenlänge-gewichtet);
    Start/Ende bleiben exakt erhalten (dort schließt die normale Straße an, siehe Design-Spec Abschnitt 2)."""
    arr = np.array(coords, dtype=float)
    xy = arr[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return coords
    t = cum / total
    z = arr[0, 2] + t * (arr[-1, 2] - arr[0, 2])
    return [(float(x), float(y), float(zz)) for (x, y), zz in zip(xy, z)]


def apply_structure_elevation_profiles(road_polygons):
    """Brücken/Tunnel/Galerien (siehe geometry.road_structures.classify_structure) bekommen ein lineares
    Höhenprofil zwischen ihren Endpunkten statt der rohen DGM-Höhe an jedem Punkt - siehe Design-Spec Abschnitt 2
    (z.B. der 16,9 km lange Gotthard-Straßentunnel bekommt sonst die Bergrücken-Höhe darüber zugewiesen)."""
    for road in road_polygons:
        if classify_structure(road.get("osm_tags", {})) != "surface" and len(road["coords"]) >= 2:
            road["coords"] = _linear_elevation_profile(road["coords"])
    return road_polygons
```

In `get_road_polygons()`, den Aufruf direkt vor "SCHRITT 4" einfügen (aktuell Zeile ~333):

```python
    # SCHRITT 3b: Brücken/Tunnel/Galerien bekommen ein lineares Höhenprofil statt der rohen DGM-Abtastung
    # (siehe Design-Spec Abschnitt 2) - VOR dem Smoothing, damit dieses auf dem bereits korrekten Profil arbeitet.
    road_polygons = apply_structure_elevation_profiles(road_polygons)

    # SCHRITT 4: Optional - mildes XY-Smoothing (Z bleibt erhalten oder nur leicht geglättet)
    if config.ENABLE_ROAD_SMOOTHING:
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/geometry/test_structure_elevation_profile.py tests/geometry/test_polygon.py -v`
Expected: PASS (beide Dateien, keine Regression in `test_polygon.py`)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/geometry/polygon.py tests/geometry/test_structure_elevation_profile.py
git commit -m "feat: Give bridges/tunnels/galleries a linear elevation profile

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 3: Terrain-Ausnahme & DecalRoad-Ausschluss

**Files:**
- Modify: `world_to_beamng/workflow/terrain_workflow.py`
- Test: `tests/workflow/test_terrain_workflow_structures.py`

**Interfaces:**
- Consumes: `classify_structure`, `split_by_structure_type` aus Task 1.
- Produces: `process_tile()`-Rückgabe-Dict bekommt zusätzlich `"structure_road_polygons"` (Liste der Brücken/Tunnel/Galerien-Straßen, für Task 6/11); jedes Dict in `road_slope_polygons_2d` bekommt `"structure_type"`.

- [ ] **Step 1: Failing Tests schreiben**

`tests/workflow/test_terrain_workflow_structures.py`:

```python
"""Tests für die Terrain-Ausnahme und den DecalRoad-Ausschluss von Brücken/Tunnel/Galerien
(TerrainWorkflow.process_tile()-Verdrahtung und export_decal_roads())."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.geometry.road_structures import split_by_structure_type
from world_to_beamng.terrain.road_embedding import embed_roads_into_heightmap
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


def _poly(road_id, structure_type, z):
    coords = np.array([[0.0, 5.0, z], [10.0, 5.0, z]])
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return {"road_id": road_id, "road_polygon": polygon, "trimmed_centerline": coords, "osm_tags": {}, "structure_type": structure_type}


def test_bridge_and_tunnel_polygons_are_not_embedded_into_the_heightmap():
    roads = [_poly(1, "surface", z=150.0), _poly(2, "bridge", z=200.0), _poly(3, "tunnel", z=90.0)]
    heights = np.full((20, 20), 100.0)

    surface_roads, structure_roads = split_by_structure_type(roads)
    assert {r["road_id"] for r in structure_roads} == {2, 3}

    result = embed_roads_into_heightmap(heights, 0.0, 0.0, 1.0, surface_roads)

    assert result[5, 5] == 150.0  # nur die surface-Straße (1) wurde eingebettet


class _RecordingItems:
    def __init__(self):
        self.roads = {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = {"nodes": nodes, "material": material, **extra}


def test_export_decal_roads_skips_bridges_tunnels_and_galleries():
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}))
    polys = [
        {"road_id": 1, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "residential"}, "structure_type": "surface"},
        {"road_id": 2, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "primary", "bridge": "yes"}, "structure_type": "bridge"},
        {"road_id": 3, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "trunk", "tunnel": "yes"}, "structure_type": "tunnel"},
        {"road_id": 4, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"tunnel": "avalanche_protector"}, "structure_type": "gallery"},
    ]

    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})

    assert count == 1
    assert list(stub.items.roads) == ["road_1"]


def test_export_decal_roads_still_works_without_a_structure_type_field():
    # Regression: bestehende Aufrufer/Tests, die "structure_type" nicht setzen, bleiben unverändert (surface-Default).
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}))
    polys = [{"road_id": 1, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "residential"}}]

    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})

    assert count == 1
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/workflow/test_terrain_workflow_structures.py -v`
Expected: FAIL - `test_export_decal_roads_skips_bridges_tunnels_and_galleries` schlägt fehl (`count == 4`, nicht 1), die anderen beiden PASSen bereits zufällig (sie prüfen nur bereits vorhandenes Verhalten von `split_by_structure_type`/`embed_roads_into_heightmap`, die schon aus Task 1/vorhandenem Code funktionieren) - relevant ist der DecalRoad-Test.

- [ ] **Step 3: Implementierung schreiben**

In `world_to_beamng/workflow/terrain_workflow.py`, `process_tile()`: Import-Block um Zeile 200-203 ergänzen:

```python
        from shapely.geometry import LineString
        from ..config import OSM_MAPPER
        from ..geometry.road_structures import classify_structure, split_by_structure_type
        from ..utils.debug_exporter import DebugNetworkExporter
```

In derselben Methode, beim Aufbau von `road_slope_polygons_2d` (aktuell Zeilen 227-234), `"structure_type"` ergänzen:

```python
            road_slope_polygons_2d.append(
                {
                    "road_id": road_id,  # Wichtig für Material-Mapping
                    "road_polygon": road_polygon_2d,
                    "trimmed_centerline": coords,
                    "osm_tags": osm_tags,
                    "structure_type": classify_structure(osm_tags),
                }
            )
```

Direkt vor dem Aufruf von `build_road_embankment_profiles(...)` (aktuell Zeile 290) einfügen:

```python
        # Brücken/Tunnel/Galerien werden NICHT ins Terrain eingebettet und bekommen keine Böschung - siehe
        # Design-Spec Abschnitt 3 (das Gelände bleibt darunter/daneben vollständig natürlich).
        surface_road_polygons, structure_road_polygons = split_by_structure_type(road_slope_polygons_2d)

        embankment_profiles = build_road_embankment_profiles(
            surface_road_polygons,
```

(nur das erste Argument des bestehenden Aufrufs ändert sich von `road_slope_polygons_2d` zu `surface_road_polygons`, der Rest der Argumente bleibt unverändert.)

Beim Aufruf von `embed_roads_into_heightmap(...)` (aktuell Zeile 307-313) ebenfalls das letzte Argument ändern:

```python
        heights = embed_roads_into_heightmap(
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            surface_road_polygons,
        )
```

Im Rückgabe-Dict von `process_tile()` (aktuell Zeile ~495-497) `"structure_road_polygons"` ergänzen:

```python
            "road_polygons": road_polygons,
            "road_slope_polygons_2d": road_slope_polygons_2d,  # Für DecalRoad-Export
            "structure_road_polygons": structure_road_polygons,  # Brücken/Tunnel/Galerien - für export_bridges()/export_tunnels()
            "road_surface_union": road_surface_union,  # vereinigte Straßenfläche für Ausschlusszonen (oder None)
```

In `export_decal_roads()`, in der `for poly in road_slope_polygons_2d:`-Schleife (aktuell Zeile 733), direkt nach dem Schleifenkopf einfügen:

```python
        for poly in road_slope_polygons_2d:
            if poly.get("structure_type", "surface") != "surface":
                continue
            road_id = poly.get("road_id")
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/workflow/test_terrain_workflow_structures.py tests/geometry/test_decal_road_nodes.py -v`
Expected: PASS (keine Regression im bestehenden DecalRoad-Test)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/workflow/terrain_workflow.py tests/workflow/test_terrain_workflow_structures.py
git commit -m "feat: Exempt bridges/tunnels/galleries from terrain embedding and DecalRoad export

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 4: Gemeinsamer Mesh-Baustein `add_box_column`

**Files:**
- Modify: `world_to_beamng/walls/mesh_parts.py`
- Test: `tests/walls/test_mesh_parts_column.py`

**Interfaces:**
- Produces: `add_box_column(builder: MeshBuilder, cx: float, cy: float, bottom_z: float, top_z: float, size: float, tile_m: float) -> None` - wird von Task 5 (Brücken-Pfeiler) und Task 9 (Galerie-Stützen) verwendet.

- [ ] **Step 1: Failing Test schreiben**

`tests/walls/test_mesh_parts_column.py`:

```python
"""Tests für world_to_beamng.walls.mesh_parts.add_box_column: rechteckige Stütze (Brücken-Pfeiler, Galerie-Stützen)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.walls.mesh_parts import MeshBuilder, add_box_column


def test_column_spans_the_given_height_and_footprint():
    builder = MeshBuilder()

    add_box_column(builder, cx=10.0, cy=20.0, bottom_z=100.0, top_z=105.0, size=1.5, tile_m=1.0)

    v = np.array(builder.vertices)
    assert v[:, 2].min() == pytest.approx(100.0) and v[:, 2].max() == pytest.approx(105.0)
    assert v[:, 0].min() == pytest.approx(10.0 - 0.75) and v[:, 0].max() == pytest.approx(10.0 + 0.75)
    assert v[:, 1].min() == pytest.approx(20.0 - 0.75) and v[:, 1].max() == pytest.approx(20.0 + 0.75)


def test_column_has_four_outward_facing_side_quads():
    builder = MeshBuilder()

    add_box_column(builder, cx=0.0, cy=0.0, bottom_z=0.0, top_z=1.0, size=1.0, tile_m=1.0)

    assert len(builder.faces) == 4 * 2  # 4 Seiten, je 2 Dreiecke
    directions = {tuple(np.round(n, 2)) for n in builder.normals}
    assert directions == {(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)}
    tris = np.array([[builder.vertices[i] for i in face] for face in builder.faces])
    for face, tri in zip(builder.faces, tris):
        geometric = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        assert np.dot(geometric, builder.normals[face[0]]) > 0
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/walls/test_mesh_parts_column.py -v`
Expected: FAIL mit `ImportError: cannot import name 'add_box_column'`

- [ ] **Step 3: Implementierung schreiben**

In `world_to_beamng/walls/mesh_parts.py`, am Ende der Datei ergänzen:

```python
def add_box_column(builder: "MeshBuilder", cx: float, cy: float, bottom_z: float, top_z: float, size: float, tile_m: float) -> None:
    """Rechteckige Stütze (4 Seitenflächen) von `bottom_z` bis `top_z`, quadratischer Querschnitt `size` - für
    Brücken-Pfeiler (bridges/bridge_mesh.py) und Galerie-Stützen (tunnels/gallery_mesh.py)."""
    half = size / 2.0
    corners = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    height_tiles = (top_z - bottom_z) / tile_m
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        direction = np.array([b[0] - a[0], b[1] - a[1]])
        direction = direction / np.linalg.norm(direction)
        normal = [float(direction[1]), float(-direction[0]), 0.0]
        builder.quad(
            [[a[0], a[1], bottom_z], [b[0], b[1], bottom_z], [b[0], b[1], top_z], [a[0], a[1], top_z]],
            [[0.0, 0.0], [size / tile_m, 0.0], [size / tile_m, height_tiles], [0.0, height_tiles]],
            normal,
        )
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/walls/test_mesh_parts_column.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/walls/mesh_parts.py tests/walls/test_mesh_parts_column.py
git commit -m "feat: Add a shared box-column mesh helper for piers and gallery supports

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 5: Brücken-Mesh-Builder (Deck + Pfeiler)

**Files:**
- Create: `world_to_beamng/bridges/__init__.py` (leer)
- Create: `world_to_beamng/bridges/bridge_mesh.py`
- Test: `tests/bridges/__init__.py` (leer)
- Test: `tests/bridges/test_bridge_mesh.py`

**Interfaces:**
- Consumes: `MeshBuilder`, `offset_points` aus `world_to_beamng.walls.mesh_parts`; `add_box_column` aus Task 4.
- Produces: `build_bridge_mesh(coords, width, ground_at, deck_material, pier_material, deck_thickness=0.6, pier_spacing=25.0, pier_size=1.5, min_pier_clearance=1.0, tile_m=5.0) -> dict`; `build_bridges(bridges: list[dict], ground_at, pier_material, deck_thickness=0.6, pier_spacing=25.0, pier_size=1.5, min_pier_clearance=1.0) -> list[dict]` (jedes `bridges`-Item: `{"id", "coords", "width", "deck_material"}`). Wird von Task 6 (`TerrainWorkflow._build_bridges`) konsumiert.

- [ ] **Step 1: Failing Tests schreiben**

`tests/bridges/__init__.py`: leer.

`tests/bridges/test_bridge_mesh.py`:

```python
"""Tests für world_to_beamng.bridges.bridge_mesh: Brücken-Deck + Stützpfeiler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.bridges.bridge_mesh import build_bridge_mesh, build_bridges

DECK, PIER = "asphalt_road_standard", "bridge_concrete"


def _flat_ground(z=150.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _coords(length=60.0, z=200.0, n=7):
    return [(x, 5.0, z) for x in np.linspace(0.0, length, n)]


def test_deck_top_is_flat_at_the_given_height_and_road_width():
    mesh = build_bridge_mesh(_coords(z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, deck_thickness=0.6)
    v = mesh["vertices"]

    assert v[:, 2].max() == pytest.approx(200.0)  # Deck-Oberkante = Höhenprofil, folgt NICHT dem Gelände
    assert v[:, 2].min() == pytest.approx(200.0 - 0.6)  # Deck-Unterkante minus Pfeiler-Vertices


def test_deck_faces_use_the_road_material_not_the_pier_material():
    mesh = build_bridge_mesh(_coords(n=3), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER)

    assert DECK in mesh["faces"] and len(mesh["faces"][DECK]) > 0


def test_piers_reach_down_to_the_natural_ground_below_a_deep_span():
    mesh = build_bridge_mesh(_coords(length=60.0, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, pier_spacing=25.0)

    assert len(mesh["faces"][PIER]) > 0
    pier_face = mesh["faces"][PIER][0]
    pier_z = np.array([mesh["vertices"][i][2] for i in pier_face])
    assert pier_z.min() == pytest.approx(150.0)  # Pfeiler reicht bis zum natürlichen Gelände


def test_no_piers_when_clearance_is_too_small():
    mesh = build_bridge_mesh(_coords(length=60.0, z=151.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, deck_thickness=0.2, min_pier_clearance=1.0)

    assert mesh["faces"].get(PIER, []) == []


def test_build_bridges_returns_one_mesh_per_bridge_with_its_own_deck_material():
    bridges = [
        {"id": 1, "coords": _coords(z=200.0), "width": 8.0, "deck_material": "asphalt_road_standard"},
        {"id": 2, "coords": _coords(z=210.0), "width": 6.0, "deck_material": "concrete"},
    ]

    meshes = build_bridges(bridges, _flat_ground(150.0), pier_material=PIER)

    assert [m["id"] for m in meshes] == ["bridge_1", "bridge_2"]
    assert "asphalt_road_standard" in meshes[0]["faces"] and "concrete" in meshes[1]["faces"]


def test_build_bridges_skips_degenerate_bridges():
    bridges = [{"id": 1, "coords": [(0.0, 0.0, 200.0)], "width": 8.0, "deck_material": DECK}]

    assert build_bridges(bridges, _flat_ground(150.0), pier_material=PIER) == []
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/bridges/test_bridge_mesh.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.bridges'`

- [ ] **Step 3: Implementierung schreiben**

`world_to_beamng/bridges/__init__.py`: leer.

`world_to_beamng/bridges/bridge_mesh.py`:

```python
"""
Brücken aus OSM-Linien (highway=* mit bridge=*): generisches Beton-Deck mit dem Fahrbahnmaterial der Straße
obenauf und rechteckigen Stützpfeilern zum natürlichen Gelände darunter (siehe Design-Spec Abschnitt 4).

Das Deck folgt NICHT dem Gelände (im Gegensatz zu den Mauern) - seine Höhe kommt aus dem linear interpolierten
Brücken-Höhenprofil (geometry/road_structures.py + geometry/polygon.py), das schon in den übergebenen `coords`
steckt. Nur die Pfeiler reichen bis zum natürlichen Gelände darunter (`ground_at`).
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column, offset_points

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def build_bridge_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    ground_at: HeightAt,
    deck_material: str,
    pier_material: str,
    deck_thickness: float = 0.6,
    pier_spacing: float = 25.0,
    pier_size: float = 1.5,
    min_pier_clearance: float = 1.0,
    tile_m: float = 5.0,
) -> Dict:
    """
    Deck- und Pfeiler-Mesh für eine Brücke entlang `coords` (bereits das Brücken-Höhenprofil, x,y,z je Punkt).

    Returns:
        {"vertices": (N,3), "uvs": (N,2), "normals": (N,3), "faces": {deck_material: [...], pier_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    top = points[:, 2]
    bottom = top - deck_thickness

    left, right = offset_points(xy, width / 2.0, closed=False)

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    along = cum / tile_m
    across = width / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    deck_builder = MeshBuilder()
    for i in range(len(points) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        side_normal = [float(-direction[1]), float(direction[0]), 0.0]

        # Oberseite (Fahrbahn)
        deck_builder.quad(
            [p3(left[i], top[i]), p3(left[j], top[j]), p3(right[j], top[j]), p3(right[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )
        # Unterseite
        deck_builder.quad(
            [p3(left[i], bottom[i]), p3(right[i], bottom[i]), p3(right[j], bottom[j]), p3(left[j], bottom[j])],
            [[u0, 0.0], [u0, across], [u1, across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        # Fascia links/rechts
        deck_builder.quad(
            [p3(left[i], bottom[i]), p3(left[j], bottom[j]), p3(left[j], top[j]), p3(left[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, deck_thickness / tile_m], [u0, deck_thickness / tile_m]],
            side_normal,
        )
        deck_builder.quad(
            [p3(right[i], bottom[i]), p3(right[j], bottom[j]), p3(right[j], top[j]), p3(right[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, deck_thickness / tile_m], [u0, deck_thickness / tile_m]],
            [-side_normal[0], -side_normal[1], 0.0],
        )

    # Stirnflächen an den beiden Enden
    for index, sign, neighbour in ((0, -1.0, 1), (len(points) - 1, 1.0, len(points) - 2)):
        direction = xy[1] - xy[0] if index == 0 else xy[-1] - xy[neighbour]
        direction = direction / np.linalg.norm(direction)
        deck_builder.quad(
            [p3(left[index], bottom[index]), p3(right[index], bottom[index]), p3(right[index], top[index]), p3(left[index], top[index])],
            [[0.0, 0.0], [across, 0.0], [across, deck_thickness / tile_m], [0.0, deck_thickness / tile_m]],
            [float(sign * direction[0]), float(sign * direction[1]), 0.0],
        )

    # Pfeiler: alle pier_spacing Meter entlang der Bogenlänge, nur wenn ausreichend Abstand zum Gelände besteht
    pier_builder = MeshBuilder()
    total_len = float(cum[-1])
    pier_positions = np.arange(pier_spacing, total_len, pier_spacing) if total_len > pier_spacing else np.array([])
    for s in pier_positions:
        idx = max(1, min(int(np.searchsorted(cum, s)), len(points) - 1))
        t = (s - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
        cx = xy[idx - 1, 0] + t * (xy[idx, 0] - xy[idx - 1, 0])
        cy = xy[idx - 1, 1] + t * (xy[idx, 1] - xy[idx - 1, 1])
        deck_bottom_z = float(bottom[idx - 1] + t * (bottom[idx] - bottom[idx - 1]))
        ground_z = float(ground_at(np.array([cx]), np.array([cy]))[0])
        if deck_bottom_z - ground_z < min_pier_clearance:
            continue
        add_box_column(pier_builder, cx, cy, ground_z, deck_bottom_z, pier_size, tile_m)

    all_vertices = deck_builder.vertices + pier_builder.vertices
    all_uvs = deck_builder.uvs + pier_builder.uvs
    all_normals = deck_builder.normals + pier_builder.normals
    pier_offset = len(deck_builder.vertices)
    pier_faces = [[a + pier_offset, b + pier_offset, c + pier_offset] for a, b, c in pier_builder.faces]

    return {
        "vertices": np.array(all_vertices, dtype=float),
        "uvs": np.array(all_uvs, dtype=float),
        "normals": np.array(all_normals, dtype=float),
        "faces": {deck_material: deck_builder.faces, pier_material: pier_faces},
    }


def build_bridges(
    bridges: Sequence[Dict],
    ground_at: HeightAt,
    pier_material: str,
    deck_thickness: float = 0.6,
    pier_spacing: float = 25.0,
    pier_size: float = 1.5,
    min_pier_clearance: float = 1.0,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, eines je Brücke (`bridges`: [{"id","coords","width","deck_material"}, ...])."""
    meshes = []
    for bridge in bridges:
        coords = bridge["coords"]
        if len(coords) < 2:
            continue
        mesh = build_bridge_mesh(
            coords, bridge["width"], ground_at, bridge["deck_material"], pier_material,
            deck_thickness=deck_thickness, pier_spacing=pier_spacing, pier_size=pier_size, min_pier_clearance=min_pier_clearance,
        )
        meshes.append({"id": f"bridge_{bridge['id']}", **mesh})
    return meshes
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/bridges/test_bridge_mesh.py -v`
Expected: PASS (7 Tests)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/bridges tests/bridges
git commit -m "feat: Add bridge deck+pier mesh builder

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 6: Brücken-Integration in TerrainWorkflow + Export

**Files:**
- Modify: `world_to_beamng/config.py` (neue Konstanten)
- Modify: `world_to_beamng/workflow/terrain_workflow.py` (`_build_bridges()`, `export_bridges()`, Verdrahtung in `process_tile()`/`export_tile()`)
- Test: `tests/workflow/test_terrain_workflow_bridges.py`

**Interfaces:**
- Consumes: `build_bridges` aus Task 5; `structure_road_polygons` aus Task 3 (`process_tile()`-Rückgabe); `sample_heightmap_bilinear` aus `terrain.road_embedding` (bereits vorhanden).
- Produces: `TerrainWorkflow._build_bridges(structure_road_polygons, heights, terrain_origin_x, terrain_origin_y) -> list[dict]`; `TerrainWorkflow.export_bridges(mesh_data) -> int`. `process_tile()`-Rückgabe bekommt `"bridge_meshes"`. `export_tile()` ruft `export_bridges()` auf.

- [ ] **Step 1: Failing Tests schreiben**

`tests/workflow/test_terrain_workflow_bridges.py`:

```python
"""Tests für TerrainWorkflow._build_bridges() und export_bridges(): Brücken (Deck + Pfeiler) als eigene DAE mit einem TSStatic."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.textures import registry
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _Items:
    def __init__(self):
        self.objects = {}

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), **fields):
        self.objects[name] = {"class": item_class, "position": list(position), **fields}


class _Materials:
    def __init__(self):
        self.added = {}

    def get_templates(self):
        return {"buildings": {"wall": {"material_hints": {"groundType": "concrete", "materialTag0": "beamng", "materialTag1": "Building"}}}}

    def add_building_material(self, name, **fields):
        self.added[name] = fields


class _Dae:
    def __init__(self):
        self.calls = []

    def export_multi_mesh(self, output_path, meshes, with_uv=False, material_textures=None):
        self.calls.append((Path(output_path), meshes, with_uv))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<COLLADA/>", encoding="utf-8")
        return output_path


def _stub():
    return SimpleNamespace(items=_Items(), materials=_Materials(), dae=_Dae())


CONCRETE = {
    "baseColorMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_b.color.dds",
    "normalMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_nm.normal.dds",
    "roughnessMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_r.data.dds",
}


@pytest.fixture
def shapes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    monkeypatch.setattr(registry, "prepared_textures", lambda: {config.CONCRETE_TEXTURE_NAME: CONCRETE})
    return tmp_path / "shapes"


def _mesh(name="bridge_1"):
    return {
        "id": name,
        "vertices": np.zeros((8, 3)),
        "uvs": np.zeros((8, 2)),
        "normals": np.tile([0.0, 0.0, 1.0], (8, 1)),
        "faces": {"asphalt_road_standard": [[0, 1, 2]], config.BRIDGE_MATERIAL_NAME: [[4, 5, 6]]},
    }


def _road(road_id, highway="primary"):
    return {
        "road_id": road_id,
        "trimmed_centerline": np.array([[0.0, 0.0, 200.0], [50.0, 0.0, 200.0]]),
        "osm_tags": {"highway": highway, "bridge": "yes"},
        "structure_type": "bridge",
    }


def test_export_bridges_writes_one_dae_one_item_and_registers_deck_and_pier_materials(shapes_dir):
    stub = _stub()

    count = TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [_mesh("bridge_1"), _mesh("bridge_2")], "structure_road_polygons": [_road(1), _road(2)]})

    assert count == 2
    dae_path, meshes, with_uv = stub.dae.calls[0]
    assert dae_path == shapes_dir / "bridges" / "bridges.dae" and with_uv is True and len(meshes) == 2
    item = stub.items.objects["bridges"]
    assert item["class"] == "TSStatic" and item["shape_name"] == "levels/world_to_beamng/art/shapes/bridges/bridges.dae"
    assert item["collisionType"] == "Visible Mesh Final"
    assert config.BRIDGE_MATERIAL_NAME in stub.materials.added
    assert "asphalt_road_standard" in stub.materials.added  # Fahrbahn-Deckmaterial (highway=primary)
    assert stub.materials.added["asphalt_road_standard"]["groundType"] == "ASPHALT"


def test_export_bridges_takes_the_concrete_texture_from_the_registry_check_and_never_falls_back(shapes_dir, monkeypatch):
    def aborted():
        raise registry.MissingTexturesError("Beton-Textur fehlt")

    monkeypatch.setattr(registry, "prepared_textures", aborted)
    stub = _stub()

    with pytest.raises(registry.MissingTexturesError):
        TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [_mesh()], "structure_road_polygons": [_road(1)]})
    assert not stub.materials.added and not stub.dae.calls


def test_nothing_is_exported_and_stale_files_are_removed_without_bridges(shapes_dir, monkeypatch):
    stale = shapes_dir / "bridges"
    stale.mkdir(parents=True)
    (stale / "bridges.dae").write_text("alt", encoding="utf-8")
    (stale / "bridges.cdae").write_text("alt", encoding="utf-8")
    stub = _stub()

    assert TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [], "structure_road_polygons": []}) == 0
    assert not (stale / "bridges.dae").exists() and not (stale / "bridges.cdae").exists()

    monkeypatch.setattr(config, "BRIDGES_ENABLED", False)
    assert TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [_mesh()], "structure_road_polygons": [_road(1)]}) == 0
    assert not stub.dae.calls


def test_build_bridges_creates_a_mesh_per_bridge_with_a_pier_over_a_deep_span():
    heights = np.full((50, 50), 150.0)  # flaches Tal, 50 m unter der Brücke
    coords = np.array([[x, 5.0, 200.0] for x in np.linspace(0.0, 60.0, 7)])  # Brücke auf 200 m Höhe
    road = {"road_id": 1, "trimmed_centerline": coords, "osm_tags": {"highway": "primary", "bridge": "yes"}, "structure_type": "bridge"}

    meshes = TerrainWorkflow._build_bridges(SimpleNamespace(), [road], heights, 0.0, 0.0)

    assert len(meshes) == 1 and meshes[0]["id"] == "bridge_1"
    assert "asphalt_road_standard" in meshes[0]["faces"] and config.BRIDGE_MATERIAL_NAME in meshes[0]["faces"]
    assert len(meshes[0]["faces"][config.BRIDGE_MATERIAL_NAME]) > 0  # mindestens ein Pfeiler bei 60 m Spannweite


def test_build_bridges_skips_non_bridge_roads():
    heights = np.full((10, 10), 150.0)
    road = {"road_id": 1, "trimmed_centerline": np.array([[0.0, 0.0, 200.0], [5.0, 0.0, 200.0]]), "osm_tags": {}, "structure_type": "tunnel"}

    assert TerrainWorkflow._build_bridges(SimpleNamespace(), [road], heights, 0.0, 0.0) == []
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/workflow/test_terrain_workflow_bridges.py -v`
Expected: FAIL mit `AttributeError: type object 'TerrainWorkflow' has no attribute '_build_bridges'` (und `config` hat noch keine `BRIDGE_*`/`CONCRETE_TEXTURE_NAME`-Konstanten)

- [ ] **Step 3: Implementierung schreiben**

In `world_to_beamng/config.py`, nach dem bestehenden `WALL_*`-Block (nach der Zeile mit `WALL_MATERIAL_NAME = ...`) einfügen:

```python
# Brücken (OSM highway=* mit bridge=*): generisches Beton-Deck mit dem Fahrbahnmaterial der Straße obenauf und
# Stützpfeilern zum natürlichen Gelände darunter (siehe Design-Spec docs/superpowers/specs/
# 2026-09-22-bridges-tunnels-design.md Abschnitt 4). Ersetzt für diese Straßen die normale Terrain-Einbettung
# und den DecalRoad-Export.
BRIDGES_ENABLED = True
BRIDGE_DECK_THICKNESS = 0.6  # Deck-Dicke in Metern
BRIDGE_PIER_SPACING = 25.0  # Abstand der Stützpfeiler entlang der Brücke, in Metern
BRIDGE_PIER_SIZE = 1.5  # Querschnitt der (quadratischen) Stützpfeiler, in Metern
BRIDGE_MIN_PIER_CLEARANCE = 1.0  # kein Pfeiler, wenn der Abstand Deck-Unterkante/Gelände kleiner ist, in Metern
BRIDGE_MATERIAL_NAME = "bridge_concrete"  # Pfeiler-Material (Textur: CONCRETE_TEXTURE_NAME, siehe Task 10)
```

Nahe dem `FLAT_ROOF_GRAVEL_*`-Block einfügen (die eigentliche Textur folgt in Task 10, die Konstanten werden aber schon hier gebraucht):

```python
# Prozedurale Beton-Textur für Brücken-Pfeiler, Tunnel-Wände/Decke/Portale und Galerie-Dach/Stützen.
CONCRETE_TEXTURE_NAME = "tunnel_concrete"
CONCRETE_TEXTURE_TILE_M = 2.0
CONCRETE_TEXTURE_PX = 1024
```

In `world_to_beamng/workflow/terrain_workflow.py`, direkt nach der bestehenden `_build_wall_meshes()`-Methode zwei neue Methoden einfügen:

```python
    def _build_bridges(self, structure_road_polygons: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float) -> List[Dict]:
        """Brücken-Meshes (Deck + Pfeiler) für alle Straßen mit structure_type == "bridge" (siehe bridges/bridge_mesh.py)."""
        from ..bridges.bridge_mesh import build_bridges
        from ..terrain.road_embedding import sample_heightmap_bilinear

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        bridges = [
            {
                "id": road["road_id"],
                "coords": road["trimmed_centerline"],
                "width": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"],
                "deck_material": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {})).get("internal_name", "road_default"),
            }
            for road in structure_road_polygons
            if road.get("structure_type") == "bridge"
        ]
        return build_bridges(
            bridges,
            ground_at,
            pier_material=config.BRIDGE_MATERIAL_NAME,
            deck_thickness=config.BRIDGE_DECK_THICKNESS,
            pier_spacing=config.BRIDGE_PIER_SPACING,
            pier_size=config.BRIDGE_PIER_SIZE,
            min_pier_clearance=config.BRIDGE_MIN_PIER_CLEARANCE,
        )

    def export_bridges(self, mesh_data: Dict) -> int:
        """
        Exportiert Brücken als EINE DAE (Deck + Pfeiler je Brücke) mit EINEM TSStatic und registriert Fahrbahn-
        und Beton-Material. Ohne Brücken werden Reste eines früheren Exports entfernt.

        Returns:
            Anzahl exportierter Brücken
        """
        bridges_dir = config.BEAMNG_DIR_SHAPES / "bridges"
        meshes = mesh_data.get("bridge_meshes") or []
        if not config.BRIDGES_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (bridges_dir / f"bridges{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.BRIDGE_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        unique_deck_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") != "bridge":
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            unique_deck_materials[props.get("internal_name", "road_default")] = props

        for mat_name, props in unique_deck_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=bridges_dir / "bridges.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "bridges",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "bridges" / "bridges.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.info(f"  [OK] {len(meshes)} Brücken exportiert (bridges.dae)")
        return len(meshes)
```

In `process_tile()`, direkt nach dem bestehenden Block für `wall_meshes` (nach `wall_meshes, _ = self._build_wall_meshes(...)`) einfügen:

```python
        # Brücken (Deck + Pfeiler) auf der fertigen Heightmap - siehe bridges/bridge_mesh.py
        bridge_meshes = []
        if config.BRIDGES_ENABLED:
            bridge_meshes = self._build_bridges(structure_road_polygons, heights, terrain_origin_x, terrain_origin_y)
```

Im Rückgabe-Dict von `process_tile()`, direkt nach `"wall_meshes": wall_meshes,` einfügen:

```python
            "bridge_meshes": bridge_meshes,  # Brücken-Mesh-Dicts für export_bridges()
```

In `export_tile()`, nach `self.export_walls(mesh_data)` einfügen:

```python
        self.export_bridges(mesh_data)
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/workflow/test_terrain_workflow_bridges.py -v`
Expected: PASS (5 Tests)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/config.py world_to_beamng/workflow/terrain_workflow.py tests/workflow/test_terrain_workflow_bridges.py
git commit -m "feat: Export bridges as their own DAE/TSStatic in the terrain workflow

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 7: Portal-Hangneigung (gemeinsamer Baustein für Tunnel)

**Files:**
- Create: `world_to_beamng/tunnels/__init__.py` (leer)
- Create: `world_to_beamng/tunnels/portal.py`
- Test: `tests/tunnels/__init__.py` (leer)
- Test: `tests/tunnels/test_portal.py`

**Interfaces:**
- Produces: `sample_slope_along_axis(ground_at, point_xy, axis_direction, sample_dist) -> float`; `portal_axial_shift(height_above_floor, slope_along_axis) -> float`. Werden von Task 8 (`tunnels/tunnel_mesh.py`) konsumiert.

- [ ] **Step 1: Failing Tests schreiben**

`tests/tunnels/__init__.py`: leer.

`tests/tunnels/test_portal.py`:

```python
"""Tests für world_to_beamng.tunnels.portal: Portal-Stirnflächen von Tunneln folgen der natürlichen Hangneigung
statt rechtwinklig zur Achse abgeschnitten zu werden (siehe Design-Spec Abschnitt 5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.portal import portal_axial_shift, sample_slope_along_axis


def test_sample_slope_along_axis_is_zero_on_flat_ground():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 100.0)

    slope = sample_slope_along_axis(ground_at, (0.0, 0.0), (1.0, 0.0), sample_dist=5.0)

    assert slope == pytest.approx(0.0)


def test_sample_slope_along_axis_matches_a_known_incline():
    # Gelände steigt 1 m je 10 m in +x-Richtung
    ground_at = lambda x, y: 100.0 + 0.1 * np.asarray(x, float)

    slope = sample_slope_along_axis(ground_at, (20.0, 0.0), (1.0, 0.0), sample_dist=5.0)

    assert slope == pytest.approx(0.1)


def test_sample_slope_along_axis_flips_sign_with_direction():
    ground_at = lambda x, y: 100.0 + 0.1 * np.asarray(x, float)

    forward = sample_slope_along_axis(ground_at, (20.0, 0.0), (1.0, 0.0), sample_dist=5.0)
    backward = sample_slope_along_axis(ground_at, (20.0, 0.0), (-1.0, 0.0), sample_dist=5.0)

    assert forward == pytest.approx(-backward)


def test_portal_axial_shift_is_zero_at_floor_level():
    assert portal_axial_shift(0.0, slope_along_axis=0.2) == pytest.approx(0.0)


def test_portal_axial_shift_grows_with_height_and_slope():
    assert portal_axial_shift(5.0, slope_along_axis=0.2) == pytest.approx(1.0)
    assert portal_axial_shift(5.0, slope_along_axis=0.0) == pytest.approx(0.0)
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/tunnels/test_portal.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.tunnels'`

- [ ] **Step 3: Implementierung schreiben**

`world_to_beamng/tunnels/__init__.py`: leer.

`world_to_beamng/tunnels/portal.py`:

```python
"""
Gemeinsame Portal-Geometrie für Tunnel: die Stirnfläche wird nicht rechtwinklig zur Tunnelachse abgeschnitten,
sondern an die natürliche Hangneigung angepasst (siehe Design-Spec Abschnitt 5). Die Heightmap selbst bleibt
dabei unverändert - nur der Portal-Rahmen (tunnels/tunnel_mesh.py::build_portal_frame_mesh()) wird entlang der
Achse verschoben, abhängig von der Höhe über dem Boden.
"""

from typing import Callable, Tuple

import numpy as np

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sample_slope_along_axis(ground_at: HeightAt, point_xy: Tuple[float, float], axis_direction: Tuple[float, float], sample_dist: float) -> float:
    """
    Hangneigung entlang `axis_direction` am Punkt (Steigung, positiv = Gelände wird in Achsrichtung höher).

    `axis_direction` ist ein Einheitsvektor; typischerweise zeigt er vom Portal INS Tunnelinnere.
    """
    ax, ay = axis_direction
    forward = (point_xy[0] + ax * sample_dist, point_xy[1] + ay * sample_dist)
    backward = (point_xy[0] - ax * sample_dist, point_xy[1] - ay * sample_dist)
    h_forward = float(ground_at(np.array([forward[0]]), np.array([forward[1]]))[0])
    h_backward = float(ground_at(np.array([backward[0]]), np.array([backward[1]]))[0])
    return (h_forward - h_backward) / (2.0 * sample_dist)


def portal_axial_shift(height_above_floor: float, slope_along_axis: float) -> float:
    """
    Achsversatz (in Metern, in Richtung `axis_direction`) eines Portal-Ring-Punkts je nach Höhe über dem Boden.

    Bei steigendem Gelände (slope_along_axis > 0, `axis_direction` zeigt ins Tunnelinnere) rückt die Decke
    weiter ins Tunnelinnere (positiver Versatz) als der Boden (Versatz 0) - der Rahmen wirkt, als wäre er schräg
    in den Hang gesetzt.
    """
    return slope_along_axis * height_above_floor
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/tunnels/test_portal.py -v`
Expected: PASS (5 Tests)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/tunnels/__init__.py world_to_beamng/tunnels/portal.py tests/tunnels/__init__.py tests/tunnels/test_portal.py
git commit -m "feat: Add slope-adapted portal geometry helper for tunnels

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 8: Tunnel-Mesh-Builder (kreisrunde Röhre + Portal-Rahmen)

**Files:**
- Create: `world_to_beamng/tunnels/tunnel_mesh.py`
- Test: `tests/tunnels/test_tunnel_mesh.py`

**Interfaces:**
- Consumes: `MeshBuilder`, `offset_points` aus `world_to_beamng.walls.mesh_parts`; `sample_slope_along_axis`, `portal_axial_shift` aus Task 7.
- Produces: `tunnel_radius(width) -> float`; `tunnel_crown_height(width) -> float`; `arc_cross_section(radius, segments) -> list[tuple]`; `resample_tunnel_coords(coords, step) -> list[tuple]`; `build_tunnel_mesh(coords, width, floor_material, wall_material, arc_segments=12, tile_m=5.0) -> dict`; `portal_frame_corners(...) -> list[list[float]]`; `build_portal_frame_mesh(...) -> dict`; `build_tunnel(tunnel: dict, ground_at, wall_material, frame_material, width_margin, arc_segments, segment_step, portal_slope_sample_dist, frame_margin) -> list[dict]`; `build_tunnels(tunnels: list[dict], ...) -> list[dict]` (jedes `tunnels`-Item: `{"id","coords","width","floor_material"}`). Wird von Task 11 (`TerrainWorkflow._build_tunnels`) konsumiert.

**Geometrie (Standard-Tunnelprofil, kreisrund):** Die Bodensehne (Fahrbahnbreite `width`) spannt einen Kreis so,
dass der Bogen über der Fahrbahn 240° misst und die restlichen 120° darunter liegen (nicht modelliert - die
unsichtbare, mit Fundament/Entwässerung gefüllte Sohle). Bei dieser Aufteilung gilt `width = sqrt(3) * R`, also
`R = width / sqrt(3)`; die Kronenhöhe (Boden bis Scheitelpunkt) ist `1.5 * R`. Radius und Höhe sind damit keine
unabhängigen Größen mehr, sondern ergeben sich aus der Breite (siehe Design-Spec Abschnitt 5).

- [ ] **Step 1: Failing Tests schreiben**

`tests/tunnels/test_tunnel_mesh.py`:

```python
"""Tests für world_to_beamng.tunnels.tunnel_mesh: kreisrunde Tunnelröhre (Standardprofil: 240° Bogen über der
Fahrbahn, Boden als Sehne, Radius/Höhe aus der Breite abgeleitet) + hangneigungs-angepasste Portal-Rahmen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_mesh import (
    build_tunnel,
    build_tunnel_mesh,
    build_tunnels,
    portal_frame_corners,
    resample_tunnel_coords,
    tunnel_crown_height,
    tunnel_radius,
)

WALL, FRAME, FLOOR = "tunnel_concrete", "tunnel_concrete", "asphalt_road_standard"


def _straight_coords(length=100.0, z=500.0, n=11):
    return [(x, 0.0, z) for x in np.linspace(0.0, length, n)]


def test_tunnel_radius_and_crown_height_follow_the_floor_width():
    # Bodensehne = sqrt(3) * R bei 240°/120°-Aufteilung -> R = Breite / sqrt(3); exakt gewählt, damit R = 8
    width = 8.0 * math.sqrt(3.0)
    assert tunnel_radius(width) == pytest.approx(8.0)
    assert tunnel_crown_height(width) == pytest.approx(12.0)  # 1.5 * R


def test_tube_floor_is_flat_and_matches_the_road_width():
    mesh = build_tunnel_mesh(_straight_coords(z=500.0), width=8.0, floor_material=FLOOR, wall_material=WALL)
    v = mesh["vertices"]

    assert v[:, 2].min() == pytest.approx(500.0)  # Boden = Höhenprofil
    assert v[:, 1].min() == pytest.approx(-4.0) and v[:, 1].max() == pytest.approx(4.0)  # Bodenbreite = 8 m


def test_crown_reaches_the_derived_height_above_the_floor():
    width = 8.0
    mesh = build_tunnel_mesh(_straight_coords(z=500.0), width=width, floor_material=FLOOR, wall_material=WALL, arc_segments=12)

    assert mesh["vertices"][:, 2].max() == pytest.approx(500.0 + tunnel_crown_height(width), abs=1e-6)


def test_tube_faces_are_split_by_material():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    assert set(mesh["faces"]) == {FLOOR, WALL}
    assert len(mesh["faces"][FLOOR]) == 2 * 2  # 2 Segmente, Boden = 1 Quad = 2 Dreiecke je Segment
    assert len(mesh["faces"][WALL]) == 2 * 6 * 2  # 6 Bogen-Streifen je Segment, je 2 Dreiecke


def test_floor_normal_points_up_into_the_tube():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    assert (0.0, 0.0, 1.0) in {tuple(np.round(n, 2)) for n in mesh["normals"]}


def test_arc_normals_are_unit_length_and_do_not_point_straight_up():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    wall_normals = np.array([mesh["normals"][face[0]] for face in mesh["faces"][WALL]])
    assert np.allclose(np.linalg.norm(wall_normals, axis=1), 1.0, atol=1e-6)
    assert not np.any(np.all(np.isclose(wall_normals, [0.0, 0.0, 1.0], atol=1e-3), axis=1))


def test_floor_and_arc_share_exact_edge_vertices_even_on_a_curve():
    # arc_ring() verwendet an den Bodenrändern exakt right[i]/left[i] wie das Boden-Mesh - sonst entstünde bei
    # einer Kurve (unterschiedliche Segment-Richtungen) ein Spalt zwischen Boden und Bogen.
    coords = [(0.0, 0.0, 500.0), (20.0, 2.0, 500.0), (40.0, 0.0, 500.0)]
    mesh = build_tunnel_mesh(coords, width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    floor_idx = {i for tri in mesh["faces"][FLOOR] for i in tri}
    wall_idx = {i for tri in mesh["faces"][WALL] for i in tri}
    floor_points = {tuple(np.round(mesh["vertices"][i], 3)) for i in floor_idx}
    wall_points = {tuple(np.round(mesh["vertices"][i], 3)) for i in wall_idx}

    assert len(floor_points & wall_points) >= 4  # beide Bodenrand-Ringe (Anfang+Ende) sind gemeinsame Punkte


def test_resample_tunnel_coords_reduces_point_count_for_a_long_tunnel():
    coords = _straight_coords(length=17000.0, z=500.0, n=17001)  # 1 m Abstand wie aus der normalen Pipeline

    resampled = resample_tunnel_coords(coords, step=10.0)

    assert len(resampled) < len(coords) / 5
    assert resampled[0] == pytest.approx(coords[0])
    assert resampled[-1][0] == pytest.approx(coords[-1][0], abs=1e-6)


def test_resample_tunnel_coords_leaves_short_tunnels_unchanged():
    coords = _straight_coords(length=5.0, z=500.0, n=6)

    assert resample_tunnel_coords(coords, step=10.0) == coords


def test_portal_frame_corners_are_a_flat_rectangle_on_flat_ground():
    corners = np.array(portal_frame_corners((0.0, 0.0), (1.0, 0.0), width=8.0, height=12.0, margin=0.6, floor_z=500.0, slope_along_axis=0.0))

    assert corners[:, 0].max() == pytest.approx(0.0)  # keine Achsverschiebung bei Neigung 0
    assert corners[:, 1].min() == pytest.approx(-4.6) and corners[:, 1].max() == pytest.approx(4.6)
    assert corners[:, 2].min() == pytest.approx(500.0) and corners[:, 2].max() == pytest.approx(512.6)


def test_portal_frame_corners_shift_the_top_edge_with_slope():
    corners = np.array(portal_frame_corners((0.0, 0.0), (1.0, 0.0), width=8.0, height=12.0, margin=0.0, floor_z=500.0, slope_along_axis=0.2))

    assert corners[0, 0] == pytest.approx(0.0) and corners[1, 0] == pytest.approx(0.0)  # untere Ecken unverschoben
    assert corners[2, 0] == pytest.approx(2.4) and corners[3, 0] == pytest.approx(2.4)  # obere Ecken: 0.2 * 12m = 2.4m verschoben


def test_build_tunnel_returns_the_tube_plus_two_portal_frames_with_derived_height():
    tunnel = {"id": 42, "coords": _straight_coords(length=200.0, z=500.0), "width": 7.0, "floor_material": FLOOR}
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)

    meshes = build_tunnel(tunnel, ground_at, WALL, FRAME, width_margin=1.5, arc_segments=12, segment_step=10.0, portal_slope_sample_dist=5.0, frame_margin=0.6)

    assert [m["id"] for m in meshes] == ["tunnel_42", "tunnel_42_portal_start", "tunnel_42_portal_end"]
    assert FLOOR in meshes[0]["faces"]
    assert FRAME in meshes[1]["faces"] and FRAME in meshes[2]["faces"]
    expected_crown = tunnel_crown_height(7.0 + 1.5)
    assert max(v[2] for v in meshes[1]["vertices"]) == pytest.approx(500.0 + expected_crown)


def test_build_tunnels_skips_too_short_tunnels():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)
    tunnels = [{"id": 1, "coords": [(0.0, 0.0, 500.0)], "width": 7.0, "floor_material": FLOOR}]

    assert build_tunnels(tunnels, ground_at, WALL, FRAME) == []
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/tunnels/test_tunnel_mesh.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.tunnels.tunnel_mesh'`

- [ ] **Step 3: Implementierung schreiben**

`world_to_beamng/tunnels/tunnel_mesh.py`:

```python
"""
Tunnel aus OSM-Linien (highway=* mit tunnel=yes/culvert/building_passage): kreisrunde Röhre - Standard-
Tunnelprofil, 240° Kreisbogen über einer flachen Bodensehne (Fahrbahn), die restlichen 120° liegen unterhalb der
Sehne und werden nicht modelliert (unsichtbare Sohle) - entlang des linear interpolierten Höhenprofils (siehe
geometry/road_structures.py + geometry/polygon.py), mit an die natürliche Hangneigung angepassten Portal-Rahmen
an beiden Enden (siehe Design-Spec Abschnitt 5). Die Röhre selbst hat rechtwinklige (nicht geschnittene) Enden -
die Schräge steckt im separaten, flachen Portal-Rahmen-Mesh, der die Öffnung umgibt (der Rahmen ist von außen
sichtbar, das Rohr-Ende dahinter nicht). Die Heightmap bleibt unverändert - die Röhre liegt "im Berg".
"""

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, offset_points
from .portal import portal_axial_shift, sample_slope_along_axis

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]

ARC_SPAN_DEG = 240.0  # Kreisbogen über der Fahrbahn
ARC_START_DEG = -30.0  # Startwinkel (rechter Bodenrand), Standardkreis-Konvention (0°=+x, CCW)


def tunnel_radius(width: float) -> float:
    """Radius der kreisrunden Tunnelröhre aus der Bodenbreite (Bodensehne = sqrt(3)*R bei 240°/120°-Aufteilung)."""
    return width / math.sqrt(3.0)


def tunnel_crown_height(width: float) -> float:
    """Lichte Höhe (Boden bis Kronenscheitel) einer kreisrunden Tunnelröhre der gegebenen Bodenbreite."""
    return 1.5 * tunnel_radius(width)


def arc_cross_section(radius: float, segments: int) -> List[Tuple[float, float]]:
    """
    (across, height)-Punkte des 240°-Kreisbogens über der Fahrbahn, `segments` Streifen (segments+1 Punkte), vom
    rechten Bodenrand (θ=-30°) über die Krone (θ=90°) zum linken Bodenrand (θ=210°). Boden ist y=0, "across" ist
    quer zur Fahrtrichtung (positiv = rechts). Kreismittelpunkt liegt bei (0, radius/2) - siehe Design-Spec
    Abschnitt 5 für die Herleitung.
    """
    points = []
    for k in range(segments + 1):
        theta = math.radians(ARC_START_DEG + (k / segments) * ARC_SPAN_DEG)
        points.append((radius * math.cos(theta), radius / 2.0 + radius * math.sin(theta)))
    return points


def build_tunnel_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    floor_material: str,
    wall_material: str,
    arc_segments: int = 12,
    tile_m: float = 5.0,
) -> Dict:
    """
    Röhren-Mesh (Boden + kreisrunder 240°-Bogen darüber) entlang `coords` (bereits das Tunnel-Höhenprofil).
    Radius und Kronenhöhe ergeben sich aus `width` (siehe tunnel_radius()/tunnel_crown_height()).

    Returns:
        {"vertices", "uvs", "normals", "faces": {floor_material: [...], wall_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    floor_z = points[:, 2]
    radius = tunnel_radius(width)
    arc = arc_cross_section(radius, arc_segments)

    left, right = offset_points(xy, width / 2.0, closed=False)

    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(seg_len)]) / tile_m
    across_floor = width / tile_m
    across_arc = (radius * math.radians(ARC_SPAN_DEG)) / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    def arc_ring(i: int, k: int, perp_right: np.ndarray) -> List[float]:
        """Weltposition des Ring-Punkts k (0=rechter Bodenrand, arc_segments=linker Bodenrand) an Centerline-Punkt
        i. Die beiden Bodenrand-Punkte sind exakt right[i]/left[i] (nahtlos zum Boden-Mesh), die Zwischenpunkte
        folgen dem Kreisbogen relativ zur Segment-Richtung (kleine Facette an Kurven statt Gehrung wie bei
        offset_points() - unauffällig bei der groben Tunnel-Resampling-Schrittweite)."""
        if k == 0:
            return p3(right[i], floor_z[i])
        if k == arc_segments:
            return p3(left[i], floor_z[i])
        ax, ay = arc[k]
        return [float(xy[i, 0] + perp_right[0] * ax), float(xy[i, 1] + perp_right[1] * ax), float(floor_z[i] + ay)]

    floor_builder = MeshBuilder()
    wall_builder = MeshBuilder()
    for i in range(len(points) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        perp_right = np.array([direction[1], -direction[0]])  # zeigt "rechts" der Laufrichtung

        # Boden (Normale nach oben, ins Rohrinnere)
        floor_builder.quad(
            [p3(left[i], floor_z[i]), p3(left[j], floor_z[j]), p3(right[j], floor_z[j]), p3(right[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across_floor], [u0, across_floor]],
            [0.0, 0.0, 1.0],
        )

        # Kreisbogen (240°) über der Fahrbahn, in arc_segments Streifen
        for k in range(arc_segments):
            theta_mid = math.radians(ARC_START_DEG + ((k + 0.5) / arc_segments) * ARC_SPAN_DEG)
            inward = [-math.cos(theta_mid) * perp_right[0], -math.cos(theta_mid) * perp_right[1], -math.sin(theta_mid)]
            v0 = (k / arc_segments) * across_arc
            v1 = ((k + 1) / arc_segments) * across_arc
            wall_builder.quad(
                [arc_ring(i, k, perp_right), arc_ring(j, k, perp_right), arc_ring(j, k + 1, perp_right), arc_ring(i, k + 1, perp_right)],
                [[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
                inward,
            )

    all_vertices = floor_builder.vertices + wall_builder.vertices
    all_uvs = floor_builder.uvs + wall_builder.uvs
    all_normals = floor_builder.normals + wall_builder.normals
    wall_offset = len(floor_builder.vertices)
    wall_faces = [[a + wall_offset, b + wall_offset, c + wall_offset] for a, b, c in wall_builder.faces]

    return {
        "vertices": np.array(all_vertices, dtype=float),
        "uvs": np.array(all_uvs, dtype=float),
        "normals": np.array(all_normals, dtype=float),
        "faces": {floor_material: floor_builder.faces, wall_material: wall_faces},
    }


def portal_frame_corners(
    xy_point: Tuple[float, float],
    axis_direction: Tuple[float, float],
    width: float,
    height: float,
    margin: float,
    floor_z: float,
    slope_along_axis: float,
) -> List[List[float]]:
    """
    4 Eckpunkte (Weltkoordinaten) eines Portal-Rahmen-Rings: unten-links, unten-rechts, oben-rechts, oben-links.
    `margin` vergrößert den Ring gegenüber der reinen Röhrenöffnung (0.0 = deckt genau die Öffnung ab). Die
    oberen Ecken sind entlang `axis_direction` verschoben (siehe portal.portal_axial_shift()), damit der Ring der
    natürlichen Hangneigung folgt statt rechtwinklig zur Achse zu stehen.
    """
    p = np.array(xy_point, dtype=float)
    axis = np.array(axis_direction, dtype=float)
    perp = np.array([-axis[1], axis[0]])
    half_w = width / 2.0 + margin

    bottom_shift = portal_axial_shift(0.0, slope_along_axis)
    top_shift = portal_axial_shift(height + margin, slope_along_axis)
    bottom = p + axis * bottom_shift
    top = p + axis * top_shift

    bl = [float(bottom[0] - perp[0] * half_w), float(bottom[1] - perp[1] * half_w), floor_z]
    br = [float(bottom[0] + perp[0] * half_w), float(bottom[1] + perp[1] * half_w), floor_z]
    tr = [float(top[0] + perp[0] * half_w), float(top[1] + perp[1] * half_w), floor_z + height + margin]
    tl = [float(top[0] - perp[0] * half_w), float(top[1] - perp[1] * half_w), floor_z + height + margin]
    return [bl, br, tr, tl]


def build_portal_frame_mesh(
    xy_point: Tuple[float, float],
    axis_direction: Tuple[float, float],
    width: float,
    height: float,
    floor_z: float,
    slope_along_axis: float,
    frame_margin: float,
    material: str,
) -> Dict:
    """Flacher Rahmen (4 Trapez-Flächen) um die Tunnelöffnung, an die Hangneigung angepasst (siehe portal_frame_corners())."""
    outer = portal_frame_corners(xy_point, axis_direction, width, height, frame_margin, floor_z, slope_along_axis)
    inner = portal_frame_corners(xy_point, axis_direction, width, height, 0.0, floor_z, slope_along_axis)
    axis = np.array(axis_direction, dtype=float)
    normal = [float(-axis[0]), float(-axis[1]), 0.0]  # zeigt vom Tunnelinneren weg (nach außen, sichtbare Seite)

    builder = MeshBuilder()
    for i in range(4):
        j = (i + 1) % 4
        builder.quad([outer[i], outer[j], inner[j], inner[i]], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], normal)

    return {
        "vertices": np.array(builder.vertices, dtype=float),
        "uvs": np.array(builder.uvs, dtype=float),
        "normals": np.array(builder.normals, dtype=float),
        "faces": {material: builder.faces},
    }


def build_tunnel(
    tunnel: Dict,
    ground_at: HeightAt,
    wall_material: str,
    frame_material: str,
    width_margin: float,
    arc_segments: int,
    segment_step: float,
    portal_slope_sample_dist: float,
    frame_margin: float,
) -> List[Dict]:
    """Tunnelröhre (Kreisbogen-Profil) + zwei Portal-Rahmen für einen Tunnel-Way (`tunnel`: {"id","coords","width","floor_material"})."""
    coords = resample_tunnel_coords(tunnel["coords"], segment_step)
    if len(coords) < 2:
        return []
    width = tunnel["width"] + width_margin
    crown_height = tunnel_crown_height(width)
    points = np.array(coords, dtype=float)

    tube = build_tunnel_mesh(coords, width, tunnel["floor_material"], wall_material, arc_segments=arc_segments)
    meshes = [{"id": f"tunnel_{tunnel['id']}", **tube}]

    for index, neighbour, label in ((0, 1, "start"), (len(points) - 1, len(points) - 2, "end")):
        direction = points[neighbour, :2] - points[index, :2]
        direction = direction / np.linalg.norm(direction)
        axis_direction = (float(direction[0]), float(direction[1]))
        slope = sample_slope_along_axis(ground_at, tuple(points[index, :2]), axis_direction, portal_slope_sample_dist)
        frame = build_portal_frame_mesh(
            tuple(points[index, :2]), axis_direction, width, crown_height, float(points[index, 2]), slope, frame_margin, frame_material
        )
        meshes.append({"id": f"tunnel_{tunnel['id']}_portal_{label}", **frame})
    return meshes


def build_tunnels(
    tunnels: Sequence[Dict],
    ground_at: HeightAt,
    wall_material: str,
    frame_material: str,
    width_margin: float = 1.5,
    arc_segments: int = 12,
    segment_step: float = 10.0,
    portal_slope_sample_dist: float = 5.0,
    frame_margin: float = 0.6,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, drei je Tunnel (`tunnels`: [{"id","coords","width","floor_material"}, ...])."""
    meshes = []
    for tunnel in tunnels:
        meshes.extend(build_tunnel(tunnel, ground_at, wall_material, frame_material, width_margin, arc_segments, segment_step, portal_slope_sample_dist, frame_margin))
    return meshes
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/tunnels/test_tunnel_mesh.py -v`
Expected: PASS (13 Tests)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/tunnels/tunnel_mesh.py tests/tunnels/test_tunnel_mesh.py
git commit -m "feat: Add tunnel tube + slope-adapted portal frame mesh builder

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 9: Galerie-Mesh-Builder (talseitig offen)

**Files:**
- Create: `world_to_beamng/tunnels/gallery_mesh.py`
- Test: `tests/tunnels/test_gallery_mesh.py`

**Interfaces:**
- Consumes: `MeshBuilder`, `offset_points`, `add_box_column` aus `world_to_beamng.walls.mesh_parts` (letzteres aus Task 4).
- Produces: `valley_side(xy, ground_at, half_width) -> np.ndarray`; `build_gallery_mesh(coords, width, height, ground_at, floor_material, roof_material, column_spacing=6.0, roof_thickness=0.35, column_size=0.4, tile_m=5.0) -> dict`; `build_galleries(galleries: list[dict], ground_at, roof_material, height=5.0, column_spacing=6.0, roof_thickness=0.35, column_size=0.4) -> list[dict]` (jedes `galleries`-Item: `{"id","coords","width","floor_material"}`). Wird von Task 11 konsumiert.

- [ ] **Step 1: Failing Tests schreiben**

`tests/tunnels/test_gallery_mesh.py`:

```python
"""Tests für world_to_beamng.tunnels.gallery_mesh: talseitig offene Lawinengalerie (Dach + Stützen)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.gallery_mesh import build_gallery_mesh, build_galleries, valley_side

FLOOR, ROOF = "asphalt_road_standard", "tunnel_concrete"


def _straight_coords(length=60.0, z=500.0, n=13):
    return [(x, 0.0, z) for x in np.linspace(0.0, length, n)]


def test_valley_side_picks_the_lower_natural_terrain():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    # Gelände fällt nach +y ab -> rechts der Laufrichtung (+x) liegt bei +y, ist also die Talseite
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)

    side = valley_side(xy, ground_at, half_width=4.0)

    assert np.all(side > 0)


def test_valley_side_flips_when_the_slope_is_mirrored():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    ground_at = lambda x, y: 500.0 + 2.0 * np.asarray(y, float)  # steigt nach +y -> links ist die Talseite

    side = valley_side(xy, ground_at, half_width=4.0)

    assert np.all(side < 0)


def test_roof_and_floor_are_flat_at_the_given_heights():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF, roof_thickness=0.35)
    v = mesh["vertices"]

    assert v[:, 2].min() == pytest.approx(500.0)
    assert v[:, 2].max() == pytest.approx(505.35)  # Boden(500) + Höhe(5) + Dachdicke(0.35)


def test_faces_are_split_by_material():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    assert set(mesh["faces"]) == {FLOOR, ROOF}
    assert len(mesh["faces"][FLOOR]) > 0 and len(mesh["faces"][ROOF]) > 0


def test_columns_are_placed_on_the_open_valley_side():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # Talseite = +y = rechts
    mesh = build_gallery_mesh(_straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF, column_spacing=10.0)

    roof_vertices = np.array(mesh["vertices"])
    # Stützen-Vertices liegen bei y nahe der rechten Kante (Talseite), nicht bei y=0 (Mitte) oder links
    near_right_edge = roof_vertices[np.abs(roof_vertices[:, 1] - 4.0) < 0.5]
    assert len(near_right_edge) > 0


def test_build_galleries_returns_one_mesh_per_gallery():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    galleries = [{"id": 1, "coords": _straight_coords(z=500.0), "width": 8.0, "floor_material": FLOOR}]

    meshes = build_galleries(galleries, ground_at, roof_material=ROOF, height=5.0)

    assert [m["id"] for m in meshes] == ["gallery_1"]


def test_build_galleries_skips_degenerate_galleries():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)
    galleries = [{"id": 1, "coords": [(0.0, 0.0, 500.0)], "width": 8.0, "floor_material": FLOOR}]

    assert build_galleries(galleries, ground_at, roof_material=ROOF) == []
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/tunnels/test_gallery_mesh.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.tunnels.gallery_mesh'`

- [ ] **Step 3: Implementierung schreiben**

`world_to_beamng/tunnels/gallery_mesh.py`:

```python
"""
Galerien aus OSM-Linien (highway=* mit tunnel=avalanche_protector): wie ein Tunnel, aber talseitig offen (Dach +
Stützen statt einer zweiten Wand) - siehe Design-Spec Abschnitt 6. Keine Portal-Rahmen: Galerien sind keine in
den Fels geschnittenen Öffnungen, sondern offene Schutzbauten entlang der Straße - ihre Enden bleiben rechtwinklig.
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column, offset_points

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def valley_side(xy: np.ndarray, ground_at: HeightAt, half_width: float) -> np.ndarray:
    """
    Pro Punkt: +1.0, wenn die Seite RECHTS der Laufrichtung talwärts liegt (niedrigere natürliche Geländehöhe),
    sonst -1.0 (links talwärts). Gleiche Technik wie terrain.road_embedding.build_road_embankment_profiles()
    (natürliche Geländehöhe links/rechts der Centerline vergleichen).
    """
    directions = np.diff(xy, axis=0)
    directions = np.vstack([directions, directions[-1:]])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    directions = directions / norms
    perp = np.column_stack([-directions[:, 1], directions[:, 0]])

    left_xy = xy - perp * half_width
    right_xy = xy + perp * half_width
    left_z = np.asarray(ground_at(left_xy[:, 0], left_xy[:, 1]), dtype=float)
    right_z = np.asarray(ground_at(right_xy[:, 0], right_xy[:, 1]), dtype=float)
    return np.where(right_z < left_z, 1.0, -1.0)


def build_gallery_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    height: float,
    ground_at: HeightAt,
    floor_material: str,
    roof_material: str,
    column_spacing: float = 6.0,
    roof_thickness: float = 0.35,
    column_size: float = 0.4,
    tile_m: float = 5.0,
) -> Dict:
    """
    Galerie-Mesh: Boden, Dach (Ober-/Unterseite), eine bergseitige Wand und Stützen auf der talseitig offenen Seite.

    Returns:
        {"vertices", "uvs", "normals", "faces": {floor_material: [...], roof_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    floor_z = points[:, 2]
    roof_bottom_z = floor_z + height
    roof_top_z = roof_bottom_z + roof_thickness

    left, right = offset_points(xy, width / 2.0, closed=False)
    side = valley_side(xy, ground_at, width / 2.0)  # +1 = rechts offen (Tal), -1 = links offen

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)]) / tile_m
    across = width / tile_m
    across_h = height / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    floor_builder = MeshBuilder()
    roof_builder = MeshBuilder()
    for i in range(len(points) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        side_normal = [float(-direction[1]), float(direction[0]), 0.0]

        floor_builder.quad(
            [p3(left[i], floor_z[i]), p3(left[j], floor_z[j]), p3(right[j], floor_z[j]), p3(right[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )
        roof_builder.quad(
            [p3(left[i], roof_bottom_z[i]), p3(right[i], roof_bottom_z[i]), p3(right[j], roof_bottom_z[j]), p3(left[j], roof_bottom_z[j])],
            [[u0, 0.0], [u0, across], [u1, across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        roof_builder.quad(
            [p3(left[i], roof_top_z[i]), p3(left[j], roof_top_z[j]), p3(right[j], roof_top_z[j]), p3(right[i], roof_top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )

        # bergseitige Wand: die Seite, die (an diesem Segment) NICHT talwärts liegt; bei einem Wechsel mitten im
        # Segment (selten) gewinnt die Seite am Segment-Anfang - akzeptierte Vereinfachung.
        mountain_is_left = side[i] > 0
        edge = left if mountain_is_left else right
        wall_normal = [-side_normal[0], -side_normal[1], 0.0] if mountain_is_left else side_normal
        roof_builder.quad(
            [p3(edge[i], floor_z[i]), p3(edge[j], floor_z[j]), p3(edge[j], roof_bottom_z[j]), p3(edge[i], roof_bottom_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across_h], [u0, across_h]],
            wall_normal,
        )

    cum = np.concatenate([[0.0], np.cumsum(steps)])
    total_len = float(cum[-1]) if len(cum) else 0.0
    column_positions = np.arange(column_spacing / 2.0, total_len, column_spacing) if total_len > 0 else np.array([])
    for s in column_positions:
        idx = max(1, min(int(np.searchsorted(cum, s)), len(points) - 1))
        t = (s - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
        open_edge = right if side[idx - 1] > 0 else left
        cx = open_edge[idx - 1, 0] + t * (open_edge[idx, 0] - open_edge[idx - 1, 0])
        cy = open_edge[idx - 1, 1] + t * (open_edge[idx, 1] - open_edge[idx - 1, 1])
        cz = float(floor_z[idx - 1] + t * (floor_z[idx] - floor_z[idx - 1]))
        add_box_column(roof_builder, cx, cy, cz, cz + height, column_size, tile_m)

    all_vertices = floor_builder.vertices + roof_builder.vertices
    all_uvs = floor_builder.uvs + roof_builder.uvs
    all_normals = floor_builder.normals + roof_builder.normals
    roof_offset = len(floor_builder.vertices)
    roof_faces = [[a + roof_offset, b + roof_offset, c + roof_offset] for a, b, c in roof_builder.faces]

    return {
        "vertices": np.array(all_vertices, dtype=float),
        "uvs": np.array(all_uvs, dtype=float),
        "normals": np.array(all_normals, dtype=float),
        "faces": {floor_material: floor_builder.faces, roof_material: roof_faces},
    }


def build_galleries(
    galleries: Sequence[Dict],
    ground_at: HeightAt,
    roof_material: str,
    height: float = 5.0,
    column_spacing: float = 6.0,
    roof_thickness: float = 0.35,
    column_size: float = 0.4,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, eines je Galerie (`galleries`: [{"id","coords","width","floor_material"}, ...])."""
    meshes = []
    for gallery in galleries:
        coords = gallery["coords"]
        if len(coords) < 2:
            continue
        mesh = build_gallery_mesh(
            coords, gallery["width"], height, ground_at, gallery["floor_material"], roof_material,
            column_spacing=column_spacing, roof_thickness=roof_thickness, column_size=column_size,
        )
        meshes.append({"id": f"gallery_{gallery['id']}", **mesh})
    return meshes
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/tunnels/test_gallery_mesh.py -v`
Expected: PASS (7 Tests)

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/tunnels/gallery_mesh.py tests/tunnels/test_gallery_mesh.py
git commit -m "feat: Add valley-side-open gallery mesh builder

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 10: Prozedurale Beton-Textur

**Files:**
- Create: `world_to_beamng/textures/concrete.py`
- Modify: `world_to_beamng/textures/registry.py` (neuer `TextureSpec`)
- Test: `tests/textures/test_concrete.py`
- Modify: `tests/textures/test_registry.py` (ein neuer Test am Ende der Datei)

**Interfaces:**
- Consumes: `fbm`, `periodic_noise`, `normal_from_height`, `to_uint8`, `gray_to_rgb` aus `world_to_beamng.facade.texture_utils`; `library.store_texture` aus `world_to_beamng.textures.library` (bestehend, wie in `textures/gravel.py` verwendet).
- Produces: `ConcreteTextureGenerator(size_px, repeat_m).generate(seed) -> dict` (`{"albedo","normal","roughness"}` als uint8 RGB); `generate_concrete_texture(library_dir=None, seed=8181) -> Path`. Wird von `textures/registry.py` (`generate=generate_concrete_texture`) und in Task 6/11 über `registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]` konsumiert.

- [ ] **Step 1: Failing Tests schreiben**

`tests/textures/test_concrete.py`:

```python
"""Tests für world_to_beamng.textures.concrete: prozedurale Beton-Textur für Brücken/Tunnel/Galerien."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng import config
from world_to_beamng.textures.concrete import ConcreteTextureGenerator, generate_concrete_texture


def test_generated_maps_have_the_configured_size_and_are_valid_rgb_images():
    maps = ConcreteTextureGenerator(size_px=64, repeat_m=2.0).generate(seed=1)

    for key in ("albedo", "normal", "roughness"):
        assert maps[key].shape == (64, 64, 3)
        assert maps[key].dtype == np.uint8


def test_generation_is_deterministic_for_the_same_seed():
    a = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=7)
    b = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=7)

    np.testing.assert_array_equal(a["albedo"], b["albedo"])


def test_different_seeds_produce_different_textures():
    a = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=1)
    b = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=2)

    assert not np.array_equal(a["albedo"], b["albedo"])


def test_generate_concrete_texture_stores_it_in_the_library(tmp_path):
    folder = generate_concrete_texture(library_dir=tmp_path, seed=3)

    assert folder == tmp_path / config.CONCRETE_TEXTURE_NAME
    for channel in ("color", "normal", "roughness"):
        assert (folder / f"{channel}.png").exists()
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert config.CONCRETE_TEXTURE_NAME in manifest
```

Am Ende von `tests/textures/test_registry.py` ergänzen:

```python
def test_concrete_texture_is_registered_when_bridges_or_tunnels_are_enabled(monkeypatch):
    from world_to_beamng import config
    from world_to_beamng.textures import registry

    monkeypatch.setattr(config, "BRIDGES_ENABLED", True)
    monkeypatch.setattr(config, "TUNNELS_ENABLED", False)
    assert any(spec.name == config.CONCRETE_TEXTURE_NAME for spec in registry.REGISTRY if spec.required())

    monkeypatch.setattr(config, "BRIDGES_ENABLED", False)
    monkeypatch.setattr(config, "TUNNELS_ENABLED", False)
    assert not any(spec.name == config.CONCRETE_TEXTURE_NAME for spec in registry.REGISTRY if spec.required())
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/textures/test_concrete.py tests/textures/test_registry.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.textures.concrete'`

- [ ] **Step 3: Implementierung schreiben**

`world_to_beamng/textures/concrete.py`:

```python
"""
Prozedurale, kachelbare Beton-Textur für Brücken-Pfeiler, Tunnel-Wände/-Decke/-Portale und Galerie-Dach/-Stützen -
wird einmalig erzeugt und in data/textures abgelegt (nicht bei jedem Export), automatisch falls sie fehlt
(textures/registry.py).

Schalglatte Fläche: leichtes Grau-Rauschen (große Flecken + feine Körnung) plus eine feine Schalungsstruktur in
der Normalmap.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import config
from ..facade.texture_utils import fbm, gray_to_rgb, normal_from_height, periodic_noise, to_uint8
from . import library


class ConcreteTextureGenerator:
    """Erzeugt Albedo, Normalmap und Roughness einer schalglatten Betonfläche."""

    def __init__(self, size_px: int = config.CONCRETE_TEXTURE_PX, repeat_m: float = config.CONCRETE_TEXTURE_TILE_M):
        self._size = size_px
        self._repeat_m = repeat_m

    def generate(self, seed: Optional[int] = 8181) -> Dict[str, np.ndarray]:
        """
        Returns:
            {"albedo", "normal", "roughness"}: uint8-RGB-Bilder (size_px x size_px x 3)
        """
        rng = np.random.default_rng(seed)
        size = self._size

        base_gray = 0.62 + 0.06 * fbm(size, size, rng, betas=(1.3, 2.2, 3.2), weights=(0.5, 0.3, 0.2))
        stain = 0.05 * periodic_noise(size, size, rng, beta=0.6)  # größere, weiche Wasserflecken
        gray = np.clip(base_gray + stain, 0.0, 1.0)

        albedo = gray_to_rgb(gray).astype(np.float64) / 255.0
        albedo = albedo * (0.92 + 0.08 * fbm(size, size, rng, betas=(4.0,), weights=(1.0,)))[..., None]  # feine Körnung

        height = 0.4 * periodic_noise(size, size, rng, beta=3.0)  # feine Schalungsstruktur
        roughness = np.full((size, size), 0.85)

        return {"albedo": to_uint8(albedo), "normal": normal_from_height(height, 0.6), "roughness": gray_to_rgb(roughness)}


def generate_concrete_texture(library_dir: Optional[Path] = None, seed: int = 8181) -> Path:
    """
    Erzeugt die Beton-Textur und legt sie in der Textur-Bibliothek ab (data/textures/tunnel_concrete).

    Returns:
        Ordner der Textur
    """
    generated = ConcreteTextureGenerator().generate(seed=seed)
    maps = {"color": generated["albedo"], "normal": generated["normal"], "roughness": generated["roughness"]}
    return library.store_texture(
        config.CONCRETE_TEXTURE_NAME,
        maps,
        tile_m=config.CONCRETE_TEXTURE_TILE_M,
        source=f"prozedural (textures/concrete.py), Seed {seed}, {config.CONCRETE_TEXTURE_PX} px",
        library_dir=library_dir,
    )
```

In `world_to_beamng/textures/registry.py`, Import-Block erweitern:

```python
from . import library
from .concrete import generate_concrete_texture
from .gravel import generate_gravel_texture
```

Im `REGISTRY`-Tupel, nach dem `WALL_TEXTURE_NAME`-Eintrag ergänzen:

```python
    TextureSpec(
        config.WALL_TEXTURE_NAME,
        "Bruchsteinmauern (Mauerkörper und Abdeckplatten)",
        required=lambda: config.WALLS_ENABLED,
        hint=f"python tools/make_seamless_texture.py <Foto> --name {config.WALL_TEXTURE_NAME} --width-m <reale Breite des Fotos in Metern>",
    ),
    TextureSpec(
        config.CONCRETE_TEXTURE_NAME,
        "Brücken (Pfeiler), Tunnel (Wände/Decke/Portale), Galerien (Dach/Stützen)",
        required=lambda: config.BRIDGES_ENABLED or config.TUNNELS_ENABLED,
        generate=generate_concrete_texture,
    ),
)
```

(die schließende Klammer `)` gehört zum bestehenden `REGISTRY`-Tupel - der neue Eintrag wird davor eingefügt.)

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/textures/test_concrete.py tests/textures/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/textures/concrete.py world_to_beamng/textures/registry.py tests/textures/test_concrete.py tests/textures/test_registry.py
git commit -m "feat: Add procedural concrete texture for bridges/tunnels/galleries

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Task 11: Tunnel-/Galerie-Integration in TerrainWorkflow + Export

**Files:**
- Modify: `world_to_beamng/config.py` (neue Konstanten)
- Modify: `world_to_beamng/workflow/terrain_workflow.py` (`_build_tunnels()`, `export_tunnels()`, Verdrahtung in `process_tile()`/`export_tile()`)
- Test: `tests/workflow/test_terrain_workflow_tunnels.py`

**Interfaces:**
- Consumes: `build_tunnels` aus Task 8, `build_galleries` aus Task 9; `structure_road_polygons` aus Task 3.
- Produces: `TerrainWorkflow._build_tunnels(structure_road_polygons, heights, terrain_origin_x, terrain_origin_y) -> list[dict]` (Tunnel- + Galerie-Meshes zusammen); `TerrainWorkflow.export_tunnels(mesh_data) -> int`. `process_tile()`-Rückgabe bekommt `"tunnel_meshes"`. `export_tile()` ruft `export_tunnels()` auf.

- [ ] **Step 1: Failing Tests schreiben**

`tests/workflow/test_terrain_workflow_tunnels.py`:

```python
"""Tests für TerrainWorkflow._build_tunnels() und export_tunnels(): Tunnel (Röhre+Portale) und Galerien
(Dach+Stützen) als eine gemeinsame DAE mit einem TSStatic."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.textures import registry
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _Items:
    def __init__(self):
        self.objects = {}

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), **fields):
        self.objects[name] = {"class": item_class, "position": list(position), **fields}


class _Materials:
    def __init__(self):
        self.added = {}

    def get_templates(self):
        return {"buildings": {"wall": {"material_hints": {"groundType": "concrete", "materialTag0": "beamng", "materialTag1": "Building"}}}}

    def add_building_material(self, name, **fields):
        self.added[name] = fields


class _Dae:
    def __init__(self):
        self.calls = []

    def export_multi_mesh(self, output_path, meshes, with_uv=False, material_textures=None):
        self.calls.append((Path(output_path), meshes, with_uv))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<COLLADA/>", encoding="utf-8")
        return output_path


def _stub():
    return SimpleNamespace(items=_Items(), materials=_Materials(), dae=_Dae())


CONCRETE = {
    "baseColorMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_b.color.dds",
    "normalMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_nm.normal.dds",
    "roughnessMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_r.data.dds",
}


@pytest.fixture
def shapes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    monkeypatch.setattr(registry, "prepared_textures", lambda: {config.CONCRETE_TEXTURE_NAME: CONCRETE})
    return tmp_path / "shapes"


def _mesh(name):
    return {
        "id": name,
        "vertices": np.zeros((8, 3)),
        "uvs": np.zeros((8, 2)),
        "normals": np.tile([0.0, 0.0, 1.0], (8, 1)),
        "faces": {"asphalt_road_standard": [[0, 1, 2]], config.TUNNEL_MATERIAL_NAME: [[4, 5, 6]]},
    }


def _road(road_id, structure_type, tags):
    return {
        "road_id": road_id,
        "trimmed_centerline": np.array([[0.0, 0.0, 500.0], [200.0, 0.0, 500.0]]),
        "osm_tags": tags,
        "structure_type": structure_type,
    }


def test_export_tunnels_writes_one_dae_one_item_and_registers_floor_and_concrete_materials(shapes_dir):
    stub = _stub()
    roads = [_road(1, "tunnel", {"highway": "trunk", "tunnel": "yes"}), _road(2, "gallery", {"highway": "primary", "tunnel": "avalanche_protector"})]

    count = TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1"), _mesh("tunnel_1_portal_start"), _mesh("tunnel_1_portal_end"), _mesh("gallery_2")], "structure_road_polygons": roads})

    assert count == 4
    dae_path, meshes, with_uv = stub.dae.calls[0]
    assert dae_path == shapes_dir / "tunnels" / "tunnels.dae" and with_uv is True and len(meshes) == 4
    item = stub.items.objects["tunnels"]
    assert item["class"] == "TSStatic" and item["shape_name"] == "levels/world_to_beamng/art/shapes/tunnels/tunnels.dae"
    assert item["collisionType"] == "Visible Mesh Final"
    assert config.TUNNEL_MATERIAL_NAME in stub.materials.added
    assert "asphalt_road_standard" in stub.materials.added
    assert stub.materials.added["asphalt_road_standard"]["groundType"] == "ASPHALT"


def test_export_tunnels_takes_the_concrete_texture_from_the_registry_check_and_never_falls_back(shapes_dir, monkeypatch):
    def aborted():
        raise registry.MissingTexturesError("Beton-Textur fehlt")

    monkeypatch.setattr(registry, "prepared_textures", aborted)
    stub = _stub()

    with pytest.raises(registry.MissingTexturesError):
        TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1")], "structure_road_polygons": [_road(1, "tunnel", {"tunnel": "yes"})]})
    assert not stub.materials.added and not stub.dae.calls


def test_nothing_is_exported_and_stale_files_are_removed_without_tunnels(shapes_dir, monkeypatch):
    stale = shapes_dir / "tunnels"
    stale.mkdir(parents=True)
    (stale / "tunnels.dae").write_text("alt", encoding="utf-8")
    (stale / "tunnels.cdae").write_text("alt", encoding="utf-8")
    stub = _stub()

    assert TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [], "structure_road_polygons": []}) == 0
    assert not (stale / "tunnels.dae").exists() and not (stale / "tunnels.cdae").exists()

    monkeypatch.setattr(config, "TUNNELS_ENABLED", False)
    assert TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1")], "structure_road_polygons": [_road(1, "tunnel", {"tunnel": "yes"})]}) == 0
    assert not stub.dae.calls


def test_build_tunnels_creates_tube_plus_portals_for_a_tunnel_and_one_mesh_per_gallery():
    heights = np.full((50, 50), 495.0)
    tunnel_road = _road(1, "tunnel", {"highway": "trunk", "tunnel": "yes"})
    gallery_road = _road(2, "gallery", {"highway": "primary", "tunnel": "avalanche_protector"})

    meshes = TerrainWorkflow._build_tunnels(SimpleNamespace(), [tunnel_road, gallery_road], heights, 0.0, 0.0)

    ids = [m["id"] for m in meshes]
    assert "tunnel_1" in ids and "tunnel_1_portal_start" in ids and "tunnel_1_portal_end" in ids and "gallery_2" in ids


def test_build_tunnels_skips_surface_and_bridge_roads():
    heights = np.full((10, 10), 495.0)
    road = _road(1, "bridge", {"bridge": "yes"})

    assert TerrainWorkflow._build_tunnels(SimpleNamespace(), [road], heights, 0.0, 0.0) == []
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `python -m pytest tests/workflow/test_terrain_workflow_tunnels.py -v`
Expected: FAIL mit `AttributeError: type object 'TerrainWorkflow' has no attribute '_build_tunnels'` (und `config` hat noch keine `TUNNEL_*`/`GALLERY_*`-Konstanten)

- [ ] **Step 3: Implementierung schreiben**

In `world_to_beamng/config.py`, nach dem in Task 6 ergänzten `BRIDGE_*`-Block einfügen:

```python
# Tunnel (OSM highway=* mit tunnel=yes/culvert/building_passage) und Galerien (tunnel=avalanche_protector):
# Röhre bzw. talseitig offene Galerie entlang des linear interpolierten Höhenprofils (siehe Design-Spec
# Abschnitt 5/6). Ersetzt für diese Straßen die normale Terrain-Einbettung und den DecalRoad-Export.
TUNNELS_ENABLED = True  # deckt auch Galerien (tunnel=avalanche_protector) ab
TUNNEL_WIDTH_MARGIN = 1.5  # zusätzliche Breite über die Fahrbahnbreite hinaus, in Metern
TUNNEL_ARC_SEGMENTS = 12  # Diskretisierung des 240°-Kreisbogens (Radius/Kronenhöhe ergeben sich aus der Breite)
TUNNEL_SEGMENT_STEP = 10.0  # Extrusions-Schrittweite entlang der Achse, in Metern (grob, da geradlinig)
TUNNEL_PORTAL_SLOPE_SAMPLE_DIST = 5.0  # Abtastradius der Hangneigung an den Portalen, in Metern
TUNNEL_PORTAL_FRAME_MARGIN = 0.6  # Rahmenbreite um die Portalöffnung, in Metern
GALLERY_HEIGHT = 5.0  # lichte Höhe der (rechteckigen, nicht kreisrunden) Galerie, in Metern
GALLERY_COLUMN_SPACING = 6.0  # Stützenabstand auf der offenen Talseite, in Metern
GALLERY_ROOF_THICKNESS = 0.35  # Dachdicke, in Metern
GALLERY_COLUMN_SIZE = 0.4  # Querschnitt der (quadratischen) Stützen, in Metern
TUNNEL_MATERIAL_NAME = "tunnel_concrete"  # Wand-/Decke-/Rahmen-/Dach-Material (Textur: CONCRETE_TEXTURE_NAME)
```

In `world_to_beamng/workflow/terrain_workflow.py`, direkt nach der in Task 6 ergänzten `export_bridges()`-Methode zwei neue Methoden einfügen:

```python
    def _build_tunnels(self, structure_road_polygons: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float) -> List[Dict]:
        """Tunnel- (Röhre+Portale) und Galerie-Meshes (Dach+Stützen) für alle Straßen mit structure_type in
        ("tunnel", "gallery") - siehe tunnels/tunnel_mesh.py und tunnels/gallery_mesh.py."""
        from ..terrain.road_embedding import sample_heightmap_bilinear
        from ..tunnels.gallery_mesh import build_galleries
        from ..tunnels.tunnel_mesh import build_tunnels

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        def _items(structure_type):
            return [
                {
                    "id": road["road_id"],
                    "coords": road["trimmed_centerline"],
                    "width": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"],
                    "floor_material": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {})).get("internal_name", "road_default"),
                }
                for road in structure_road_polygons
                if road.get("structure_type") == structure_type
            ]

        tunnel_meshes = build_tunnels(
            _items("tunnel"),
            ground_at,
            wall_material=config.TUNNEL_MATERIAL_NAME,
            frame_material=config.TUNNEL_MATERIAL_NAME,
            width_margin=config.TUNNEL_WIDTH_MARGIN,
            arc_segments=config.TUNNEL_ARC_SEGMENTS,
            segment_step=config.TUNNEL_SEGMENT_STEP,
            portal_slope_sample_dist=config.TUNNEL_PORTAL_SLOPE_SAMPLE_DIST,
            frame_margin=config.TUNNEL_PORTAL_FRAME_MARGIN,
        )
        gallery_meshes = build_galleries(
            _items("gallery"),
            ground_at,
            roof_material=config.TUNNEL_MATERIAL_NAME,
            height=config.GALLERY_HEIGHT,
            column_spacing=config.GALLERY_COLUMN_SPACING,
            roof_thickness=config.GALLERY_ROOF_THICKNESS,
            column_size=config.GALLERY_COLUMN_SIZE,
        )
        return tunnel_meshes + gallery_meshes

    def export_tunnels(self, mesh_data: Dict) -> int:
        """
        Exportiert Tunnel (Röhre + 2 Portal-Rahmen je Tunnel) und Galerien (Dach + Stützen) als EINE DAE mit
        EINEM TSStatic und registriert Fahrbahn- und Beton-Material. Ohne Tunnel/Galerien werden Reste eines
        früheren Exports entfernt.

        Returns:
            Anzahl exportierter Tunnel-/Portal-/Galerie-Meshes
        """
        tunnels_dir = config.BEAMNG_DIR_SHAPES / "tunnels"
        meshes = mesh_data.get("tunnel_meshes") or []
        if not config.TUNNELS_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (tunnels_dir / f"tunnels{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.TUNNEL_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        unique_floor_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") not in ("tunnel", "gallery"):
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            unique_floor_materials[props.get("internal_name", "road_default")] = props

        for mat_name, props in unique_floor_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=tunnels_dir / "tunnels.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "tunnels",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "tunnels" / "tunnels.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.info(f"  [OK] {len(meshes)} Tunnel-/Galerie-Mesh(e) exportiert (tunnels.dae)")
        return len(meshes)
```

In `process_tile()`, direkt nach dem in Task 6 ergänzten `bridge_meshes`-Block einfügen:

```python
        # Tunnel (Röhre + Portale) und Galerien (Dach + Stützen) auf der fertigen Heightmap - siehe tunnels/
        tunnel_meshes = []
        if config.TUNNELS_ENABLED:
            tunnel_meshes = self._build_tunnels(structure_road_polygons, heights, terrain_origin_x, terrain_origin_y)
```

Im Rückgabe-Dict von `process_tile()`, direkt nach `"bridge_meshes": bridge_meshes,` einfügen:

```python
            "tunnel_meshes": tunnel_meshes,  # Tunnel-/Galerie-Mesh-Dicts für export_tunnels()
```

In `export_tile()`, nach `self.export_bridges(mesh_data)` einfügen:

```python
        self.export_tunnels(mesh_data)
```

- [ ] **Step 4: Test laufen lassen, muss bestehen**

Run: `python -m pytest tests/workflow/test_terrain_workflow_tunnels.py -v`
Expected: PASS (5 Tests)

- [ ] **Step 5: Kompletten Testlauf gegen Regressionen prüfen**

Run: `python -m pytest tests/ -v`
Expected: PASS (alle Tests, inkl. aller in Task 1-11 geschriebenen und aller bereits vorher bestehenden)

- [ ] **Step 6: Commit**

```bash
git add world_to_beamng/config.py world_to_beamng/workflow/terrain_workflow.py tests/workflow/test_terrain_workflow_tunnels.py
git commit -m "feat: Export tunnels and galleries as their own DAE/TSStatic in the terrain workflow

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TiBuQMbCNsjt4Cmm8bpB1a"
```

---

## Nach der Implementierung

Nach Task 11 ist die Pipeline bereit, um gegen das laufende Gotthard-Exportgebiet getestet zu werden (siehe CLAUDE.md-Abschnitt zum Prüfen von `beamng.log` nach dem Laden der Map). Erwartete manuelle Prüfpunkte in-game, nicht Teil dieses Plans (keine automatisierten Tests möglich):

- Der 16,9 km lange Gotthard-Straßentunnel (Way 49124512) erscheint als befahrbare Röhre, nicht mehr als Phantom-Straße auf dem Bergrücken.
- Die Talbrücken der "Nuova strada del Passo del San Gottardo" (u.a. "Viadotto di Albinengo") zeigen ein Deck mit Pfeilern über dem Tal, das Terrain darunter bleibt unverändert.
- Die Lawinengalerie "Galleria artificiale Piano dei buoi" zeigt die talseitig offene Konstruktion.
- Portale wirken schräg in den Hang gesetzt, nicht wie ein gerader Schnitt.
- `beamng.log` enthält keine neuen `|E|`-Zeilen oder `Fatal-ISV`/`assert`-Meldungen beim Laden.
