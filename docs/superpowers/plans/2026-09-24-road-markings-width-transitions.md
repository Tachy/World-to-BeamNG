# Fahrbahnmarkierungen + weiche Breitenübergänge – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Asphaltierte Hauptstraßen bekommen weiße Randlinien (1- und 2-spurig) und bei zwei Fahrstreifen eine gestrichelte
Leitlinie. Ändert sich die Straßenbreite an einem Stoß zweier DecalRoads, wird sie über 10 m (5 m davor, 5 m danach)
mit einem Spline übergeblendet.

**Architecture:** Zwei neue, reine Geometrie-Module ohne BeamNG- und Config-Abhängigkeit:
`geometry/road_width_transitions.py` (Geradeaus-Stöße finden, Breiten per kubischem Hermite-Spline überblenden) und
`geometry/road_markings.py` (Layout-Regeln, Linienversatz je Knoten, Kehren-Bereinigung, Kreuzungs-Clipping).
`TerrainWorkflow.export_decal_roads()` wendet zuerst die Breitenübergänge auf die Fahrbahn-Knoten an und leitet daraus
die Markierungslinien ab. Die Linien sind eigene, schmale DecalRoads mit BeamNGs Vanilla-Linienmaterialien (Schema wie
`west_coast_usa`: `line_white`, `line_dashed_long`).

**Tech Stack:** Python 3.13, numpy, scipy (`cKDTree`), shapely 2.1 (`STRtree`, `buffer`, `difference`), pytest.

**Spec:** Die Anforderungen stammen aus dem Chat vom 2026-09-24 (unten unter „Global Constraints“ wörtlich
festgehalten). Hintergrund: `docs/OSM_ROAD_ANALYSIS.md` (Abschnitte 3.3 und 4).

## Global Constraints

- Randlinien: weiß, durchgezogen, bei ein- **und** zweispurigen Straßen, links und rechts.
- Zweispurige Straßen: zusätzlich eine **gestrichelte** Mittellinie (Leitlinie).
- Breitenübergänge: **innerhalb von 10 Metern, 5 Meter in die eine, 5 Meter in die andere Richtung**, mit Spline geglättet
  (kubischer Hermite-Spline `3t² − 2t³` über die ganzen 10 m; am Stoßpunkt die mittlere Breite).
- Nur DecalRoads (Oberflächenstraßen). Brücken, Tunnel und Galerien sind eigene Meshes und bleiben in diesem Plan
  **ohne** Markierung und ohne Breitenübergang.
- Chat-Antworten und Code-Kommentare auf Deutsch; Bezeichner, Strings im Code und Commit-Präfixe (`feat:`, `fix:`, …)
  auf Englisch (CLAUDE.md).
- Markierungs-DecalRoads bekommen `drivability = -1`. BeamNGs KI-Straßennetz (`lua/ge/map.lua`, Zeile 589:
  `if road and road.drivability > 0`) darf die Linien nicht als eigene Straßen übernehmen.
- Kein DecalRoad-Segment unter `config.DECAL_ROAD_MIN_NODE_SPACING` (0,5 m), sonst zeichnet BeamNG das ganze Decal nicht
  (siehe `tests/geometry/test_decal_road_nodes.py`).
- Tests: `.\.venv\Scripts\python.exe -m pytest …` aus dem Repo-Wurzelverzeichnis. Die ganze Suite muss nach jedem Task
  grün sein (Stand vor dem Plan: 1077 passed, 2 skipped).
- Commits enden mit den Attribution-Zeilen:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` und
  `Claude-Session: https://claude.ai/code/session_01Ma5GXmaJe8fEMhHha6f7tB`.
- **Vor Task 1:** Im Arbeitsbaum liegen noch unkommittete Änderungen (Maßnahme 1 der Straßenanalyse, Tunnelportal-Arbeit)
  in genau den Dateien, die dieser Plan anfasst (`config.py`, `osm_to_beamng.json`, `terrain_workflow.py`). Diese
  müssen zuerst committet sein. Sonst landen sie in den Commits dieses Plans (`git add -p` geht in dieser Umgebung
  nicht).

**Bewusst nicht im Umfang** (Folgeschritte laut `docs/OSM_ROAD_ANALYSIS.md`): Einbahn/`oneWay`, durchgezogene
Mittellinie bei mehrspurigen Gegenverkehrsstraßen (hier bekommt jede Fahrstreifengrenze eine gestrichelte Linie),
Sperrflächen und Pfeile, Markierungen auf Brückendecks, Breitenübergänge zwischen DecalRoad und Brücke.

## Review Focus

1. **Ringstraße** (Start- und Endpunkt derselben Straße fallen zusammen): Die Straße darf nicht mit sich selbst gepaart
   und übergeblendet werden. Test: `test_ring_road_is_not_paired_with_itself` (Task 2).
2. **Kurzes Zwischenstück** (< 10 m) zwischen zwei Breiten: Die Zone schrumpft symmetrisch, ohne Breitensprung am
   Zonenende. Test: `test_short_road_shrinks_zone_symmetrically` (Task 2).
3. **Doppelte aufeinanderfolgende Knoten**: Der Versatz darf keine NaN-Koordinaten erzeugen, sonst bekommt BeamNG ein
   kaputtes Decal. Test: `test_offset_polyline_survives_duplicate_nodes` (Task 3).
4. **Flach abzweigende Rampe** (20° zur Hauptfahrbahn): Sie darf der Hauptfahrbahn die Geradeaus-Paarung nicht
   wegnehmen, sonst würde die Hauptstraße auf Rampenbreite übergeblendet. Test:
   `test_link_branching_off_at_shallow_angle_loses_against_straight_main_road` (Task 2).
5. **Unsaubere `lanes`-Werte** (`"2;3"`, `""`, `"0"`): Die Regel muss auf die Breite zurückfallen und darf nicht
   abstürzen. Test: `test_parse_lanes` (Task 3).

Zusätzlich im Spiel zu prüfen (kein Unit-Test möglich, Tasks 5 und 7): Zeichenreihenfolge (`renderPriority`),
Strichlänge und Linien in den Kehren der Nuova strada del San Gottardo.

---

## Dateistruktur

| Datei | Verantwortung |
|---|---|
| `world_to_beamng/geometry/road_width_transitions.py` (neu) | Geradeaus-Stöße finden (`find_continuations`), Partner-Map, Breiten-Spline in der Übergangszone (`apply_width_transitions`) |
| `world_to_beamng/geometry/road_markings.py` (neu) | Markierungsregeln (`marking_layout`), Linienversatz (`offset_polyline`), Kehren (`forward_indices`), Linien je Straße (`build_marking_lines`), Kreuzungs-Clipping (`clip_line`, `road_surface_polygon`, `junction_obstacles`) |
| `world_to_beamng/workflow/terrain_workflow.py` | `export_decal_roads()`: Übergänge anwenden, Markierungs-DecalRoads schreiben; Modul-Helfer `_road_marking_lines()` |
| `world_to_beamng/osm/osm_mapper.py` | `road_markings` laden, `generate_marking_material_entry()` |
| `world_to_beamng/config.py` | Konstanten für Übergänge und Markierungen |
| `data/osm_to_beamng.json` | Linienmaterialien unter `road_markings` |
| Tests | `tests/geometry/test_road_width_transitions.py`, `tests/geometry/test_road_markings.py`, `tests/test_osm_mapper_road_markings.py`, `tests/workflow/test_decal_road_markings_export.py` |

---

### Task 1: Konfiguration und Linienmaterialien

**Files:**
- Modify: `world_to_beamng/config.py` (Block direkt nach `DECAL_ROAD_MIN_NODE_SPACING = 0.5`, ca. Zeile 197)
- Modify: `data/osm_to_beamng.json` (neuer Top-Level-Key `road_markings` direkt nach `surface_overrides`)
- Modify: `world_to_beamng/osm/osm_mapper.py` (`__init__`, neue Methode nach `generate_materials_json_entry`)
- Test: `tests/test_osm_mapper_road_markings.py` (neu)

**Interfaces:**
- Produces: `config.ROAD_WIDTH_TRANSITION_LENGTH`, `ROAD_WIDTH_TRANSITION_STEP`, `ROAD_WIDTH_TRANSITION_MIN_DELTA`,
  `ROAD_CONTINUATION_ENDPOINT_TOL`, `ROAD_CONTINUATION_MAX_ANGLE_DEG`, `ROAD_MARKINGS_ENABLED`, `ROAD_MARKING_HIGHWAYS`,
  `ROAD_MARKING_SURFACE`, `ROAD_MARKING_MIN_TWO_LANE_WIDTH`, `ROAD_MARKING_LINE_WIDTH`, `ROAD_MARKING_EDGE_INSET`,
  `ROAD_MARKING_EDGE_MATERIAL`, `ROAD_MARKING_DIVIDER_MATERIAL`, `ROAD_MARKING_RENDER_PRIORITY`,
  `ROAD_MARKING_JUNCTION_CLEARANCE`, `ROAD_MARKING_MIN_PIECE_LENGTH`, `ROAD_MARKING_NO_GAP_HIGHWAYS`.
- Produces: `OSMMapper.road_markings: Dict[str, Dict]` (Name → `{"annotation", "textureLength", "textures"}`),
  `OSMMapper.generate_marking_material_entry(mat_name: str, props: Dict) -> Dict`.

- [ ] **Step 1: Failing Test schreiben** – `tests/test_osm_mapper_road_markings.py`:

```python
"""Tests für die Markierungs-Linienmaterialien (data/osm_to_beamng.json -> road_markings) und ihren
materials.json-Eintrag. Vorbild: BeamNGs eigene line_white / line_dashed_long
(west_coast_usa/art/road/main.materials.json)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng import config
from world_to_beamng.osm.osm_mapper import OSMMapper

CONFIG_PATH = Path(__file__).parent.parent / "data" / "osm_to_beamng.json"
LINES_PREFIX = "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/"


@pytest.fixture(scope="module")
def mapper():
    return OSMMapper(config_path=str(CONFIG_PATH))


def test_marking_materials_exist_for_configured_names(mapper):
    assert config.ROAD_MARKING_EDGE_MATERIAL in mapper.road_markings
    assert config.ROAD_MARKING_DIVIDER_MATERIAL in mapper.road_markings


def test_marking_materials_use_vanilla_line_textures(mapper):
    edge = mapper.road_markings[config.ROAD_MARKING_EDGE_MATERIAL]
    divider = mapper.road_markings[config.ROAD_MARKING_DIVIDER_MATERIAL]

    assert edge["textures"]["opacityMap"] == LINES_PREFIX + "line_white/t_line_white_o.data.dds"
    assert divider["textures"]["opacityMap"] == LINES_PREFIX + "line_dashed_long/t_line_dashed_long_o.data.dds"
    for marking in (edge, divider):
        assert set(marking["textures"]) == {"baseColorMap", "normalMap", "opacityMap"}
        assert marking["textureLength"] > 0
    assert edge["annotation"] == "SOLID_LINE"
    assert divider["annotation"] == "DASHED_LINE"


def test_marking_material_entry_follows_vanilla_line_schema(mapper):
    props = mapper.road_markings[config.ROAD_MARKING_DIVIDER_MATERIAL]
    entry = mapper.generate_marking_material_entry(config.ROAD_MARKING_DIVIDER_MATERIAL, props)

    assert entry["name"] == entry["mapTo"] == config.ROAD_MARKING_DIVIDER_MATERIAL
    assert entry["class"] == "Material"
    assert entry["annotation"] == "DASHED_LINE"
    assert entry["translucent"] is True
    assert entry["translucentZWrite"] is True
    assert entry["castShadows"] is False
    assert entry["materialTag0"] == "RoadAndPath"
    assert entry["Stages"][0]["opacityMap"].endswith("t_line_dashed_long_o.data.dds")
    assert "__name" not in entry


def test_marking_material_entries_get_unique_persistent_ids(mapper):
    props = mapper.road_markings[config.ROAD_MARKING_EDGE_MATERIAL]
    first = mapper.generate_marking_material_entry("a", props)
    second = mapper.generate_marking_material_entry("b", props)

    assert first["persistentId"] != second["persistentId"]


def test_marking_render_priority_is_above_all_road_surfaces(mapper):
    assert all(config.ROAD_MARKING_RENDER_PRIORITY > s["priority"] for s in mapper.surface_types.values())
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_osm_mapper_road_markings.py -v`
Expected: FAIL mit `AttributeError: module 'world_to_beamng.config' has no attribute 'ROAD_MARKING_EDGE_MATERIAL'`

- [ ] **Step 3: Config-Block einfügen** – in `world_to_beamng/config.py` direkt nach der Zeile
`DECAL_ROAD_MIN_NODE_SPACING = 0.5`:

```python

# === BREITENÜBERGÄNGE / FAHRBAHNMARKIERUNG ===
# Siehe docs/superpowers/plans/2026-09-24-road-markings-width-transitions.md.
# Breitenübergang an Geradeaus-Stößen zweier DecalRoads (geometry/road_width_transitions.py): über 10 m, je 5 m vor und
# nach dem Stoßpunkt, kubischer Hermite-Spline; in der Zone ein Knoten je ROAD_WIDTH_TRANSITION_STEP Meter.
ROAD_WIDTH_TRANSITION_LENGTH = 10.0
ROAD_WIDTH_TRANSITION_STEP = 1.0
ROAD_WIDTH_TRANSITION_MIN_DELTA = 0.05  # kleinere Breitenunterschiede bleiben unverändert, in Metern
ROAD_CONTINUATION_ENDPOINT_TOL = 0.5  # so nah müssen zwei Straßenenden beieinander liegen, in Metern
ROAD_CONTINUATION_MAX_ANGLE_DEG = 30.0  # größter Knick, der noch als "geradeaus weiter" gilt

# Markierungen (geometry/road_markings.py): eigene schmale DecalRoads über der Fahrbahn wie in BeamNGs Vanilla-Levels.
ROAD_MARKINGS_ENABLED = True
ROAD_MARKING_HIGHWAYS = frozenset(
    {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    }
)
ROAD_MARKING_SURFACE = "asphalt_road_standard"  # nur Asphalt - kein Pflaster, Kies, Erdweg
ROAD_MARKING_MIN_TWO_LANE_WIDTH = 5.5  # ohne lanes-Tag: schmalere Straßen sind einspurig (nur Randlinien)
ROAD_MARKING_LINE_WIDTH = 0.15  # Linienbreite in Metern (Vanilla: 0,15-0,2 m)
ROAD_MARKING_EDGE_INSET = 0.25  # Abstand der Randlinien-Mitte vom Fahrbahnrand, in Metern
ROAD_MARKING_EDGE_MATERIAL = "line_edge_white"  # Einträge in data/osm_to_beamng.json -> road_markings
ROAD_MARKING_DIVIDER_MATERIAL = "line_divider_dashed"
# Über allen Fahrbahn-Prioritäten (surface_types[*].priority <= 8). Annahme wie im bestehenden Export: höherer Wert =
# weiter oben (Italy-Level: Linien 23-31 über Straßen 10) - im Spiel bestätigen (Plan Task 5).
ROAD_MARKING_RENDER_PRIORITY = 20
ROAD_MARKING_JUNCTION_CLEARANCE = 0.5  # die Lücke in der Randlinie reicht so weit über die einmündende Fahrbahn hinaus
ROAD_MARKING_MIN_PIECE_LENGTH = 2.0  # kürzere Linienreste nach dem Kreuzungsschnitt entfallen, in Metern
# Wege, deren Einmündung KEINE Lücke in die Randlinie schneidet (Feldweg, Fußweg ...)
ROAD_MARKING_NO_GAP_HIGHWAYS = frozenset({"track", "path", "footway", "cycleway", "bridleway", "steps"})
```

- [ ] **Step 4: Linienmaterialien in `data/osm_to_beamng.json`** – den Block `"surface_overrides": { … }` suchen. Sein
letzter Eintrag ist `"paving_stones": { "internal_name": "cobblestone_road" }`. Direkt dahinter, nach der schließenden
`},` von `surface_overrides` und vor `"forest_type_templates"`, einfügen (4 Leerzeichen Einrückung wie der Rest der
Datei):

```json
    "road_markings": {
        "line_edge_white": {
            "annotation": "SOLID_LINE",
            "textureLength": 10.0,
            "textures": {
                "baseColorMap": "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/line_white/t_line_white_b.color.dds",
                "normalMap": "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/line_white/t_line_white_nm.normal.dds",
                "opacityMap": "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/line_white/t_line_white_o.data.dds"
            }
        },
        "line_divider_dashed": {
            "annotation": "DASHED_LINE",
            "textureLength": 14.0,
            "textures": {
                "baseColorMap": "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/line_dashed_long/t_line_dashed_long_b.color.dds",
                "normalMap": "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/line_dashed_long/t_line_dashed_long_nm.normal.dds",
                "opacityMap": "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/line_dashed_long/t_line_dashed_long_o.data.dds"
            }
        }
    },
```

(`t_line_dashed_long_o` hat einen Strich pro Kachel mit 42 % Strich und 58 % Lücke. Bei `textureLength` 14 m ergibt
das rund 5,9 m Strich und 8,1 m Lücke.) Danach prüfen, ob die Datei gültiges JSON ist:
`.\.venv\Scripts\python.exe -c "import json; json.load(open('data/osm_to_beamng.json', encoding='utf-8'))"`

- [ ] **Step 5: Mapper erweitern** – in `world_to_beamng/osm/osm_mapper.py`, `__init__`, nach
`self.surface_types = self.config.get("surface_types", {})`:

```python
        self.road_markings = self.config.get("road_markings", {})
```

und direkt nach der Methode `generate_materials_json_entry()` die neue Methode:

```python
    def generate_marking_material_entry(self, mat_name, props):
        """
        materials.json-Eintrag für eine Markierungslinie (Rand-/Leitlinie). Schema wie BeamNGs eigene `line_white` /
        `line_dashed_long` (west_coast_usa/art/road/main.materials.json): translucent mit opacityMap, ohne Schatten,
        annotation SOLID_LINE bzw. DASHED_LINE. Anders als generate_materials_json_entry() ohne "__name".
        """
        tex = props.get("textures", {})
        stage = {key: tex[key] for key in ("baseColorMap", "normalMap", "opacityMap") if tex.get(key)}
        return {
            "name": mat_name,
            "mapTo": mat_name,
            "class": "Material",
            "version": 1.5,
            "Stages": [stage],
            "annotation": props.get("annotation", "SOLID_LINE"),
            "alphaRef": 255,
            "castShadows": False,
            "materialTag0": "RoadAndPath",
            "materialTag1": "beamng",
            "translucent": True,
            "translucentZWrite": True,
            "persistentId": str(uuid.uuid4()),
        }
```

- [ ] **Step 6: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_osm_mapper_road_markings.py tests/test_data_ground_models.py tests/test_osm_mapper_road_surfaces.py -v`
Expected: alle PASS

- [ ] **Step 7: Texturen ins Level kopieren**

Run: `.\.venv\Scripts\python.exe tools\vendor_shared_textures.py`
Expected: `[DONE] 40 Texturen kopiert` (vorher 34, dazu 6 Linientexturen) und `[✓] ERFOLGREICH ABGESCHLOSSEN` ohne
`NICHT in den Content-ZIPs gefunden`.

- [ ] **Step 8: Ganze Suite, dann Commit**

Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün.

```bash
git add world_to_beamng/config.py data/osm_to_beamng.json world_to_beamng/osm/osm_mapper.py tests/test_osm_mapper_road_markings.py
git commit -m "feat: Add road marking line materials and config for markings and width transitions"
```

---

### Task 2: Weiche Breitenübergänge (reines Geometrie-Modul)

**Files:**
- Create: `world_to_beamng/geometry/road_width_transitions.py`
- Test: `tests/geometry/test_road_width_transitions.py`

**Interfaces:**
- Produces: `smoothstep(t: float) -> float`
- Produces: `find_continuations(roads, endpoint_tol: float, max_angle_deg: float) -> List[Tuple[Endpoint, Endpoint]]`
  mit `Endpoint = Tuple[int, str]` (Straßen-Index, `"start"`/`"end"`); `roads`: Liste von Knotenlisten
  `[[x, y, z, width], ...]`
- Produces: `continuation_partners(pairs) -> Dict[int, Set[int]]`
- Produces: `apply_width_transitions(roads, transition_length, step, endpoint_tol, max_angle_deg, min_delta, min_spacing) -> List[List[List[float]]]`
  (neue Listen, Eingabe bleibt unverändert)

- [ ] **Step 1: Failing Tests schreiben** – `tests/geometry/test_road_width_transitions.py`:

```python
"""Tests für die weichen Breitenübergänge zwischen DecalRoads (geometry/road_width_transitions.py).

Anforderung: Breitenwechsel an einem Stoß werden über 10 m (5 m davor, 5 m danach) mit einem Spline geglättet.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.road_width_transitions import (
    apply_width_transitions,
    continuation_partners,
    find_continuations,
    smoothstep,
)

KW = dict(transition_length=10.0, step=1.0, endpoint_tol=0.5, max_angle_deg=30.0, min_delta=0.05, min_spacing=0.5)


def _road(points, width):
    return [[float(x), float(y), 100.0, float(width)] for x, y in points]


def _width_at(nodes, x):
    for n in nodes:
        if abs(n[0] - x) < 1e-6:
            return n[3]
    raise AssertionError(f"kein Knoten bei x={x}")


def test_smoothstep_is_cubic_hermite():
    assert smoothstep(0.0) == 0.0 and smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert smoothstep(0.25) == pytest.approx(0.15625)
    assert smoothstep(-1.0) == 0.0 and smoothstep(2.0) == 1.0


def test_straight_continuation_is_paired():
    a = _road([(0, 0), (10, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (30, 0), (40, 0)], 9.75)
    assert find_continuations([a, b], 0.5, 30.0) == [((0, "end"), (1, "start"))]


def test_right_angle_corner_is_not_paired():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (20, 20)], 9.75)
    assert find_continuations([a, b], 0.5, 30.0) == []


def test_link_branching_off_at_shallow_angle_loses_against_straight_main_road():
    a = _road([(-20, 0), (0, 0)], 6.5)
    b = _road([(0, 0), (20, 0)], 13.0)
    angle = np.radians(20.0)
    c = _road([(0, 0), (20 * np.cos(angle), 20 * np.sin(angle))], 4.0)
    pairs = find_continuations([a, b, c], 0.5, 30.0)
    assert pairs == [((0, "end"), (1, "start"))]
    assert continuation_partners(pairs) == {0: {1}, 1: {0}}


def test_ring_road_is_not_paired_with_itself():
    ring = _road([(0, 0), (10, 0), (10, 10), (0.2, 0.0)], 6.5)
    assert find_continuations([ring], 0.5, 180.0) == []


def test_width_blends_over_five_metres_on_each_side():
    a = _road([(0, 0), (10, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (30, 0), (40, 0)], 9.75)
    new_a, new_b = apply_width_transitions([a, b], **KW)
    assert _width_at(new_a, 20.0) == pytest.approx(8.125)
    assert _width_at(new_b, 20.0) == pytest.approx(8.125)
    assert _width_at(new_a, 15.0) == pytest.approx(6.5)
    assert _width_at(new_b, 25.0) == pytest.approx(9.75)
    assert _width_at(new_a, 17.0) == pytest.approx(6.5 + 3.25 * smoothstep(0.2))
    assert _width_at(new_b, 23.0) == pytest.approx(6.5 + 3.25 * smoothstep(0.8))
    assert _width_at(new_a, 10.0) == pytest.approx(6.5)
    assert _width_at(new_b, 40.0) == pytest.approx(9.75)


def test_blend_is_monotonic_and_inserts_nodes_every_metre():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    new_a, new_b = apply_width_transitions([a, b], **KW)
    xs = [n[0] for n in new_a] + [n[0] for n in new_b[1:]]
    assert {15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 22.0, 23.0, 24.0, 25.0} <= set(xs)
    widths = [n[3] for n in new_a] + [n[3] for n in new_b[1:]]
    assert all(w2 >= w1 - 1e-12 for w1, w2 in zip(widths, widths[1:]))


def test_inserted_nodes_keep_min_spacing():
    a = _road([(0, 0), (16.8, 0), (19.7, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    new_a, _ = apply_width_transitions([a, b], **KW)
    gaps = np.diff([n[0] for n in new_a])
    # die vorhandene 0,3-m-Lücke (19.7 -> 20) bleibt, eingefügt wird nur mit >= 0,5 m Abstand
    assert sorted(gaps)[0] == pytest.approx(0.3)
    assert sum(g < 0.5 for g in gaps) == 1


def test_equal_widths_are_left_untouched():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 6.5)
    assert apply_width_transitions([a, b], **KW) == [a, b]


def test_short_road_shrinks_zone_symmetrically():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (24, 0)], 9.75)
    new_a, new_b = apply_width_transitions([a, b], **KW)
    assert _width_at(new_a, 18.0) == pytest.approx(6.5)
    assert _width_at(new_b, 22.0) == pytest.approx(9.75)
    assert _width_at(new_b, 24.0) == pytest.approx(9.75)
    assert _width_at(new_a, 20.0) == pytest.approx(8.125)


def test_both_ends_of_a_road_can_blend():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    c = _road([(40, 0), (60, 0)], 6.5)
    _, new_b, _ = apply_width_transitions([a, b, c], **KW)
    assert _width_at(new_b, 20.0) == pytest.approx(8.125)
    assert _width_at(new_b, 25.0) == pytest.approx(9.75)
    assert _width_at(new_b, 35.0) == pytest.approx(9.75)
    assert _width_at(new_b, 40.0) == pytest.approx(8.125)


def test_z_is_interpolated_for_inserted_nodes():
    a = [[0.0, 0.0, 100.0, 6.5], [20.0, 0.0, 120.0, 6.5]]
    b = _road([(20, 0), (40, 0)], 9.75)
    new_a, _ = apply_width_transitions([a, b], **KW)
    node = next(n for n in new_a if abs(n[0] - 17.0) < 1e-6)
    assert node[2] == pytest.approx(117.0)


def test_input_lists_are_not_modified():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    before = [list(map(list, a)), list(map(list, b))]
    apply_width_transitions([a, b], **KW)
    assert [a, b] == before
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_width_transitions.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.geometry.road_width_transitions'`

- [ ] **Step 3: Modul implementieren** – `world_to_beamng/geometry/road_width_transitions.py`:

```python
"""
Weiche Breitenübergänge zwischen aneinanderstoßenden DecalRoads.

Ändert sich an einem Stoßpunkt zweier Straßen, die geradeaus ineinander übergehen, die Breite (z.B. lanes=2 ->
lanes=3), springt sie nicht mehr hart um: über ROAD_WIDTH_TRANSITION_LENGTH (je die Hälfte vor und nach dem
Stoßpunkt) wird sie mit einem kubischen Hermite-Spline (smoothstep, Steigung 0 an beiden Zonenenden) übergeblendet.
Die Breite steckt im 4. Eintrag jedes DecalRoad-Knotens [x, y, z, width]; BeamNG interpoliert zwischen den Knoten,
deshalb bekommt die Übergangszone zusätzliche Knoten im Abstand `step`.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

Endpoint = Tuple[int, str]  # (Index der Straße, "start" | "end")


def smoothstep(t: float) -> float:
    """Kubischer Hermite-Spline 3t^2 - 2t^3 auf [0, 1] (außerhalb geklemmt)."""
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _arc_lengths(nodes: Sequence[Sequence[float]]) -> np.ndarray:
    xy = np.asarray(nodes, dtype=float)[:, :2]
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def _outward_direction(nodes: Sequence[Sequence[float]], end: str) -> Optional[np.ndarray]:
    """Einheitsvektor vom Stoßpunkt in die Straße hinein (None bei einer Straße ohne Ausdehnung)."""
    xy = np.asarray(nodes, dtype=float)[:, :2]
    if end == "end":
        xy = xy[::-1]
    for other in xy[1:]:
        vector = other - xy[0]
        length = float(np.linalg.norm(vector))
        if length > 1e-9:
            return vector / length
    return None


def find_continuations(
    roads: Sequence[Sequence[Sequence[float]]], endpoint_tol: float, max_angle_deg: float
) -> List[Tuple[Endpoint, Endpoint]]:
    """
    Paare von Straßenenden, die am selben Punkt liegen (Abstand <= endpoint_tol) und geradeaus ineinander übergehen
    (Knick <= max_angle_deg). An einer Einmündung gewinnt das gestreckteste Paar; eine schräg abzweigende Rampe
    bleibt ungepaart. Eine Straße wird nie mit sich selbst gepaart (Ring).
    """
    endpoints = []  # ((road_idx, end), (x, y), outward_direction)
    for road_idx, nodes in enumerate(roads):
        if len(nodes) < 2:
            continue
        for end in ("start", "end"):
            direction = _outward_direction(nodes, end)
            if direction is None:
                continue
            point = nodes[0] if end == "start" else nodes[-1]
            endpoints.append(((road_idx, end), (float(point[0]), float(point[1])), direction))
    if len(endpoints) < 2:
        return []

    parent = list(range(len(endpoints)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(np.array([e[1] for e in endpoints])).query_pairs(endpoint_tol):
        parent[root(a)] = root(b)
    clusters: Dict[int, List[int]] = {}
    for i in range(len(endpoints)):
        clusters.setdefault(root(i), []).append(i)

    min_opposition = float(np.cos(np.radians(max_angle_deg)))
    pairs = []
    for members in clusters.values():
        candidates = []
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = endpoints[members[x]], endpoints[members[y]]
                if a[0][0] == b[0][0]:
                    continue
                opposition = -float(np.dot(a[2], b[2]))  # 1 = exakt geradeaus
                if opposition >= min_opposition:
                    candidates.append((opposition, members[x], members[y]))
        used = set()
        for _, a, b in sorted(candidates, key=lambda c: -c[0]):
            if a in used or b in used:
                continue
            used.update((a, b))
            pairs.append((endpoints[a][0], endpoints[b][0]))
    return pairs


def continuation_partners(pairs: Sequence[Tuple[Endpoint, Endpoint]]) -> Dict[int, Set[int]]:
    """Straßen-Index -> Indizes der Straßen, in die sie geradeaus übergeht."""
    partners: Dict[int, Set[int]] = {}
    for (a, _), (b, _) in pairs:
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def _insert_nodes(nodes: List[List[float]], distances: Sequence[float], min_spacing: float) -> List[List[float]]:
    """Zusätzliche Knoten bei den Bogenlängen `distances` (ab nodes[0]), linear interpoliert (x, y, z, Breite).
    Wo schon ein Knoten näher als min_spacing liegt, wird nichts eingefügt (BeamNG verwirft zu kurze Segmente)."""
    arr = np.asarray(nodes, dtype=float)
    cum = _arc_lengths(nodes)
    entries = [(float(cum[i]), [float(v) for v in arr[i]]) for i in range(len(arr))]
    taken = list(cum)
    for d in distances:
        if d <= 0.0 or d >= cum[-1] or min(abs(t - d) for t in taken) < min_spacing:
            continue
        seg = int(np.searchsorted(cum, d)) - 1
        f = (d - cum[seg]) / (cum[seg + 1] - cum[seg])
        entries.append((float(d), [float(v) for v in arr[seg] + f * (arr[seg + 1] - arr[seg])]))
        taken.append(d)
    entries.sort(key=lambda e: e[0])
    return [node for _, node in entries]


def _blend_end(nodes, end, own_width, other_width, half, step, min_spacing):
    """Breiten in der Übergangszone am Ende `end`: Spline von der mittleren Breite (Stoßpunkt) zur eigenen (half)."""
    work = [list(n) for n in (nodes if end == "start" else nodes[::-1])]
    distances = [float(d) for d in np.arange(step, half, step)] + [half]
    work = _insert_nodes(work, distances, min_spacing)
    for node, s in zip(work, _arc_lengths(work)):
        if s <= half + 1e-9:
            node[3] = own_width + (other_width - own_width) * smoothstep((half - s) / (2.0 * half))
    return work if end == "start" else work[::-1]


def apply_width_transitions(
    roads: Sequence[Sequence[Sequence[float]]],
    transition_length: float,
    step: float,
    endpoint_tol: float,
    max_angle_deg: float,
    min_delta: float,
    min_spacing: float,
) -> List[List[List[float]]]:
    """
    Neue Knotenlisten ([x, y, z, width] je Knoten) mit weichen Breitenübergängen an allen Geradeaus-Stößen, deren
    Breiten sich um mindestens min_delta unterscheiden. Am Stoßpunkt liegt die mittlere Breite, transition_length / 2
    davor und dahinter wieder die eigene. Ist eine der beiden Straßen kürzer als transition_length, schrumpft die Zone
    auf beiden Seiten symmetrisch auf die halbe Länge der kürzeren Straße.
    """
    result = [[[float(v) for v in n] for n in nodes] for nodes in roads]
    for (ia, ea), (ib, eb) in find_continuations(roads, endpoint_tol, max_angle_deg):
        wa = float(roads[ia][0 if ea == "start" else -1][3])
        wb = float(roads[ib][0 if eb == "start" else -1][3])
        if abs(wa - wb) < min_delta:
            continue
        half = min(transition_length / 2.0, _arc_lengths(roads[ia])[-1] / 2.0, _arc_lengths(roads[ib])[-1] / 2.0)
        if half <= 0.0:
            continue
        result[ia] = _blend_end(result[ia], ea, wa, wb, half, step, min_spacing)
        result[ib] = _blend_end(result[ib], eb, wb, wa, half, step, min_spacing)
    return result
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_width_transitions.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/geometry/road_width_transitions.py tests/geometry/test_road_width_transitions.py
git commit -m "feat: Add spline width transitions between straight-continuing road segments"
```

---

### Task 3: Markierungsgeometrie (Regeln, Versatz, Kehren)

**Files:**
- Create: `world_to_beamng/geometry/road_markings.py`
- Test: `tests/geometry/test_road_markings.py`

**Interfaces:**
- Produces: Konstanten `EDGE = "edge"`, `DIVIDER = "divider"`; `@dataclass(frozen=True) class MarkingLayout: lanes: int`
- Produces: `parse_lanes(value) -> Optional[int]`
- Produces: `marking_layout(tags: dict, width: float, internal_name: str, marked_highways, marked_surface: str, min_two_lane_width: float) -> Optional[MarkingLayout]`
- Produces: `line_offsets(widths: np.ndarray, lanes: int, edge_inset: float) -> List[Tuple[str, np.ndarray]]`
  (Reihenfolge: linke Randlinie, rechte Randlinie, dann Leitlinien)
- Produces: `offset_polyline(xy: np.ndarray, offsets: np.ndarray) -> np.ndarray`
- Produces: `forward_indices(offset_xy: np.ndarray, center_xy: np.ndarray) -> np.ndarray`
- Produces: `build_marking_lines(nodes, layout: MarkingLayout, edge_inset: float) -> List[Tuple[str, np.ndarray]]`
  (Linien als `(N, 3)`-Arrays x, y, z; Index in der Liste = `line_idx` im Item-Namen, Task 4)

- [ ] **Step 1: Failing Tests schreiben** – `tests/geometry/test_road_markings.py`:

```python
"""Tests für die Fahrbahnmarkierungs-Geometrie (geometry/road_markings.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.road_markings import (
    DIVIDER,
    EDGE,
    MarkingLayout,
    build_marking_lines,
    forward_indices,
    line_offsets,
    marking_layout,
    offset_polyline,
    parse_lanes,
)

MARKED = {"primary", "secondary", "primary_link"}


def _layout(tags, width=6.5, surface="asphalt_road_standard"):
    return marking_layout(tags, width, surface, MARKED, "asphalt_road_standard", 5.5)


@pytest.mark.parametrize(
    "value, expected", [("2", 2), (" 3 ", 3), (1, 1), ("2;3", None), ("", None), ("0", None), (None, None)]
)
def test_parse_lanes(value, expected):
    assert parse_lanes(value) == expected


def test_layout_uses_lanes_tag():
    assert _layout({"highway": "primary", "lanes": "3"}) == MarkingLayout(lanes=3)


def test_layout_without_lanes_two_lanes_for_wide_road_one_for_narrow_or_link():
    assert _layout({"highway": "secondary"}, width=7.0) == MarkingLayout(lanes=2)
    assert _layout({"highway": "secondary"}, width=5.0) == MarkingLayout(lanes=1)
    assert _layout({"highway": "primary_link"}, width=7.0) == MarkingLayout(lanes=1)


def test_layout_none_for_unmarked_types_surfaces_and_lane_markings_no():
    assert _layout({"highway": "track"}) is None
    assert _layout({"highway": "secondary", "lane_markings": "no"}) is None
    assert _layout({"highway": "secondary"}, surface="cobblestone_road") is None


def test_line_offsets_single_lane_has_two_edges_no_divider():
    lines = line_offsets(np.array([4.0, 4.0]), 1, 0.25)
    assert [k for k, _ in lines] == [EDGE, EDGE]
    assert lines[0][1] == pytest.approx([1.75, 1.75])
    assert lines[1][1] == pytest.approx([-1.75, -1.75])


def test_line_offsets_two_lanes_have_centre_divider_following_width():
    lines = line_offsets(np.array([6.5, 9.75]), 2, 0.25)
    assert [k for k, _ in lines] == [EDGE, EDGE, DIVIDER]
    assert lines[0][1] == pytest.approx([3.0, 4.625])
    assert lines[2][1] == pytest.approx([0.0, 0.0])


def test_offset_polyline_straight_and_left_is_positive():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    assert offset_polyline(xy, np.array([2.0, 2.0, 2.0])) == pytest.approx(np.array([[0, 2], [10, 2], [20, 2]]))


def test_offset_polyline_survives_duplicate_nodes():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    result = offset_polyline(xy, np.full(4, -1.0))
    assert np.isfinite(result).all()
    assert result[:, 1] == pytest.approx([-1.0] * 4)


def _hairpin():
    # 10 m geradeaus nach Osten, Rechtskehre mit 2 m Radius, 10 m zurück nach Westen
    straight_in = [(x, 2.0) for x in np.arange(-10.0, 0.0, 1.0)]
    arc = [(2.0 * np.cos(a), 2.0 * np.sin(a)) for a in np.linspace(np.pi / 2, -np.pi / 2, 13)]
    straight_out = [(x, -2.0) for x in np.arange(-1.0, -11.0, -1.0)]
    return np.array(straight_in + arc + straight_out)


def _backward_steps(line, center, indices):
    """Anzahl Liniensegmente, die entgegen der Fahrtrichtung (Tangente der Mittellinie) laufen."""
    count = 0
    for a, b in zip(indices, indices[1:]):
        tangent = center[min(b + 1, len(center) - 1)] - center[max(b - 1, 0)]
        count += float(np.dot(line[b] - line[a], tangent)) <= 0.0
    return count


def test_forward_indices_remove_backward_running_inner_line_in_tight_hairpin():
    center = _hairpin()
    inner = offset_polyline(center, np.full(len(center), -3.0))  # rechts = innen, Versatz > Radius
    everything = np.arange(len(center))
    assert _backward_steps(inner, center, everything) > 0  # Ausgangslage: läuft in der Kehre rückwärts
    kept = forward_indices(inner, center)
    assert _backward_steps(inner, center, kept) == 0
    assert LineString(inner[kept]).is_simple
    assert kept[0] == 0 and kept[-1] == len(center) - 1


def test_forward_indices_keep_outer_line_of_hairpin_complete():
    center = _hairpin()
    outer = offset_polyline(center, np.full(len(center), 3.0))
    assert len(forward_indices(outer, center)) == len(center)


def test_build_marking_lines_two_lane_road():
    nodes = [[0.0, 0.0, 100.0, 6.5], [10.0, 0.0, 101.0, 6.5], [20.0, 0.0, 102.0, 6.5]]
    lines = build_marking_lines(nodes, MarkingLayout(lanes=2), 0.25)
    assert [k for k, _ in lines] == [EDGE, EDGE, DIVIDER]
    left = lines[0][1]
    assert left[:, 1] == pytest.approx([3.0, 3.0, 3.0])
    assert left[:, 2] == pytest.approx([100.0, 101.0, 102.0])
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_markings.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.geometry.road_markings'`

- [ ] **Step 3: Modul implementieren** – `world_to_beamng/geometry/road_markings.py`:

```python
"""
Fahrbahnmarkierungen als eigene, schmale DecalRoads über der Fahrbahn - so wie BeamNGs eigene Levels es machen
(west_coast_usa: ~3100 `line_white`- und ~200 `line_dashed_short`-DecalRoads mit 0,15-0,2 m Breite). Weiße
Randlinien links und rechts, gestrichelte Leitlinien an den Fahrstreifengrenzen. Die Linien folgen der Knotenbreite
der Fahrbahn (also auch den weichen Breitenübergängen aus road_width_transitions.py). Hintergrund und Regeln siehe
docs/OSM_ROAD_ANALYSIS.md und docs/superpowers/plans/2026-09-24-road-markings-width-transitions.md.
"""

from dataclasses import dataclass
from typing import Collection, List, Optional, Sequence, Tuple

import numpy as np

EDGE = "edge"
DIVIDER = "divider"
MAX_MITRE_FACTOR = 2.0  # spitze Knicke: Versatz höchstens doppelt so weit wie verlangt


@dataclass(frozen=True)
class MarkingLayout:
    lanes: int


def parse_lanes(value) -> Optional[int]:
    """OSM-`lanes` als positive Ganzzahl, sonst None (fehlend, "2;3", "", "0", ...)."""
    try:
        lanes = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return lanes if lanes >= 1 else None


def marking_layout(
    tags: dict,
    width: float,
    internal_name: str,
    marked_highways: Collection[str],
    marked_surface: str,
    min_two_lane_width: float,
) -> Optional[MarkingLayout]:
    """
    Markierungs-Layout einer Straße oder None (keine Markierung): nur Straßentypen aus `marked_highways` mit der
    Oberfläche `marked_surface` (Asphalt) und ohne `lane_markings=no`. Fahrstreifen aus `lanes`; fehlt der Tag, ist
    eine Rampe (*_link) oder eine Straße schmaler als min_two_lane_width einspurig, alles andere zweispurig.
    """
    tags = tags or {}
    highway = str(tags.get("highway", ""))
    if tags.get("lane_markings") == "no" or highway not in marked_highways or internal_name != marked_surface:
        return None
    lanes = parse_lanes(tags.get("lanes"))
    if lanes is None:
        lanes = 1 if highway.endswith("_link") or width < min_two_lane_width else 2
    return MarkingLayout(lanes=lanes)


def line_offsets(widths: np.ndarray, lanes: int, edge_inset: float) -> List[Tuple[str, np.ndarray]]:
    """(Art, seitlicher Versatz je Knoten), positiv = links der Laufrichtung. Randlinien bei +-(Breite/2 -
    edge_inset), Leitlinien an den lanes-1 Fahrstreifengrenzen."""
    widths = np.asarray(widths, dtype=float)
    half = widths / 2.0
    lines = [(EDGE, half - edge_inset), (EDGE, -(half - edge_inset))]
    lines += [(DIVIDER, -half + k * widths / lanes) for k in range(1, lanes)]
    return lines


def offset_polyline(xy: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Polylinie mit Versatz je Knoten (positiv = links), an Knicken auf Gehrung. Nullsegmente (doppelte Knoten)
    übernehmen die Richtung des Nachbarsegments."""
    xy = np.asarray(xy, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    segments = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-9
    if not valid.any():
        return xy.copy()
    normals = np.zeros_like(segments)
    normals[valid] = np.column_stack([-segments[valid, 1], segments[valid, 0]]) / lengths[valid, None]
    last = normals[int(np.argmax(valid))]
    for i in range(len(normals)):
        if valid[i]:
            last = normals[i]
        else:
            normals[i] = last

    result = np.empty_like(xy)
    count = len(xy)
    for i in range(count):
        before, after = normals[max(i - 1, 0)], normals[min(i, count - 2)]
        miter = before + after
        norm = float(np.linalg.norm(miter))
        miter = after if norm < 1e-9 else miter / norm
        scale = 1.0 / max(float(np.dot(miter, after)), 1.0 / MAX_MITRE_FACTOR)
        result[i] = xy[i] + miter * offsets[i] * scale
    return result


def forward_indices(offset_xy: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """Indizes der Linienknoten, die in Fahrtrichtung vorankommen. In engen Kehren läuft die innere Linie sonst
    rückwärts (Versatz größer als der Kurvenradius) - diese Knoten fallen weg."""
    center_xy = np.asarray(center_xy, dtype=float)
    count = len(center_xy)
    kept = [0]
    for j in range(1, count):
        tangent = center_xy[min(j + 1, count - 1)] - center_xy[max(j - 1, 0)]
        if float(np.dot(offset_xy[j] - offset_xy[kept[-1]], tangent)) > 1e-9:
            kept.append(j)
    return np.array(kept, dtype=int)


def build_marking_lines(
    nodes: Sequence[Sequence[float]], layout: MarkingLayout, edge_inset: float
) -> List[Tuple[str, np.ndarray]]:
    """(Art, (N, 3)-Linie) für alle Markierungslinien einer Straße aus ihren DecalRoad-Knoten [x, y, z, width];
    z je Linienknoten vom zugehörigen Fahrbahnknoten (BeamNG projiziert die Linie ohnehin aufs Terrain)."""
    arr = np.asarray(nodes, dtype=float)
    center_xy = arr[:, :2]
    lines = []
    for kind, offsets in line_offsets(arr[:, 3], layout.lanes, edge_inset):
        offset_xy = offset_polyline(center_xy, offsets)
        kept = forward_indices(offset_xy, center_xy)
        if len(kept) >= 2:
            lines.append((kind, np.column_stack([offset_xy[kept], arr[kept, 2]])))
    return lines
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_markings.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/geometry/road_markings.py tests/geometry/test_road_markings.py
git commit -m "feat: Add road marking geometry (layout rules, per-node offsets, hairpin cleanup)"
```

---

### Task 4: Export – Breitenübergänge und Markierungs-DecalRoads (noch ohne Kreuzungsschnitt)

**Files:**
- Modify: `world_to_beamng/workflow/terrain_workflow.py` – `export_decal_roads()` (ab ca. Zeile 1092) und neuer
  Modul-Helfer `_road_marking_lines()` direkt nach `_structure_items()` (ca. Zeile 42)
- Test: `tests/workflow/test_decal_road_markings_export.py` (neu)

**Interfaces:**
- Consumes: `apply_width_transitions(...)` (Task 2), `marking_layout`, `build_marking_lines`, `EDGE` (Task 3),
  `OSM_MAPPER.road_markings`, `OSM_MAPPER.generate_marking_material_entry` und die `config`-Konstanten (Task 1),
  `drop_close_nodes` (`world_to_beamng/geometry/polygon.py`)
- Produces: `_road_marking_lines(specs, node_lists) -> List[Dict]` mit Einträgen `{"name", "nodes", "material"}`.
  `specs` ist eine Liste von `(road_slope_polygon: Dict, props: Dict, nodes)` je exportierter DecalRoad.
  Item-Namen: `marking_<road_id>_<line_idx>_<piece_idx>`, in diesem Task ist `piece_idx` immer `0`.

- [ ] **Step 1: Failing Tests schreiben** – `tests/workflow/test_decal_road_markings_export.py`:

```python
"""Tests für den DecalRoad-Export mit weichen Breitenübergängen und Markierungslinien
(TerrainWorkflow.export_decal_roads(), siehe docs/superpowers/plans/2026-09-24-road-markings-width-transitions.md)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _RecordingItems:
    def __init__(self):
        self.roads = {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = {"nodes": nodes, "material": material, **extra}


def _export(polys):
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}))
    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})
    return count, stub.items.roads, stub.materials.materials


def _poly(road_id, points, **tags):
    coords = np.array([[float(x), float(y), 100.0] for x, y in points])
    return {"road_id": road_id, "trimmed_centerline": coords, "osm_tags": tags}


def _markings(roads):
    return {name: road for name, road in roads.items() if name.startswith("marking_")}


def test_two_lane_primary_gets_two_edge_lines_and_a_dashed_divider():
    count, roads, materials = _export([_poly(1, [(0, 0), (10, 0), (20, 0), (30, 0)], highway="primary", lanes="2")])

    assert count == 1  # Rückgabe zählt weiterhin nur Fahrbahnen
    markings = _markings(roads)
    assert sorted(markings) == ["marking_1_0_0", "marking_1_1_0", "marking_1_2_0"]
    left, right, divider = markings["marking_1_0_0"], markings["marking_1_1_0"], markings["marking_1_2_0"]
    assert left["material"] == right["material"] == config.ROAD_MARKING_EDGE_MATERIAL
    assert divider["material"] == config.ROAD_MARKING_DIVIDER_MATERIAL
    assert [n[1] for n in left["nodes"]] == pytest.approx([3.0] * 4)  # 6,5 m / 2 - 0,25 m
    assert [n[1] for n in right["nodes"]] == pytest.approx([-3.0] * 4)
    assert [n[1] for n in divider["nodes"]] == pytest.approx([0.0] * 4)
    assert all(n[3] == config.ROAD_MARKING_LINE_WIDTH for n in left["nodes"])
    assert left["drivability"] == -1
    assert left["renderPriority"] == config.ROAD_MARKING_RENDER_PRIORITY
    assert divider["textureLength"] == pytest.approx(
        config.OSM_MAPPER.road_markings[config.ROAD_MARKING_DIVIDER_MATERIAL]["textureLength"]
    )
    assert {config.ROAD_MARKING_EDGE_MATERIAL, config.ROAD_MARKING_DIVIDER_MATERIAL} <= set(materials)


def test_single_lane_link_gets_edge_lines_only():
    _, roads, _ = _export([_poly(7, [(0, 0), (10, 0), (20, 0)], highway="primary_link")])

    assert sorted(_markings(roads)) == ["marking_7_0_0", "marking_7_1_0"]


@pytest.mark.parametrize(
    "tags",
    [
        {"highway": "residential"},
        {"highway": "track"},
        {"highway": "secondary", "lanes": "2", "lane_markings": "no"},
        {"highway": "secondary", "lanes": "2", "surface": "sett"},
    ],
)
def test_unmarked_roads_get_no_markings(tags):
    _, roads, materials = _export([_poly(1, [(0, 0), (10, 0), (20, 0)], **tags)])

    assert _markings(roads) == {}
    assert config.ROAD_MARKING_EDGE_MATERIAL not in materials


def test_markings_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "ROAD_MARKINGS_ENABLED", False)
    _, roads, _ = _export([_poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2")])

    assert _markings(roads) == {}


def test_width_transition_is_applied_to_road_and_followed_by_edge_line():
    polys = [
        _poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2"),
        _poly(2, [(20, 0), (30, 0), (40, 0)], highway="primary", lanes="3"),
    ]
    _, roads, _ = _export(polys)

    assert roads["road_1"]["nodes"][-1][3] == pytest.approx(8.125)  # Mittel aus 6,5 und 9,75 m
    assert roads["road_1"]["nodes"][0][3] == pytest.approx(6.5)
    assert roads["road_2"]["nodes"][0][3] == pytest.approx(8.125)
    assert roads["marking_1_0_0"]["nodes"][-1][1] == pytest.approx(8.125 / 2 - 0.25)


def test_marking_nodes_keep_min_spacing():
    polys = [_poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2"),
             _poly(2, [(20, 0), (30, 0), (40, 0)], highway="primary", lanes="3")]
    _, roads, _ = _export(polys)

    for road in roads.values():
        pts = np.array(road["nodes"])[:, :2]
        assert np.linalg.norm(np.diff(pts, axis=0), axis=1).min() >= config.DECAL_ROAD_MIN_NODE_SPACING
```

- [ ] **Step 2: Test laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/workflow/test_decal_road_markings_export.py -v`
Expected: FAIL, u.a. `assert [] == ['marking_1_0_0', 'marking_1_1_0', 'marking_1_2_0']`

- [ ] **Step 3: Modul-Helfer einfügen** – in `world_to_beamng/workflow/terrain_workflow.py` direkt nach der Funktion
`_structure_items()`:

```python
def _road_marking_lines(specs: List[Tuple[Dict, Dict, List]], node_lists: List[List[List[float]]]) -> List[Dict]:
    """
    Markierungslinien (Rand- und Leitlinien) aller markierten DecalRoads als {"name", "nodes", "material"} - siehe
    geometry/road_markings.py. `specs`: (road_slope_polygon, road_props, _) je DecalRoad, `node_lists`: deren fertige
    Knoten (mit Breitenübergängen), in derselben Reihenfolge.
    """
    from ..geometry.polygon import drop_close_nodes
    from ..geometry.road_markings import EDGE, build_marking_lines, marking_layout

    lines = []
    for (poly, props, _), nodes in zip(specs, node_lists):
        layout = marking_layout(
            poly.get("osm_tags", {}),
            float(props.get("width", 4.0)),
            props.get("internal_name", ""),
            config.ROAD_MARKING_HIGHWAYS,
            config.ROAD_MARKING_SURFACE,
            config.ROAD_MARKING_MIN_TWO_LANE_WIDTH,
        )
        if layout is None:
            continue
        for line_idx, (kind, line) in enumerate(build_marking_lines(nodes, layout, config.ROAD_MARKING_EDGE_INSET)):
            material = config.ROAD_MARKING_EDGE_MATERIAL if kind == EDGE else config.ROAD_MARKING_DIVIDER_MATERIAL
            line_nodes = [[x, y, z, config.ROAD_MARKING_LINE_WIDTH] for x, y, z in line.tolist()]
            # Innen in Kurven rücken die Linienknoten zusammen - dieselbe Mindestsegmentlänge wie bei der Fahrbahn
            line_nodes = drop_close_nodes(line_nodes, config.DECAL_ROAD_MIN_NODE_SPACING)
            if len(line_nodes) >= 2:
                lines.append({"name": f"marking_{poly['road_id']}_{line_idx}_0", "nodes": line_nodes, "material": material})
    return lines
```

- [ ] **Step 4: `export_decal_roads()` umbauen** – den Rumpf ab `from ..config import OSM_MAPPER` bis einschließlich
`return count` durch Folgendes ersetzen. Die bestehenden Kommentare (Nulllängen-Straßen, Mindestabstand,
renderPriority) bleiben wortgleich erhalten, der Docstring bleibt. Neu sind nur das zweiphasige Vorgehen
(sammeln → Übergänge → schreiben) und der Markierungsblock:

```python
        from ..config import OSM_MAPPER
        from ..geometry.polygon import drop_close_nodes
        from ..geometry.road_width_transitions import apply_width_transitions

        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        unique_materials: Dict[str, Dict] = {}
        specs = []  # (road_slope_polygon, road_props, nodes) je exportierbarer DecalRoad

        for poly in road_slope_polygons_2d:
            if poly.get("structure_type", "surface") != "surface":
                continue
            road_id = poly.get("road_id")
            centerline = poly.get("trimmed_centerline")
            if road_id is None or centerline is None or len(centerline) < 2:
                continue

            # Entartete (Nulllängen-)Straßen überspringen: Clipping/Junction-
            # Split können vereinzelt einen "Rest" mit 2 identischen Punkten
            # hinterlassen. Ein DecalRoad mit Länge 0 ist ein degenerierter
            # Spline (im alten Mesh-Ansatz war das ein unsichtbares
            # Nulldreieck, hier würde es ein kaputtes Decal-Item erzeugen).
            xy_unique = {(round(float(x), 3), round(float(y), 3)) for x, y, _ in centerline}
            if len(xy_unique) < 2:
                continue

            props = OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
            width = float(props.get("width", 4.0))
            nodes = [[float(x), float(y), float(z), width] for x, y, z in centerline]

            # Zu kurze Segmente entfernen: BeamNG zeichnet ein DecalRoad mit
            # einem zu kurzen Segment (z.B. 0,10 m vom Junction-Schnitt neben
            # einem Resample-Punkt) gar nicht - das ganze Stück fehlt dann.
            nodes = drop_close_nodes(nodes, config.DECAL_ROAD_MIN_NODE_SPACING)
            if len(nodes) < 2:
                continue
            specs.append((poly, props, nodes))

        # Weiche Breitenübergänge an Geradeaus-Stößen (je 5 m davor/dahinter, Spline) - siehe
        # geometry/road_width_transitions.py. Fügt Knoten nur mit >= DECAL_ROAD_MIN_NODE_SPACING Abstand ein.
        node_lists = apply_width_transitions(
            [nodes for _, _, nodes in specs],
            transition_length=config.ROAD_WIDTH_TRANSITION_LENGTH,
            step=config.ROAD_WIDTH_TRANSITION_STEP,
            endpoint_tol=config.ROAD_CONTINUATION_ENDPOINT_TOL,
            max_angle_deg=config.ROAD_CONTINUATION_MAX_ANGLE_DEG,
            min_delta=config.ROAD_WIDTH_TRANSITION_MIN_DELTA,
            min_spacing=config.DECAL_ROAD_MIN_NODE_SPACING,
        )

        count = 0
        for (poly, props, _), nodes in zip(specs, node_lists):
            mat_name = props.get("internal_name", "road_default")
            unique_materials[mat_name] = props

            # renderPriority aus dem vorhandenen "priority"-Feld ableiten
            # (surface_types in data/osm_to_beamng.json): an Kreuzungen
            # überlappen sich die (immer volle Breite habenden) Enden
            # mehrerer DecalRoad-Objekte - ohne explizite, konsistente
            # Zeichenreihenfolge sortiert BeamNG das beliebig, was an
            # Kreuzungen wie ein "Flickenteppich" aussieht. Höherwertige
            # Straßen (Asphalt) werden so immer über niedrigerwertigen
            # (Dirt/Concrete) gezeichnet.
            render_priority = int(props.get("priority", 0))

            self.items.add_decal_road(
                name=f"road_{poly.get('road_id')}",
                nodes=nodes,
                material=mat_name,
                drivability=props.get("drivability", 1.0),
                overwrite=True,
                autoLanes=True,
                autoJunction=True,
                improvedSpline=True,
                renderPriority=render_priority,
            )
            count += 1

        road_material_entries = [
            OSM_MAPPER.generate_materials_json_entry(mat_name, props) for mat_name, props in unique_materials.items()
        ]
        for mat_entry in road_material_entries:
            mat_name = mat_entry.pop("__name", None)
            if mat_name:
                self.materials.materials[mat_name] = mat_entry

        # Fahrbahnmarkierungen als eigene schmale DecalRoads obenauf (geometry/road_markings.py). drivability=-1:
        # BeamNGs KI-Straßennetz (lua/ge/map.lua) übernimmt nur DecalRoads mit drivability > 0.
        marking_count = 0
        if config.ROAD_MARKINGS_ENABLED:
            used_markings = set()
            for line in _road_marking_lines(specs, node_lists):
                marking = OSM_MAPPER.road_markings[line["material"]]
                self.items.add_decal_road(
                    name=line["name"],
                    nodes=line["nodes"],
                    material=line["material"],
                    drivability=-1,
                    overwrite=True,
                    improvedSpline=True,
                    textureLength=marking["textureLength"],
                    renderPriority=config.ROAD_MARKING_RENDER_PRIORITY,
                )
                used_markings.add(line["material"])
                marking_count += 1
            for mat_name in sorted(used_markings):
                self.materials.materials[mat_name] = OSM_MAPPER.generate_marking_material_entry(
                    mat_name, OSM_MAPPER.road_markings[mat_name]
                )

        logger.debug(
            f"  [OK] {count} DecalRoad-Item(s) exportiert ({len(unique_materials)} Materialien), "
            f"{marking_count} Markierungslinie(n)"
        )
        return count
```

Hinweis: Materialien werden jetzt nur noch für tatsächlich exportierte Straßen registriert. Vorher kam auch das
Material einer Straße hinzu, die an `drop_close_nodes` scheiterte.

- [ ] **Step 5: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/workflow/test_decal_road_markings_export.py tests/geometry/test_decal_road_nodes.py -v`
Expected: alle PASS

- [ ] **Step 6: Ganze Suite, dann Commit**

Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün.

```bash
git add world_to_beamng/workflow/terrain_workflow.py tests/workflow/test_decal_road_markings_export.py
git commit -m "feat: Export width transitions and edge/centre road markings as DecalRoads"
```

---

### Task 5: Prüfung im Spiel (Prototyp ohne Kreuzungsschnitt) – Checkpoint mit dem User

Kein Code. Der Task klärt die Punkte, die nur im Spiel sichtbar sind, bevor Task 6 darauf aufbaut. Die Befunde gehen an
den User. Werte in `config.py` / `osm_to_beamng.json` nur nach Rücksprache ändern.

- [ ] **Step 1: Level exportieren**

Run: `.\.venv\Scripts\python.exe world_to_beamng.py`
Expected: Lauf ohne Fehler. Danach zählen:
`.\.venv\Scripts\python.exe -c "import pathlib,re; p=pathlib.Path(r'C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\levels\world_to_beamng\main\MissionGroup\items.level.json'); t=p.read_text(encoding='utf-8'); print('markings', len(re.findall(r'\"name\": ?\"marking_', t)))"`
Erwartung: einige hundert Markierungslinien (rund 150 markierte Hauptstraßen-Stücke mit 2–3 Linien).

- [ ] **Step 2: Level in BeamNG laden, Log prüfen** – `C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\beamng.log`
gegen Ende auf `|E|`-Zeilen prüfen, vor allem `line_white`, `line_dashed_long`, `Missing source texture` und DecalRoad.

- [ ] **Step 3: Sichtprüfung (dem User zur Bestätigung vorlegen)**
  1. **Zeichenreihenfolge:** Liegen die Linien sichtbar **über** dem Asphalt? Falls nicht, gilt `renderPriority`
     umgekehrt (kleiner = oben). Dann liegen auch die bestehenden Straßenprioritäten falsch herum (Asphalt 8 über
     Erdweg 2). Befund festhalten und mit dem User entscheiden.
  2. **Strichbild der Leitlinie:** Rund 6 m Strich, 8 m Lücke? Bei Bedarf `textureLength` in
     `data/osm_to_beamng.json → road_markings.line_divider_dashed` anpassen.
  3. **Kehren** der Nuova strada del San Gottardo (Tremola hat `lane_markings=no` und bleibt unmarkiert): Innere
     Randlinie ohne Zacken und Schlaufen?
  4. **Breitenübergänge** an den `lanes`-Wechseln der Hauptstraße 2 (2 → 3 → 4 Spuren): 10 m weich statt Stufe?
     Folgen die Randlinien?
  5. **Kreuzungen:** Linien laufen hier noch durch Einmündungen. Das ist erwartet und wird in Task 6 behoben.
  6. **KI-Verkehr** einschalten: Fahren die Fahrzeuge normal auf der Fahrbahn und nicht entlang der Linien?

---

### Task 6: Linien an Einmündungen und Kreuzungen unterbrechen

**Files:**
- Modify: `world_to_beamng/geometry/road_markings.py` (neue Funktionen am Dateiende, Konstante `BOUNDARY_EPS`)
- Modify: `world_to_beamng/workflow/terrain_workflow.py` (`_road_marking_lines()` aus Task 4)
- Test: `tests/geometry/test_road_markings.py` (ergänzen), `tests/workflow/test_decal_road_markings_export.py` (ergänzen)

**Interfaces:**
- Consumes: `find_continuations`, `continuation_partners` (Task 2); `build_marking_lines` (Task 3); `_road_marking_lines` (Task 4)
- Produces: `clip_line(line: np.ndarray, obstacles, min_length: float) -> List[np.ndarray]`,
  `road_surface_polygon(nodes, clearance: float) -> shapely.Polygon`,
  `junction_obstacles(index: int, polygons, tree, excluded) -> Optional[shapely geometry]`

- [ ] **Step 1: Failing Tests ergänzen** – ans Ende von `tests/geometry/test_road_markings.py` anhängen und oben den
Import `from shapely.geometry import LineString` zu `from shapely.geometry import LineString, box` erweitern:

```python
from shapely import STRtree

from world_to_beamng.geometry.road_markings import clip_line, junction_obstacles, road_surface_polygon


def test_clip_line_cuts_out_junction_area_and_interpolates_z():
    line = np.array([[0.0, 3.0, 100.0], [40.0, 3.0, 104.0]])
    pieces = clip_line(line, box(18.0, -10.0, 22.0, 10.0), 1.0)
    assert len(pieces) == 2
    assert pieces[0][-1, 0] == pytest.approx(18.0)
    assert pieces[0][-1, 2] == pytest.approx(101.8)
    assert pieces[1][0, 0] == pytest.approx(22.0)


def test_clip_line_drops_short_pieces_and_handles_no_obstacles():
    line = np.array([[0.0, 3.0, 100.0], [40.0, 3.0, 100.0]])
    assert len(clip_line(line, box(0.5, -10.0, 39.5, 10.0), 1.0)) == 0
    assert len(clip_line(line, None, 1.0)) == 1


def test_junction_obstacles_skip_self_and_excluded_and_far_roads():
    main_a = [[-20.0, 0.0, 0.0, 6.5], [0.0, 0.0, 0.0, 6.5]]
    main_b = [[0.0, 0.0, 0.0, 6.5], [20.0, 0.0, 0.0, 6.5]]
    side = [[0.0, 0.0, 0.0, 5.0], [0.0, 20.0, 0.0, 5.0]]
    far = [[100.0, 100.0, 0.0, 5.0], [120.0, 100.0, 0.0, 5.0]]
    polygons = [road_surface_polygon(n, 0.5) for n in (main_a, main_b, side, far)]
    tree = STRtree(polygons)

    obstacles = junction_obstacles(0, polygons, tree, excluded={1})
    assert obstacles.symmetric_difference(polygons[2]).area < 1.0  # bis auf den 1-cm-Rand (Umfang 52 m)
    assert junction_obstacles(3, polygons, tree, excluded=set()) is None
    assert polygons[2].bounds == pytest.approx((-3.0, 0.0, 3.0, 20.0))


def test_side_road_cuts_gap_into_main_road_edge_line_only_on_its_side():
    main = [[-20.0, 0.0, 0.0, 6.5], [0.0, 0.0, 0.0, 6.5], [20.0, 0.0, 0.0, 6.5]]
    side = [[0.0, 0.0, 0.0, 5.0], [0.0, 20.0, 0.0, 5.0]]  # beginnt am gemeinsamen Knoten auf der Mittellinie
    polygons = [road_surface_polygon(n, 0.5) for n in (main, side)]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set())
    left, right, divider = [line for _, line in build_marking_lines(main, MarkingLayout(lanes=2), 0.25)]
    assert len(clip_line(left, obstacles, 1.0)) == 2  # Lücke in der Randlinie auf der Einmündungsseite
    assert len(clip_line(right, obstacles, 1.0)) == 1  # gegenüber durchgehend
    assert len(clip_line(divider, obstacles, 1.0)) == 1  # Leitlinie läuft an der T-Einmündung durch


def test_crossing_road_interrupts_divider():
    main = [[-20.0, 0.0, 0.0, 6.5], [0.0, 0.0, 0.0, 6.5], [20.0, 0.0, 0.0, 6.5]]
    crossing = [[0.0, -20.0, 0.0, 5.0], [0.0, 20.0, 0.0, 5.0]]
    polygons = [road_surface_polygon(n, 0.5) for n in (main, crossing)]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set())
    divider = build_marking_lines(main, MarkingLayout(lanes=2), 0.25)[2][1]
    pieces = clip_line(divider, obstacles, 1.0)
    assert len(pieces) == 2
    assert pieces[0][-1, 0] == pytest.approx(-3.0, abs=0.02)
```

und ans Ende von `tests/workflow/test_decal_road_markings_export.py`:

```python
def test_side_road_interrupts_main_edge_line_but_not_divider():
    polys = [
        _poly(1, [(-40, 0), (-20, 0), (0, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 0), (20, 0), (40, 0)], highway="primary", lanes="2"),
        _poly(3, [(0, 0), (0, 20), (0, 40)], highway="secondary", lanes="2"),
    ]
    _, roads, _ = _export(polys)

    clearance = 6.5 / 2 + config.ROAD_MARKING_JUNCTION_CLEARANCE  # 3,75 m
    assert roads["marking_1_0_0"]["nodes"][-1][0] == pytest.approx(-clearance, abs=0.02)  # links endet vor der Einmündung
    assert "marking_1_0_1" not in roads
    assert roads["marking_1_1_0"]["nodes"][-1][0] == pytest.approx(0.0)  # rechte Randlinie läuft durch
    assert roads["marking_1_2_0"]["nodes"][-1][0] == pytest.approx(0.0)  # Leitlinie läuft durch
    assert roads["marking_2_0_0"]["nodes"][0][0] == pytest.approx(clearance, abs=0.02)
    for side_edge in ("marking_3_0_0", "marking_3_1_0"):
        assert roads[side_edge]["nodes"][0][1] == pytest.approx(clearance, abs=0.02)  # Nebenstraße beginnt am Rand


def test_track_junction_does_not_interrupt_edge_line():
    polys = [
        _poly(1, [(-40, 0), (0, 0), (40, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 0), (0, 20), (0, 40)], highway="track"),
    ]
    _, roads, _ = _export(polys)

    assert "marking_1_0_1" not in roads
    assert roads["marking_1_0_0"]["nodes"][-1][0] == pytest.approx(40.0)
```

- [ ] **Step 2: Tests laufen lassen, müssen fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_markings.py tests/workflow/test_decal_road_markings_export.py -v`
Expected: FAIL mit `ImportError: cannot import name 'clip_line'`

- [ ] **Step 3: Funktionen in `world_to_beamng/geometry/road_markings.py` ergänzen** – unter
`MAX_MITRE_FACTOR = …` die Konstante

```python
BOUNDARY_EPS = 0.01  # Hindernisflächen um 1 cm schrumpfen, siehe junction_obstacles()
```

und ans Dateiende:

```python
def clip_line(line: np.ndarray, obstacles, min_length: float) -> List[np.ndarray]:
    """Teile einer (N, 3)-Linie außerhalb von `obstacles` (shapely-Fläche, z.B. die Fahrbahnen einmündender Straßen)
    mit mindestens min_length Länge; z linear entlang der ursprünglichen Linie."""
    from shapely.geometry import LineString, Point

    shape = LineString(line[:, :2])
    if obstacles is None or obstacles.is_empty:
        pieces = [shape]
    else:
        rest = shape.difference(obstacles)
        pieces = list(getattr(rest, "geoms", [rest]))
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1))])
    result = []
    for piece in pieces:
        if piece.is_empty or piece.geom_type != "LineString" or piece.length < min_length:
            continue
        xy = np.asarray(piece.coords, dtype=float)
        s = np.array([shape.project(Point(p)) for p in xy])
        result.append(np.column_stack([xy, np.interp(s, cum, line[:, 2])]))
    return result


def road_surface_polygon(nodes: Sequence[Sequence[float]], clearance: float):
    """Fahrbahnfläche einer DecalRoad (Puffer um die Mittellinie mit der größten Knotenbreite, flache Enden),
    um `clearance` verbreitert."""
    from shapely.geometry import LineString

    arr = np.asarray(nodes, dtype=float)
    return LineString(arr[:, :2]).buffer(float(arr[:, 3].max()) / 2.0 + clearance, cap_style="flat")


def junction_obstacles(index: int, polygons: Sequence, tree, excluded: Collection[int]):
    """
    Vereinigung der Fahrbahnflächen, die die Fläche `index` berühren - ohne sie selbst und ohne `excluded`
    (Geradeaus-Partner, Wege ohne Markierungslücke). None, wenn keine übrig bleibt. `tree`: shapely.STRtree über
    `polygons`.

    Um BOUNDARY_EPS geschrumpft: eine Nebenstraße beginnt am gemeinsamen Knoten auf der Mittellinie der Hauptstraße,
    ihre flache Kante liegt also genau auf deren Leitlinie. Ohne das Schrumpfen bekäme die Leitlinie an jeder
    T-Einmündung eine Lücke; in einer echten Kreuzung (Fläche überdeckt die Linie) wird sie weiterhin unterbrochen.
    """
    from shapely import unary_union

    others = [polygons[j] for j in tree.query(polygons[index]) if j != index and j not in excluded]
    if not others:
        return None
    return unary_union(others).buffer(-BOUNDARY_EPS)
```

- [ ] **Step 4: `_road_marking_lines()` in `terrain_workflow.py` auf Clipping umstellen** – die ganze Funktion aus
Task 4 ersetzen durch:

```python
def _road_marking_lines(specs: List[Tuple[Dict, Dict, List]], node_lists: List[List[List[float]]]) -> List[Dict]:
    """
    Markierungslinien (Rand- und Leitlinien) aller markierten DecalRoads als {"name", "nodes", "material"} - siehe
    geometry/road_markings.py. `specs`: (road_slope_polygon, road_props, _) je DecalRoad, `node_lists`: deren fertige
    Knoten (mit Breitenübergängen), in derselben Reihenfolge.

    An Einmündungen und Kreuzungen werden die Linien unterbrochen: geschnitten wird mit den Fahrbahnflächen aller
    berührenden Straßen außer den Geradeaus-Partnern (dort läuft die Linie weiter) und außer Feld-/Fußwegen
    (ROAD_MARKING_NO_GAP_HIGHWAYS).
    """
    from shapely import STRtree

    from ..geometry.polygon import drop_close_nodes
    from ..geometry.road_markings import (
        EDGE,
        build_marking_lines,
        clip_line,
        junction_obstacles,
        marking_layout,
        road_surface_polygon,
    )
    from ..geometry.road_width_transitions import continuation_partners, find_continuations

    if not specs:
        return []
    partners = continuation_partners(
        find_continuations(node_lists, config.ROAD_CONTINUATION_ENDPOINT_TOL, config.ROAD_CONTINUATION_MAX_ANGLE_DEG)
    )
    polygons = [road_surface_polygon(nodes, config.ROAD_MARKING_JUNCTION_CLEARANCE) for nodes in node_lists]
    tree = STRtree(polygons)
    no_gap = {
        i for i, (poly, _, _) in enumerate(specs)
        if poly.get("osm_tags", {}).get("highway") in config.ROAD_MARKING_NO_GAP_HIGHWAYS
    }

    lines = []
    for index, ((poly, props, _), nodes) in enumerate(zip(specs, node_lists)):
        layout = marking_layout(
            poly.get("osm_tags", {}),
            float(props.get("width", 4.0)),
            props.get("internal_name", ""),
            config.ROAD_MARKING_HIGHWAYS,
            config.ROAD_MARKING_SURFACE,
            config.ROAD_MARKING_MIN_TWO_LANE_WIDTH,
        )
        if layout is None:
            continue
        obstacles = junction_obstacles(index, polygons, tree, excluded=partners.get(index, set()) | no_gap)
        for line_idx, (kind, line) in enumerate(build_marking_lines(nodes, layout, config.ROAD_MARKING_EDGE_INSET)):
            material = config.ROAD_MARKING_EDGE_MATERIAL if kind == EDGE else config.ROAD_MARKING_DIVIDER_MATERIAL
            for piece_idx, piece in enumerate(clip_line(line, obstacles, config.ROAD_MARKING_MIN_PIECE_LENGTH)):
                line_nodes = [[x, y, z, config.ROAD_MARKING_LINE_WIDTH] for x, y, z in piece.tolist()]
                # Innen in Kurven rücken die Linienknoten zusammen - dieselbe Mindestsegmentlänge wie bei der Fahrbahn
                line_nodes = drop_close_nodes(line_nodes, config.DECAL_ROAD_MIN_NODE_SPACING)
                if len(line_nodes) >= 2:
                    lines.append(
                        {"name": f"marking_{poly['road_id']}_{line_idx}_{piece_idx}", "nodes": line_nodes, "material": material}
                    )
    return lines
```

- [ ] **Step 5: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_markings.py tests/workflow/test_decal_road_markings_export.py -v`
Expected: alle PASS (Geometrie 22, Export 11)

- [ ] **Step 6: Ganze Suite, dann Commit**

Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün.

```bash
git add world_to_beamng/geometry/road_markings.py world_to_beamng/workflow/terrain_workflow.py tests/geometry/test_road_markings.py tests/workflow/test_decal_road_markings_export.py
git commit -m "feat: Interrupt road markings at junctions and crossings"
```

---

### Task 7: Zweite Prüfung im Spiel und Dokumentation

**Files:**
- Modify: `docs/OSM_ROAD_ANALYSIS.md` (Abschnitt 4)

- [ ] **Step 1: Export und Laden wie in Task 5, Steps 1–2.**

- [ ] **Step 2: Sichtprüfung mit dem User**
  1. T-Einmündungen: Randlinie auf der Einmündungsseite unterbrochen, gegenüber und Leitlinie durchgehend.
  2. Kreuzungen: alle Linien unterbrochen.
  3. Auf- und Abfahrten (`primary_link`) am Boden, z.B. Knoten 24869503 „Motto Bartola“: Die Randlinien der Rampe
     beginnen am Rand der Hauptfahrbahn.
  4. Feldwege (`track`) schneiden keine Lücke.
  5. Keine neuen `|E|`-Zeilen in `beamng.log`.

- [ ] **Step 3: Doku nachführen** – In `docs/OSM_ROAD_ANALYSIS.md` unter Abschnitt 4 einen Punkt ergänzen:
**Markierungen + Breitenübergänge (umgesetzt <Datum>)**. Darin: Rand- und Leitlinien als Vanilla-Linien-DecalRoads,
Regeln (Straßentypen, Asphalt, `lane_markings`, Spurzahl-Fallback), 10-m-Spline-Übergang, Kreuzungsschnitt, die in
Task 5/7 bestätigte `renderPriority`-Richtung und das gewählte Strichbild. Offen bleiben Brückendecks,
durchgezogene Mittellinie bei mehrspurigem Gegenverkehr und Einbahn.

- [ ] **Step 4: Commit**

```bash
git add docs/OSM_ROAD_ANALYSIS.md
git commit -m "docs: Record road markings and width transitions in the OSM road analysis"
```
