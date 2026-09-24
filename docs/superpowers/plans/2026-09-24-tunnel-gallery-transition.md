# Übergang Tunnel ↔ Galerie – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Geht eine Tunnelröhre direkt in eine Galerie über, entsteht ein durchgehendes Bauwerk: Die Röhre endet an
einer Betonstirnwand mit Galerie-Öffnung, die Galerie schließt ohne eigene Stirnflächen an, und die Überdeckung
läuft ohne Erdwand in der Röhre bis in den Berg.

**Architecture:**
- `classify_structure()` erkennt `covered=yes` ohne `tunnel`-Tag als Galerie.
- `plan_tunnels()` markiert Portale, die auf einem Galerie-Endpunkt liegen, als `kind="gallery"` und vergrößert
  ihren Block auf den Galerie-Querschnitt.
- `build_portal_block_mesh()` baut für diese Portale eine Stirnwand mit Rechteck-Öffnung plus eine nach innen
  gerichtete Stufenfläche.
- `build_gallery_mesh()` lässt am Übergang die Stirnfläche weg.
- `tunnel_terrain` behandelt Übergangs-Portale immer als offen und überdeckt kurze Lücken zwischen Portal und Berg.

**Tech Stack:** Python 3.13, numpy, shapely 2.1 (`constrained_delaunay_triangles`, `Polygon.difference`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md`

## Global Constraints

- **Klassifizierung:**
  - `covered=yes` und `tunnel` fehlt oder `no` → `"gallery"`
  - `tunnel=yes` + `covered=yes` → `"tunnel"`
  - `bridge` hat weiter Vorrang vor allem
- **Übergang:** Portalpunkt höchstens `TUNNEL_TRANSITION_ENDPOINT_TOL = 0.5` m von einem **Endpunkt** einer
  Galerie-Centerline.
- **Block am Übergang:**
  - `half_width = max(radius + wing, gallery_half_width + GALLERY_WALL_THICKNESS)`
  - `top_z = max(top_z, floor_z + GALLERY_HEIGHT + GALLERY_ROOF_THICKNESS + 0.2)`
  - `bottom_z = min(bottom_z, floor_z - GALLERY_FLOOR_THICKNESS)`
- **Öffnung zur Galerie:** Rechteck `±gallery_half_width` × `0 … GALLERY_HEIGHT` über dem Röhrenboden.
- **Überdeckungslücke:** `TUNNEL_COVER_GAP_MAX = 25.0` m, gilt für alle offenen Portale.
- **Sprachen:** Chat und Code-Kommentare auf Deutsch; Bezeichner und Strings im Code auf Englisch (CLAUDE.md).
- **Nicht committen:** Im Arbeitsbaum liegen fremde, unkommittete Änderungen in denselben Dateien. Jeder Task endet
  mit grüner Gesamt-Suite: `.\.venv\Scripts\python.exe -m pytest -q` (Stand vor dem Plan: 1162 passed, 2 skipped).
- **Vor Task 7:** Der OSM-Cache muss den Stand nach der Nutzer-Korrektur enthalten. Way 430132545 hat dann kein
  `tunnel`-Tag mehr.

## Review Focus

1. **Tunnel mit Übergang an beiden Enden** (Tunnel Banchi, 44 m): Beide Portale müssen `kind="gallery"` sein, keins
   darf „offen ins Gelände“ werden. Test: `test_both_ends_can_be_gallery_transitions` (Task 2).
2. **Galerie, die den Tunnel nur mit der Mitte berührt** (kein Endpunkt): Das ist kein Übergang. Die Röhre bekäme
   sonst eine Galerie-Wand vor ein offenes Portal. Test: `test_gallery_touching_the_portal_with_its_middle_is_no_transition`
   (Task 2).
3. **Breitere Galerie als Tunnel** (z.B. 9,75 m Fahrbahn): Der Block muss den ganzen Galerie-Querschnitt abdecken,
   sonst sieht man seitlich in hohle Wandquader. Test: `test_wide_gallery_widens_the_block` (Task 2).
4. **Galerie gegen die Tunnelrichtung digitalisiert** (Endpunkt am Tunnel ist ihr Start statt ihr Ende): Die
   Stirnfläche muss am richtigen Ende entfallen. Test: `test_transition_removes_only_the_cap_at_the_transition_end`
   mit beiden Richtungen (Task 4).
5. **Tunnel, der nirgends im Gelände steckt** (offene Galerie im Höhenmodell): Die Lückenfüllung darf keinen Damm
   erzeugen. Test: `test_cover_gap_longer_than_the_limit_stays_open` (Task 5); der bestehende Test
   `test_no_cover_dam_where_the_tube_is_not_at_least_half_in_the_ground` bleibt grün.

---

## Dateistruktur

| Datei | Änderung |
|---|---|
| `world_to_beamng/geometry/road_structures.py` | `classify_structure`: Regel `covered=yes` |
| `world_to_beamng/tunnels/tunnel_portal.py` | `plan_tunnels`: Übergänge erkennen, Block vergrößern; `build_portal_block_mesh`: Stirnwand mit Rechteck-Öffnung + Stufenfläche |
| `world_to_beamng/tunnels/gallery_mesh.py` | `build_gallery_mesh(cap_start, cap_end)`, `build_galleries(transition_points, transition_tol)` |
| `world_to_beamng/terrain/tunnel_terrain.py` | Übergangs-Portale immer offen; Lückenfüllung in `_raise_cover` |
| `world_to_beamng/workflow/terrain_workflow.py` | Modul-Helfer `_plan_tunnels()`, Übergänge an `build_galleries`, `cover_gap_max` durchreichen |
| `world_to_beamng/config.py` | `TUNNEL_TRANSITION_ENDPOINT_TOL`, `TUNNEL_COVER_GAP_MAX` |
| Tests | `tests/geometry/test_road_structures.py`, `tests/tunnels/test_tunnel_gallery_transition.py` (neu), `tests/tunnels/test_gallery_mesh.py`, `tests/terrain/test_tunnel_terrain.py`, `tests/workflow/test_terrain_workflow_tunnels.py` |

---

### Task 1: Klassifizierung `covered=yes` → Galerie

**Files:**
- Modify: `world_to_beamng/geometry/road_structures.py:9-25`
- Test: `tests/geometry/test_road_structures.py`

**Interfaces:**
- Produces: `classify_structure(osm_tags) -> str` mit neuer Regel.

- [ ] **Step 1: Failing Tests anhängen** an `tests/geometry/test_road_structures.py`:

```python
def test_covered_road_without_tunnel_tag_is_a_gallery():
    # z.B. die lange Galerie der Nuova strada (Way 746194686) und Galleria artificiale Banchi (430132545)
    assert classify_structure({"highway": "primary", "covered": "yes"}) == "gallery"
    assert classify_structure({"highway": "primary", "covered": "yes", "tunnel": "no"}) == "gallery"


def test_covered_tunnel_stays_a_tunnel():
    assert classify_structure({"highway": "primary", "covered": "yes", "tunnel": "yes"}) == "tunnel"


def test_covered_no_is_a_surface_road():
    assert classify_structure({"highway": "primary", "covered": "no"}) == "surface"


def test_covered_bridge_stays_a_bridge():
    assert classify_structure({"highway": "primary", "covered": "yes", "bridge": "yes"}) == "bridge"
```

- [ ] **Step 2: Laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_structures.py -v`
Expected: FAIL in `test_covered_road_without_tunnel_tag_is_a_gallery` (`'surface' == 'gallery'`)

- [ ] **Step 3: Implementieren** – `classify_structure` ersetzen durch:

```python
def classify_structure(osm_tags: Dict) -> str:
    """
    "bridge" | "tunnel" | "gallery" | "surface", anhand von `bridge`/`tunnel`/`covered`-Tags.

    Reihenfolge: bridge=* (außer "no") -> "bridge"; tunnel=avalanche_protector -> "gallery"; covered=yes ohne
    tunnel-Tag (oder tunnel=no) -> "gallery" (überdachte Straße, z.B. die Galerien der Nuova strada del San
    Gottardo, siehe docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md); jedes andere
    tunnel=* (außer "no") -> "tunnel"; sonst "surface".
    """
    osm_tags = osm_tags or {}
    bridge = str(osm_tags.get("bridge", "")).strip().lower()
    if bridge and bridge != "no":
        return "bridge"
    tunnel = str(osm_tags.get("tunnel", "")).strip().lower()
    if tunnel == "avalanche_protector":
        return "gallery"
    covered = str(osm_tags.get("covered", "")).strip().lower()
    if covered == "yes" and tunnel in ("", "no"):
        return "gallery"
    if tunnel and tunnel != "no":
        return "tunnel"
    return "surface"
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/geometry/test_road_structures.py -v` → Expected: alle PASS
Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün

---

### Task 2: Übergänge in `plan_tunnels` erkennen

**Files:**
- Modify: `world_to_beamng/tunnels/tunnel_portal.py` (`plan_tunnels`, ab Zeile 21)
- Test: `tests/tunnels/test_tunnel_gallery_transition.py` (neu)

**Interfaces:**
- Consumes: Galerie-Eingaben im Format von `terrain_workflow._structure_items(…, "gallery")`:
  `{"id", "coords", "width", "floor_material", "osm_tags"}`.
- Produces: `plan_tunnels(tunnels, width_margin, segment_step, wing, flat_depth, length, cover, galleries=None,
  gallery_height=5.0, gallery_roof_thickness=0.5, gallery_floor_thickness=5.0, gallery_wall_thickness=5.0,
  transition_tol=0.5)`. Jedes Portal hat jetzt `"kind"`: `"open"` | `"gallery"`. Bei `"gallery"` zusätzlich
  `"gallery_half_width"`, `"gallery_height"`, mit vergrößertem `"half_width"`, `"top_z"` und `"bottom_z"`.

- [ ] **Step 1: Failing Tests schreiben** – `tests/tunnels/test_tunnel_gallery_transition.py`:

```python
"""Übergang Tunnel <-> Galerie (docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_portal import plan_tunnels

FLOOR = "asphalt_road_standard"
TUNNEL = [(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)]


def _tunnel(coords=TUNNEL, width=6.5):
    return {"id": 1, "coords": coords, "width": width, "floor_material": FLOOR}


def _gallery(coords, width=6.5, gallery_id=2):
    return {"id": gallery_id, "coords": coords, "width": width, "floor_material": FLOOR, "osm_tags": {"covered": "yes"}}


def _plans(tunnels, galleries=None):
    return plan_tunnels(
        tunnels, width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=1.0,
        galleries=galleries, gallery_height=5.0, gallery_roof_thickness=0.5, gallery_floor_thickness=5.0,
        gallery_wall_thickness=5.0, transition_tol=0.5,
    )


def test_portals_are_open_without_galleries():
    start, end = _plans([_tunnel()])[0]["portals"]
    assert start["kind"] == "open" and end["kind"] == "open"


def test_portal_on_a_gallery_endpoint_becomes_a_gallery_transition():
    start, end = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)])])[0]["portals"]

    assert start["kind"] == "gallery" and end["kind"] == "open"
    assert start["gallery_half_width"] == pytest.approx(3.25)
    assert start["gallery_height"] == pytest.approx(5.0)
    # Block deckt den ganzen Galerie-Querschnitt ab: Bergwand (3,25 + 5 m), Dach (5,5 m), Bodenquader (5 m)
    assert start["half_width"] == pytest.approx(max(start["radius"] + 2.0, 3.25 + 5.0))
    assert start["top_z"] >= 500.0 + 5.0 + 0.5 + 0.2 - 1e-9
    assert start["bottom_z"] == pytest.approx(500.0 - 5.0)


def test_gallery_digitised_away_from_the_tunnel_is_also_a_transition():
    start, _ = _plans([_tunnel()], [_gallery([(0.0, 0.0, 500.0), (-50.0, 0.0, 500.0)])])[0]["portals"]
    assert start["kind"] == "gallery"


def test_both_ends_can_be_gallery_transitions():
    galleries = [
        _gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], gallery_id=2),
        _gallery([(100.0, 0.0, 500.0), (150.0, 0.0, 500.0)], gallery_id=3),
    ]
    start, end = _plans([_tunnel()], galleries)[0]["portals"]
    assert start["kind"] == "gallery" and end["kind"] == "gallery"


def test_gallery_touching_the_portal_with_its_middle_is_no_transition():
    start, _ = _plans([_tunnel()], [_gallery([(0.0, -20.0, 500.0), (0.0, 20.0, 500.0)])])[0]["portals"]
    assert start["kind"] == "open"


def test_wide_gallery_widens_the_block():
    start, _ = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], width=9.75)])[0]["portals"]
    assert start["gallery_half_width"] == pytest.approx(4.875)
    assert start["half_width"] == pytest.approx(4.875 + 5.0)
```

- [ ] **Step 2: Laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tunnels/test_tunnel_gallery_transition.py -v`
Expected: FAIL mit `TypeError: plan_tunnels() got an unexpected keyword argument 'galleries'`

- [ ] **Step 3: Implementieren** – in `plan_tunnels` die Signatur erweitern:

```python
def plan_tunnels(
    tunnels: Sequence[Dict],
    width_margin: float,
    segment_step: float,
    wing: float,
    flat_depth: float,
    length: float,
    cover: float,
    galleries: Sequence[Dict] = None,
    gallery_height: float = 5.0,
    gallery_roof_thickness: float = 0.5,
    gallery_floor_thickness: float = 5.0,
    gallery_wall_thickness: float = 5.0,
    transition_tol: float = 0.5,
) -> List[Dict]:
```

In den Docstring unter `Args` ergänzen:

```
        galleries: Galerie-Eingaben (wie build_galleries()); ein Portal, das höchstens transition_tol von einem
            ENDPUNKT einer Galerie-Centerline liegt, ist ein Übergang Tunnel -> Galerie (portal["kind"] ==
            "gallery"): Stirnwand mit Galerie-Öffnung statt offenes Portal ins Gelände, Block auf den ganzen
            Galerie-Querschnitt vergrößert.
```

und im `Returns`-Text `"kind" ("open" | "gallery")` ergänzen. Direkt vor `plans = []` einfügen:

```python
    gallery_ends = []  # (x, y, Fahrbahnbreite) je Galerie-Endpunkt
    for gallery in galleries or []:
        coords = gallery["coords"]
        if len(coords) >= 2:
            for point in (coords[0], coords[-1]):
                gallery_ends.append((float(point[0]), float(point[1]), float(gallery["width"])))

    def gallery_width_at(x: float, y: float):
        for gx, gy, width in gallery_ends:
            if np.hypot(gx - x, gy - y) <= transition_tol:
                return width
        return None
```

Im Portal-Dict `"kind": "open",` ergänzen (nach `"open": True,`). Direkt nach `portals.append({...})`, noch in der
Schleife über `("start", 0, 1), ("end", -1, -2)`, einfügen:

```python
            gallery_width = gallery_width_at(*portals[-1]["xy"])
            if gallery_width is not None:
                portal = portals[-1]
                portal["kind"] = "gallery"
                portal["gallery_half_width"] = gallery_width / 2.0
                portal["gallery_height"] = gallery_height
                portal["half_width"] = max(portal["half_width"], gallery_width / 2.0 + gallery_wall_thickness)
                portal["top_z"] = max(portal["top_z"], floor_z + gallery_height + gallery_roof_thickness + 0.2)
                portal["bottom_z"] = min(portal["bottom_z"], floor_z - gallery_floor_thickness)
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tunnels -v` → Expected: alle PASS
Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün

---

### Task 3: Portalblock mit Galerie-Öffnung und Stufenfläche

**Files:**
- Modify: `world_to_beamng/tunnels/tunnel_portal.py` (`build_portal_block_mesh`, Stirnseiten-Teil ab `# Stirnseite:`)
- Test: `tests/tunnels/test_tunnel_gallery_transition.py`

**Interfaces:**
- Consumes: Portal aus Task 2 (`kind`, `gallery_half_width`, `gallery_height`).
- Produces: `build_portal_block_mesh(portal, material, arc_segments, tile_m)` (Signatur unverändert).

- [ ] **Step 1: Failing Tests anhängen** an `tests/tunnels/test_tunnel_gallery_transition.py`:

```python
from shapely.geometry import Polygon

from world_to_beamng.tunnels.tunnel_mesh import arc_cross_section
from world_to_beamng.tunnels.tunnel_portal import build_portal_block_mesh


def _transition_block():
    plans = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)])])
    portal = plans[0]["portals"][0]  # Portalebene x = 0, Achse +x ins Tunnelinnere
    return portal, build_portal_block_mesh(portal, "concrete", arc_segments=12)


def _plane_faces(mesh, normal_x):
    """(Dreiecke als (3, 3)-Array) in der Portalebene x = 0 mit Normale (normal_x, 0, 0)."""
    v, n = mesh["vertices"], mesh["normals"]
    result = []
    for face in mesh["faces"]["concrete"]:
        pts = v[face]
        if np.allclose(pts[:, 0], 0.0) and np.allclose(n[face[0]], [normal_x, 0.0, 0.0]):
            result.append(pts)
    return result


def _area(tri):
    (_, y0, z0), (_, y1, z1), (_, y2, z2) = tri
    return abs((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)) / 2.0


def test_transition_front_wall_leaves_exactly_the_gallery_opening_free():
    portal, mesh = _transition_block()
    front = _plane_faces(mesh, -1.0)  # zur Galerie
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    hw = portal["half_width"]
    top, bottom = portal["top_z"] - 500.0, portal["bottom_z"] - 500.0

    assert sum(_area(t) for t in front) == pytest.approx(2 * hw * (top - bottom) - 2 * g * gh)
    for tri in front:  # kein Stirn-Dreieck liegt in der Öffnung
        cy, cz = tri[:, 1].mean(), tri[:, 2].mean() - 500.0
        assert not (abs(cy) < g and 0.0 < cz < gh)


def test_transition_step_faces_into_the_tunnel_between_arc_and_opening():
    portal, mesh = _transition_block()
    step = _plane_faces(mesh, 1.0)  # ins Tunnelinnere
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    ring = Polygon(arc_cross_section(portal["radius"], 12))

    assert step, "keine Stufenfläche"
    assert sum(_area(t) for t in step) == pytest.approx(ring.area - 2 * g * gh, rel=1e-6)


def test_open_portal_keeps_the_round_opening():
    plans = _plans([_tunnel()])
    mesh = build_portal_block_mesh(plans[0]["portals"][0], "concrete", arc_segments=12)
    assert _plane_faces(mesh, 1.0) == []  # keine Stufenfläche beim offenen Portal
```

- [ ] **Step 2: Laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tunnels/test_tunnel_gallery_transition.py -v`
Expected: FAIL in den beiden `test_transition_*` (Flächensumme stimmt nicht / keine Stufenfläche)

- [ ] **Step 3: Implementieren**

Zuerst eine Hilfsfunktion **vor** `build_portal_block_mesh` einfügen:

```python
def transition_step_triangles(radius: float, arc_segments: int, half_opening: float, opening_height: float):
    """Dreiecke (across, height) der Stufenfläche eines Übergangs-Portals: Röhrenquerschnitt (240°-Bogen über der
    Bodensehne) minus die rechteckige Galerie-Öffnung (±half_opening x 0..opening_height)."""
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon, box

    step = Polygon(arc_cross_section(radius, arc_segments)).difference(box(-half_opening, 0.0, half_opening, opening_height))
    return [list(tri.exterior.coords)[:3] for tri in constrained_delaunay_triangles(step).geoms]
```

Dann in `build_portal_block_mesh` den Block ab `# Stirnseite: Ring zwischen Öffnung und Rechteck-Rand` bis
einschließlich `front([(half_width, 0.0), (-half_width, 0.0), (-half_width, bottom), (half_width, bottom)])` in einen
`else`-Zweig verschieben und davor den Übergangs-Zweig setzen:

```python
    if portal.get("kind") == "gallery":
        # Übergang Tunnel -> Galerie: geschlossene Stirnwand mit rechteckiger Öffnung in Galeriegröße (Dach, Bergwand
        # und Sockel der Galerie schließen hier bündig an) ...
        g, gh = portal["gallery_half_width"], portal["gallery_height"]
        front([(-half_width, bottom), (-g, bottom), (-g, top), (-half_width, top)])
        front([(g, bottom), (half_width, bottom), (half_width, top), (g, top)])
        front([(-g, gh), (g, gh), (g, top), (-g, top)])
        front([(-g, bottom), (g, bottom), (g, 0.0), (-g, 0.0)])
        # ... und von innen der Querschnittssprung Röhrenbogen -> Rechteck, zur Röhre hin gerichtet
        for tri in transition_step_triangles(radius, arc_segments, g, gh):
            builder.triangle([world(0.0, c, h) for c, h in tri], [[c / tile_m, h / tile_m] for c, h in tri], [ux, uy, 0.0])
    else:
        # Stirnseite: Ring zwischen Öffnung und Rechteck-Rand
        arc = arc_cross_section(radius, arc_segments)
        ...  # (bestehender Code unverändert, eine Ebene tiefer eingerückt)
        # Streifen unter Bodenhöhe (falls das Gelände vor dem Portal tiefer liegt)
        front([(half_width, 0.0), (-half_width, 0.0), (-half_width, bottom), (half_width, bottom)])
```

Der Docstring von `build_portal_block_mesh` bekommt als zweiten Absatz:

```
    Übergangs-Portal (portal["kind"] == "gallery", siehe plan_tunnels()): Die Stirnseite ist eine geschlossene Wand
    mit rechteckiger Galerie-Öffnung; dazu eine zur Röhre gerichtete Stufenfläche zwischen Bogen und Rechteck.
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tunnels -v` → Expected: alle PASS (auch die bestehenden Portal-Tests in `test_tunnel_mesh.py`)
Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün

---

### Task 4: Galerie ohne Stirnfläche am Übergang

**Files:**
- Modify: `world_to_beamng/tunnels/gallery_mesh.py` (`build_gallery_mesh` Signatur + Zeilen 262-263, `build_galleries`)
- Test: `tests/tunnels/test_gallery_mesh.py`

**Interfaces:**
- Produces:
  - `build_gallery_mesh(..., open_side=None, cap_start=True, cap_end=True)`
  - `build_galleries(galleries, ground_at, roof_material, ..., transition_points=None, transition_tol=0.5)`,
    wobei `transition_points` eine Folge von (x, y) der Übergangs-Portale ist.

- [ ] **Step 1: Failing Tests anhängen** an `tests/tunnels/test_gallery_mesh.py`:

```python
def _cap_faces(mesh, x, normal_x):
    v, n = mesh["vertices"], mesh["normals"]
    faces = [f for faces in mesh["faces"].values() for f in faces]
    return [f for f in faces if np.allclose(v[f][:, 0], x) and np.allclose(n[f[0]], [normal_x, 0.0, 0.0])]


def test_end_caps_can_be_left_out():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    kwargs = dict(width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    capped = build_gallery_mesh(_straight_coords(), **kwargs)
    uncapped = build_gallery_mesh(_straight_coords(), cap_start=False, **kwargs)

    assert _cap_faces(capped, 0.0, -1.0) and not _cap_faces(uncapped, 0.0, -1.0)
    assert _cap_faces(uncapped, 60.0, 1.0)  # anderes Ende bleibt verschlossen


@pytest.mark.parametrize("reverse", [False, True])
def test_transition_removes_only_the_cap_at_the_transition_end(reverse):
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    coords = _straight_coords()
    if reverse:
        coords = coords[::-1]
    gallery = {"id": 7, "coords": coords, "width": 8.0, "floor_material": FLOOR, "osm_tags": {}}

    mesh = build_galleries([gallery], ground_at, ROOF, transition_points=[(0.0, 0.0)], transition_tol=0.5)[0]

    assert not _cap_faces(mesh, 0.0, -1.0)  # am Übergang (x = 0) keine Stirnfläche
    assert _cap_faces(mesh, 60.0, 1.0)  # freies Ende verschlossen
```

- [ ] **Step 2: Laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tunnels/test_gallery_mesh.py -v`
Expected: FAIL mit `TypeError: build_gallery_mesh() got an unexpected keyword argument 'cap_start'`

- [ ] **Step 3: Implementieren**

- In `build_gallery_mesh` nach `open_side: Optional[str] = None,` einfügen:

  ```python
      cap_start: bool = True,
      cap_end: bool = True,
  ```

  Im Docstring unter `Args` ergänzen:

  ```
          cap_start, cap_end: Stirnfläche am Anfang/Ende bauen - False an einem Übergang zu einem Tunnel-Portal
              (dessen Stirnwand deckt den Galerie-Querschnitt ab; eine eigene Stirnfläche läge in derselben Ebene
              und flackerte).
  ```

- Die beiden `_add_end_caps`-Aufrufe (Zeilen 262-263) ersetzen durch:

  ```python
      if cap_start:
          _add_end_caps(roof_builder, 0, xy[0] - xy[1], *end_cap_args)
      if cap_end:
          _add_end_caps(roof_builder, len(points) - 1, xy[-1] - xy[-2], *end_cap_args)
  ```

- `build_galleries` bekommt nach `curb_width: float = 0.4,` die Parameter
  `transition_points: Sequence[Tuple[float, float]] = (),` und `transition_tol: float = 0.5,`. Den Docstring
  ergänzen:

  ```
      transition_points: (x, y) der Übergangs-Portale (tunnel_portal.plan_tunnels(), portal["kind"] == "gallery") -
          ein Galerie-Ende, das höchstens transition_tol davon liegt, bekommt keine Stirnfläche.
  ```

  In der Schleife vor `mesh = build_gallery_mesh(` einfügen:

  ```python
          def at_transition(point) -> bool:
              return any(np.hypot(point[0] - tx, point[1] - ty) <= transition_tol for tx, ty in transition_points)
  ```

  und an den `build_gallery_mesh`-Aufruf anhängen:
  `cap_start=not at_transition(coords[0]), cap_end=not at_transition(coords[-1]),`

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tunnels -v` → Expected: alle PASS
Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün

---

### Task 5: Gelände – Übergangs-Portale immer offen, kurze Lücken überdecken

**Files:**
- Modify: `world_to_beamng/terrain/tunnel_terrain.py` (`_raise_cover`, `shape_terrain_for_tunnels`, Modul-Docstring)
- Test: `tests/terrain/test_tunnel_terrain.py`

**Interfaces:**
- Consumes: Portal-`"kind"` aus Task 2.
- Produces: `shape_terrain_for_tunnels(heights, origin_x, origin_y, square_size, plans, cover, cover_slope,
  protected=None, cover_gap_max=0.0)`. Der Default 0.0 bedeutet keine Lückenfüllung; so bleiben bestehende Aufrufer
  gleich, und der Workflow übergibt den Wert aus der Config.

- [ ] **Step 1: Failing Tests anhängen** an `tests/terrain/test_tunnel_terrain.py`:

```python
def _setup_gap(gap, size_x=160):
    # Tunnel entlang y=60 von x=30 bis x=130; hinter dem Start-Portal `gap` Meter flach (Röhre steckt dort laut
    # Höhenmodell nicht im Berg), danach Berg auf 105 m; vor beiden Portalen Straßenniveau.
    tunnel = {"id": 1, "coords": [(30.0, 60.0, FLOOR), (130.0, 60.0, FLOOR)], "width": 7.0, "floor_material": "f"}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=COVER)
    heights = np.full((120, size_x), 105.0)
    heights[:, : 30 + gap + 1] = FLOOR
    heights[:, 131:] = FLOOR
    return plans, heights


def test_short_cover_gap_behind_the_portal_is_covered_without_a_step():
    plans, heights = _setup_gap(8)
    top = FLOOR + plans[0]["crown"] + COVER

    result, _ = shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, cover_slope=1.5, cover_gap_max=25.0)

    # hinter der Portal-Zone bis in den Berg durchgehend überdeckt: keine Geländekante durch die Röhre
    assert all(result[60, x] >= top - 1e-9 for x in range(33, 50))


def test_cover_gap_longer_than_the_limit_stays_open():
    plans, heights = _setup_gap(40)

    result, _ = shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, cover_slope=1.5, cover_gap_max=25.0)

    assert result[60, 50] == pytest.approx(FLOOR)  # 20 m hinter dem Portal, mitten in der 40-m-Lücke


def test_without_gap_limit_the_old_behaviour_stays():
    plans, heights = _setup_gap(8)

    result, _ = shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, cover_slope=1.5)

    assert result[60, 36] == pytest.approx(FLOOR)


def test_gallery_transition_is_a_portal_even_with_mountain_in_front():
    tunnel = {"id": 1, "coords": [(30.0, 60.0, FLOOR), (90.0, 60.0, FLOOR)], "width": 7.0, "floor_material": "f"}
    gallery = {"id": 2, "coords": [(0.0, 60.0, FLOOR), (30.0, 60.0, FLOOR)], "width": 6.5, "floor_material": "f", "osm_tags": {}}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=COVER, galleries=[gallery])
    heights = np.full((120, 120), 150.0)  # auch vor dem Portal Berg - ein offenes Portal gäbe es hier nicht

    result, holes = _shape(plans, heights)

    assert plans[0]["portals"][0]["open"] is True
    assert result[60, 31] == pytest.approx(FLOOR - 0.05)
    assert holes[60, 31]
```

- [ ] **Step 2: Laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/terrain/test_tunnel_terrain.py -v`
Expected: FAIL. Die Lücken-Tests scheitern mit `TypeError: ... unexpected keyword argument 'cover_gap_max'`, der
Übergangs-Test mit `assert False is True`.

- [ ] **Step 3: Implementieren**

- Neue Hilfsfunktion **vor** `_raise_cover`:

  ```python
  def _fill_portal_gaps(buried: np.ndarray, s: np.ndarray, start_open: bool, end_open: bool, max_gap: float) -> np.ndarray:
      """`buried` plus die nicht eingegrabenen Strecken, die an einem offenen Portal beginnen und höchstens max_gap
      lang sind, bevor die Röhre im Gelände steckt. Ohne diese Füllung fiele das Gelände hinter der Portal-
      Überdeckung wieder auf Fahrbahnhöhe - die Geländefläche liefe als Erdwand quer durch die Röhre (Nordportal
      Tunnel Fieud: ~9 m flach hinter dem Portal). Längere Strecken bleiben offen (sonst Dämme)."""
      filled = buried.copy()
      if max_gap <= 0.0 or not buried.any():
          return filled
      first = int(np.argmax(buried))
      if start_open and first > 0 and s[first] - s[0] <= max_gap:
          filled[:first] = True
      last = len(buried) - 1 - int(np.argmax(buried[::-1]))
      if end_open and last < len(buried) - 1 and s[-1] - s[last] <= max_gap:
          filled[last + 1 :] = True
      return filled
  ```

- `_raise_cover` bekommt den Parameter `cover_gap_max: float = 0.0`
  (`def _raise_cover(heights, origin_x, origin_y, square_size, plan, cover, cover_slope, protected, cover_gap_max=0.0)`).
  Nach der Zeile `start_portal, end_portal = plan["portals"]` einfügen:

  ```python
      buried = _fill_portal_gaps(buried, s, start_portal["open"], end_portal["open"], cover_gap_max)
  ```

- `shape_terrain_for_tunnels` bekommt `cover_gap_max: float = 0.0` nach `protected=None`. Den Docstring ergänzen:

  ```
          cover_gap_max: so lange Lücke zwischen offenem Portal und eingegrabener Röhre wird noch überdeckt (0 = aus)
  ```

  Die Schleifen so anpassen:

  ```python
      for plan in plans:
          for portal in plan["portals"]:
              # Übergang in eine Galerie: davor liegt immer ein Bauwerk - immer ein Portal
              portal["open"] = portal.get("kind") == "gallery" or _portal_is_open(heights, origin_x, origin_y, square_size, portal)
      for plan in plans:
          _raise_cover(result, origin_x, origin_y, square_size, plan, cover, cover_slope, protected, cover_gap_max)
  ```

- Im Modul-Docstring nach Punkt 2 ergänzen:

  ```
  3. Lücken: liegt die Röhre hinter einem offenen Portal erst nach einer kurzen Strecke (<= cover_gap_max) im
     Gelände, wird diese Strecke mit überdeckt - sonst fiele das Gelände hinter dem Portalblock ab und schnitte
     als Erdwand durch die Röhre. Übergänge in eine Galerie (portal["kind"] == "gallery") sind immer Portale.
  ```

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/terrain/test_tunnel_terrain.py -v` → Expected: alle PASS (auch die bestehenden, insbesondere `test_no_cover_dam_where_the_tube_is_not_at_least_half_in_the_ground`)
Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün

---

### Task 6: Workflow verdrahten

**Files:**
- Modify: `world_to_beamng/config.py` (nach `TUNNEL_PORTAL_LENGTH`)
- Modify: `world_to_beamng/workflow/terrain_workflow.py` (neuer Modul-Helfer nach `_structure_items`; `plan_tunnels`-Aufruf ab Zeile 480; `shape_terrain_for_tunnels`-Aufruf; `_build_tunnels` ab Zeile 936)
- Test: `tests/workflow/test_terrain_workflow_tunnels.py`

**Interfaces:**
- Consumes: Tasks 2, 4 und 5.
- Produces: `_plan_tunnels(structure_road_polygons) -> List[Dict]` (Modul-Helfer in `terrain_workflow.py`).

- [ ] **Step 1: Failing Tests anhängen** an `tests/workflow/test_terrain_workflow_tunnels.py`:

```python
from types import SimpleNamespace

from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow, _plan_tunnels


def _structure(road_id, coords, **tags):
    from world_to_beamng.geometry.road_structures import classify_structure

    tags = {"highway": "primary", "lanes": "2", **tags}
    return {"road_id": road_id, "trimmed_centerline": np.array(coords, dtype=float), "osm_tags": tags,
            "structure_type": classify_structure(tags)}


def _tunnel_and_gallery():
    return [
        _structure(1, [(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)], tunnel="yes"),
        _structure(2, [(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], covered="yes"),
    ]


def test_plan_tunnels_marks_the_portal_at_a_covered_gallery_as_transition():
    start, end = _plan_tunnels(_tunnel_and_gallery())[0]["portals"]
    assert start["kind"] == "gallery" and end["kind"] == "open"


def test_gallery_at_a_transition_gets_no_end_cap_in_the_workflow():
    roads = _tunnel_and_gallery()
    plans = _plan_tunnels(roads)
    heights = np.full((300, 300), 500.0)

    meshes = TerrainWorkflow._build_tunnels(SimpleNamespace(), roads, plans, heights, -150.0, -150.0)

    gallery = next(m for m in meshes if m["id"] == "gallery_2")
    v, n = gallery["vertices"], gallery["normals"]
    faces = [f for fs in gallery["faces"].values() for f in fs]
    at_portal = [f for f in faces if np.allclose(v[f][:, 0], 0.0) and np.allclose(n[f[0]], [1.0, 0.0, 0.0])]
    assert at_portal == []  # die Portalwand schließt die Galerie, keine eigene Stirnfläche
    assert any(m["id"] == "tunnel_1_portal_start" for m in meshes)
```

(Falls `np` in der Testdatei noch nicht importiert ist, `import numpy as np` oben ergänzen.)

- [ ] **Step 2: Laufen lassen, muss fehlschlagen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/workflow/test_terrain_workflow_tunnels.py -v`
Expected: FAIL mit `ImportError: cannot import name '_plan_tunnels'`

- [ ] **Step 3: Implementieren**

- `config.py`, direkt nach `TUNNEL_PORTAL_LENGTH = 3.5 …`:

  ```python
  # Übergang Tunnel -> Galerie (docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md): ein
  # Tunnel-Portal, das höchstens so weit von einem Galerie-Endpunkt liegt, bekommt eine Stirnwand mit Galerie-Öffnung.
  TUNNEL_TRANSITION_ENDPOINT_TOL = 0.5  # in Metern
  TUNNEL_COVER_GAP_MAX = 25.0  # so lange Lücke zwischen offenem Portal und eingegrabener Röhre wird überdeckt, in Metern
  ```

- `terrain_workflow.py`, neuer Modul-Helfer direkt nach `_structure_items()`:

  ```python
  def _plan_tunnels(structure_road_polygons: List[Dict]) -> List[Dict]:
      """Tunnel-Pläne (tunnels/tunnel_portal.py::plan_tunnels()) mit den Galerien als möglichen Übergängen."""
      from ..tunnels.tunnel_portal import plan_tunnels

      return plan_tunnels(
          _structure_items(structure_road_polygons, "tunnel"),
          width_margin=config.TUNNEL_WIDTH_MARGIN,
          segment_step=config.TUNNEL_SEGMENT_STEP,
          wing=config.TUNNEL_PORTAL_WING,
          flat_depth=config.TUNNEL_PORTAL_FLAT_DEPTH,
          length=config.TUNNEL_PORTAL_LENGTH,
          cover=config.TUNNEL_COVER,
          galleries=_structure_items(structure_road_polygons, "gallery"),
          gallery_height=config.GALLERY_HEIGHT,
          gallery_roof_thickness=config.GALLERY_ROOF_THICKNESS,
          gallery_floor_thickness=config.GALLERY_FLOOR_THICKNESS,
          gallery_wall_thickness=config.GALLERY_WALL_THICKNESS,
          transition_tol=config.TUNNEL_TRANSITION_ENDPOINT_TOL,
      )
  ```

- Im Tunnel-Block (ab `from ..tunnels.tunnel_portal import plan_tunnels`) den Import und den Aufruf
  `tunnel_plans = plan_tunnels(…)` ersetzen durch `tunnel_plans = _plan_tunnels(structure_road_polygons)`. Den Import
  von `plan_tunnels` dort entfernen, `shape_terrain_for_tunnels` bleibt importiert.
- Im Aufruf von `shape_terrain_for_tunnels(…)` nach `protected=protected,` ergänzen:
  `cover_gap_max=config.TUNNEL_COVER_GAP_MAX,`
- In `_build_tunnels` vor `gallery_meshes = build_galleries(` einfügen:

  ```python
          # Übergangs-Portale: dort schließt die Portalwand die Galerie (keine eigene Stirnfläche, siehe gallery_mesh.py)
          transition_points = [p["xy"] for plan in tunnel_plans for p in plan["portals"] if p.get("kind") == "gallery"]
  ```

  und an den `build_galleries(…)`-Aufruf nach `curb_width=config.GALLERY_CURB_WIDTH,` anhängen:

  ```python
              transition_points=transition_points,
              transition_tol=config.TUNNEL_TRANSITION_ENDPOINT_TOL,
  ```

- [ ] **Step 4: Tests laufen lassen**

Run: `.\.venv\Scripts\python.exe -m pytest tests/workflow/test_terrain_workflow_tunnels.py -v` → Expected: alle PASS
Run: `.\.venv\Scripts\python.exe -m pytest -q` → Expected: alles grün

---

### Task 7: Prüfung am echten Export, Doku, Spiel

Kein neuer Produktionscode.

- [ ] **Step 1: OSM-Stand prüfen.** In `cache/osm_all_4113e78937c1.json` darf Way 430132545 kein `tunnel`-Tag mehr
  haben. Falls doch: Datei löschen und den Export neu laufen lassen, bis Overpass die Korrektur liefert.

- [ ] **Step 2: Export**

Run: `.\.venv\Scripts\python.exe world_to_beamng.py`
Expected: Lauf ohne Fehler.

- [ ] **Step 3: Ergebnis prüfen** (Skript im Scratchpad; Heightmap mit `terrain/ter_writer.read_ter`, Höhe =
  u16 · `maxHeight` / 65536 + `position.z` des `TerrainBlock`)
  1. In `tunnels.dae` gibt es die Nodes `tunnel_430132546_portal_start`, `tunnel_430132546_portal_end` und
     `tunnel_430132547_portal_end`. Deren Stirnseite hat keine Bogenöffnung, sondern eine Rechteck-Öffnung, erkennbar
     an Dreiecken mit Normale in Achsrichtung (Stufenfläche).
  2. Es gibt `gallery_746194686…` und `gallery_430132545…`, aber keine DecalRoads `road_746194686*`/`road_430132545*`
     und keine `marking_746194686*`.
  3. Gelände-Profil entlang der Achse am Nordportal von Tunnel Fieud (Portalpunkt ≈ (−601,7; 338,9), Achse aus dem
     letzten Tunnelsegment): Zwischen 3 m und 15 m hinter der Portalebene liegt das Gelände durchgehend mindestens auf
     Kronenhöhe über dem Röhrenboden, ohne Abfall auf Fahrbahnhöhe.
  4. `beamng.log` nach dem Laden: keine neuen `|E|`-Zeilen.

- [ ] **Step 4: Doku.** In `docs/OSM_ROAD_ANALYSIS.md` unter „3.1 Straßentypen / Filter“ die Zeile zu
  `covered=yes ohne tunnel` auf den neuen Stand bringen: „wird als Galerie gebaut, Übergänge Tunnel ↔ Galerie mit
  Portalwand, siehe Spec 2026-09-24-tunnel-gallery-transition-design.md“.

- [ ] **Step 5: Im Spiel mit dem Nutzer.** Durchfahrt Nord → Süd: Galerie (49 m) → Tunnel Banchi → Galerie (727 m) →
  Tunnel Fieud. Dabei prüfen:
  - Portalwände sitzen bündig an Dach und Bergwand.
  - Es flackert nichts.
  - Keine Erdwand in der Röhre.
  - Die Talseite der Galerien ist offen, nach den neuen `avalanche_protector:right=open`-Tags oder dem
    Geländevergleich.
