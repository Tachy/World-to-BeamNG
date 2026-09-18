"""Tests für world_to_beamng.geometry.polygon.clip_road_polygons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.polygon import clip_road_polygons


def _road(road_id, coords):
    return {"id": road_id, "coords": coords, "name": f"road_{road_id}", "osm_tags": {"highway": "track"}}


def test_clip_road_polygons_keeps_single_contiguous_road_unchanged():
    # Alle Punkte liegen innerhalb der Clip-Box -> ein Abschnitt, ID bleibt gleich.
    # Punktabstand bewusst << config.GRID_SPACING, damit die Segment-Unterteilung
    # in clip_road_polygons() (max_seg = config.GRID_SPACING) unabhängig vom
    # konfigurierten Grid-Spacing keine Zwischenpunkte einfügt.
    coords = [(0.0, 0.0, 100.0), (0.1, 0.0, 100.1), (0.2, 0.0, 100.2)]
    grid_bounds_local = (-10.0, 10.0, -10.0, 10.0)

    result = clip_road_polygons([_road(42, coords)], grid_bounds_local, margin=0.0)

    assert len(result) == 1
    assert result[0]["id"] == 42
    assert result[0]["coords"] == coords


def test_clip_road_polygons_splits_at_removed_gap_instead_of_bridging():
    """Regression: eine Strasse, die das Tile verlaesst und an ganz anderer
    Stelle wieder hineinragt, darf NICHT zu einer einzigen, durchgehenden
    Centerline mit einer künstlichen "Teleport"-Gerade über die Lücke
    zusammengefasst werden (siehe Root-Cause-Analyse der harten Böschungs-
    Klippe bei Weg 77512819: zwei weit auseinanderliegende, aber beide
    innerhalb der Clip-Box liegende Punktgruppen wurden zuvor direkt
    verbunden und per Segment-Unterteilung mit einer geraden, falschen
    Z-Interpolation über hunderte Meter gefüllt).
    """
    grid_bounds_local = (-10.0, 10.0, -10.0, 10.0)

    # Punktabstand innerhalb der Cluster bewusst << config.GRID_SPACING (siehe
    # Kommentar in test_clip_road_polygons_keeps_single_contiguous_road_unchanged).
    coords = [
        # Cluster A: innerhalb der Box
        (0.0, 0.0, 100.0),
        (0.1, 0.0, 100.1),
        # Weit ausserhalb der Box (wird entfernt)
        (500.0, 500.0, 50.0),
        (600.0, 600.0, 20.0),
        # Cluster B: wieder innerhalb der Box, aber geometrisch weit von Cluster A entfernt
        (-5.0, -5.0, 300.0),
        (-5.1, -5.0, 300.1),
    ]

    result = clip_road_polygons([_road(77512819, coords)], grid_bounds_local, margin=0.0)

    # Erwartet: ZWEI getrennte Strassen-Abschnitte, keine Brücke zwischen ihnen -
    # jeder Abschnitt enthält EXAKT die Punkte seines eigenen Clusters, keine
    # künstlich interpolierten Zwischenpunkte (die entstünden nur, wenn beide
    # Cluster fälschlich zu einer durchgehenden Centerline verbunden würden).
    assert len(result) == 2
    coord_lists = [road["coords"] for road in result]
    assert [(0.0, 0.0, 100.0), (0.1, 0.0, 100.1)] in coord_lists
    assert [(-5.0, -5.0, 300.0), (-5.1, -5.0, 300.1)] in coord_lists

    # IDs der beiden Abschnitte müssen eindeutig sein
    assert result[0]["id"] != result[1]["id"]


def test_clip_road_polygons_drops_run_with_single_surviving_point():
    grid_bounds_local = (-10.0, 10.0, -10.0, 10.0)
    coords = [
        (0.0, 0.0, 100.0),
        (0.1, 0.0, 100.1),
        (500.0, 500.0, 50.0),  # entfernt -> beendet ersten Abschnitt
        (600.0, 600.0, 20.0),  # entfernt
        (700.0, 700.0, 10.0),  # entfernt
        (5.0, 5.0, 200.0),  # einzelner Punkt -> Abschnitt mit nur 1 Punkt, wird verworfen
        (800.0, 800.0, 5.0),  # entfernt
    ]

    result = clip_road_polygons([_road(1, coords)], grid_bounds_local, margin=0.0)

    assert len(result) == 1
    assert result[0]["coords"] == [(0.0, 0.0, 100.0), (0.1, 0.0, 100.1)]
