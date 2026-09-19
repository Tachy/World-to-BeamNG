"""
Tests für den Vier-Bilder-Modus: ein Luftbild pro DGM1-Kachel (terrain/photo_tiles.py).

Die bestehende Logik (Layer-Map mit Foto = Index 0, Landnutzung nach Name) bleibt unverändert; erst am Ende
wird die Layer-Map pro Kachel in physische Materialien aufgeteilt (aerial_photo_<k>, mat_grass_t<k>, ...).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.terrain.photo_tiles import build_tile_index_map, expand_layers_per_tile, photo_tile_specs

HOLE = 255


def _tile(easting, northing, size=2000):
    return {"bbox_utm": (easting, easting + size, northing, northing + size)}


FOUR = [_tile(401000, 5298000), _tile(399000, 5296000), _tile(401000, 5296000), _tile(399000, 5298000)]
OFFSET = (401000.0, 5298000.0)


def test_specs_have_one_photo_per_tile_in_a_stable_order_with_local_bounds():
    specs = photo_tile_specs(FOUR, OFFSET)

    assert [s["name"] for s in specs] == ["aerial_photo_0", "aerial_photo_1", "aerial_photo_2", "aerial_photo_3"]
    # Reihenfolge: Süden zuerst, dann Westen zuerst - unabhängig von der Eingabereihenfolge
    assert [s["bounds"] for s in specs] == [
        (-2000.0, 0.0, -2000.0, 0.0),
        (0.0, 2000.0, -2000.0, 0.0),
        (-2000.0, 0.0, 0.0, 2000.0),
        (0.0, 2000.0, 0.0, 2000.0),
    ]
    assert photo_tile_specs(list(reversed(FOUR)), OFFSET) == specs


def test_tile_index_map_assigns_every_cell_to_the_tile_it_lies_in():
    specs = photo_tile_specs(FOUR, OFFSET)
    # 1-m-Zellen ab (-2000,-2000); die Heightmap ist auf 4096 aufgefüllt, Daten reichen bis x/y = +2000
    index_map = build_tile_index_map(4096, -2000.0, -2000.0, 1.0, specs)

    assert index_map[10, 10] == 0  # SW-Kachel
    assert index_map[10, 3000] == 1  # SO (Zeile = y, Spalte = x)
    assert index_map[3000, 10] == 2  # NW
    assert index_map[3000, 3000] == 3  # NO
    # Kachelgrenze: Zelle x = 0 gehört schon zur östlichen Kachel, x = -1 noch zur westlichen
    assert index_map[10, 1999] == 0 and index_map[10, 2000] == 1
    assert index_map[1999, 10] == 0 and index_map[2000, 10] == 2


def test_the_outermost_data_cells_and_the_padded_rim_belong_to_the_edge_tiles():
    specs = photo_tile_specs(FOUR, OFFSET)
    index_map = build_tile_index_map(4096, -2000.0, -2000.0, 1.0, specs)

    assert index_map[4000, 4000] == 3  # letzte Datenzelle (x = y = +2000)
    assert index_map[4095, 4095] == 3  # aufgefüllter Rand: nächste Kachel
    assert index_map[4095, 10] == 2 and index_map[10, 4095] == 1
    assert index_map.min() >= 0  # nichts bleibt ohne Kachel


def test_single_tile_maps_everything_to_it():
    specs = photo_tile_specs([_tile(401000, 5298000)], (402000.0, 5299000.0))

    assert build_tile_index_map(2048, -1000.0, -1000.0, 1.0, specs).max() == 0


# --- Layer-Map pro Kachel aufteilen ----------------------------------------------------------------


def _small_case():
    # 4x4-Zellen-Terrain, 2x2 Kacheln zu je 2x2 Zellen; Foto (0), Wiese (1), Wald (2)
    specs = [
        {"index": 0, "name": "aerial_photo_0", "bounds": (0, 2, 0, 2)},
        {"index": 1, "name": "aerial_photo_1", "bounds": (2, 4, 0, 2)},
        {"index": 2, "name": "aerial_photo_2", "bounds": (0, 2, 2, 4)},
        {"index": 3, "name": "aerial_photo_3", "bounds": (2, 4, 2, 4)},
    ]
    index_map = build_tile_index_map(4, 0.0, 0.0, 1.0, specs)
    layer_map = np.array(
        [
            [0, 1, 1, 0],
            [2, 2, 1, HOLE],
            [0, 0, 2, 2],
            [1, HOLE, 0, 1],
        ],
        dtype=np.uint8,
    )
    return specs, index_map, layer_map, ["aerial_photo", "mat_grass", "mat_forest"]


def test_photo_cells_get_the_photo_of_their_own_tile():
    specs, index_map, layer_map, names = _small_case()

    result, new_names, _, _ = expand_layers_per_tile(layer_map, names, index_map, [s["name"] for s in specs])

    photo = lambda r, c: new_names[result[r, c]]
    assert photo(0, 0) == "aerial_photo_0"
    assert photo(0, 3) == "aerial_photo_1"
    assert photo(2, 0) == "aerial_photo_2"
    assert photo(3, 2) == "aerial_photo_3"


def test_landuse_layers_become_one_variant_per_tile_and_keep_their_meaning():
    specs, index_map, layer_map, names = _small_case()

    result, new_names, variants, parents = expand_layers_per_tile(layer_map, names, index_map, [s["name"] for s in specs])

    assert new_names[result[0, 1]] == "mat_grass_t0"  # Wiese in Kachel 0
    assert new_names[result[1, 2]] == "mat_grass_t1"  # Wiese in Kachel 1
    assert new_names[result[1, 0]] == "mat_forest_t0"
    assert new_names[result[2, 3]] == "mat_forest_t3"
    assert variants["mat_grass"] == ["mat_grass_t0", "mat_grass_t1", "mat_grass_t2", "mat_grass_t3"]
    assert variants["mat_forest"] == ["mat_forest_t0", "mat_forest_t3"]  # nur wo es Wald gibt
    # jede Variante kennt ihre Ausgangs-Schicht und das Foto ihrer Kachel (für die Basisfarbe)
    assert parents["mat_grass_t1"] == ("mat_grass", "aerial_photo_1")
    assert parents["mat_forest_t3"] == ("mat_forest", "aerial_photo_3")


def test_holes_stay_holes_and_no_layer_is_lost():
    specs, index_map, layer_map, names = _small_case()

    result, new_names, _, _ = expand_layers_per_tile(layer_map, names, index_map, [s["name"] for s in specs])

    assert ((layer_map == HOLE) == (result == HOLE)).all()
    logical = lambda n: n.rsplit("_t", 1)[0] if "_t" in n and not n.startswith("aerial_photo") else ("aerial_photo" if n.startswith("aerial_photo") else n)
    for r in range(4):
        for c in range(4):
            if layer_map[r, c] != HOLE:
                assert logical(new_names[result[r, c]]) == names[layer_map[r, c]]


def test_photo_materials_come_first_and_only_used_variants_exist():
    specs, index_map, layer_map, names = _small_case()

    _, new_names, _, _ = expand_layers_per_tile(layer_map, names, index_map, [s["name"] for s in specs])

    assert new_names[:4] == ["aerial_photo_0", "aerial_photo_1", "aerial_photo_2", "aerial_photo_3"]
    assert "mat_forest_t1" not in new_names and "mat_forest_t2" not in new_names  # dort gibt es keinen Wald
    assert len(new_names) == len(set(new_names))


def test_more_than_254_materials_are_rejected():
    layer_map = np.arange(300, dtype=np.uint16).reshape(20, 15) % 250
    names = [f"m{i}" for i in range(250)]
    index_map = np.zeros(layer_map.shape, dtype=np.int16)
    index_map[:, 8:] = 1

    with pytest.raises(ValueError, match="254"):
        expand_layers_per_tile(layer_map.astype(np.uint8), names, index_map, ["p0", "p1"])


# --- Zusammenfassende Funktion für process_tile ---------------------------------------------------


def test_split_layers_by_tile_returns_everything_the_export_needs():
    from world_to_beamng.terrain.photo_tiles import split_layers_by_tile

    layer_map = np.zeros((4096, 4096), dtype=np.uint8)
    layer_map[100:200, 100:200] = 1  # Wiese in der SW-Kachel
    layer_map[3000:3100, 3000:3100] = 1  # Wiese in der NO-Kachel
    layer_map[:, 4001:] = HOLE  # aufgefüllter Rand

    result = split_layers_by_tile(layer_map, ["aerial_photo", "mat_grass"], FOUR, OFFSET, -2000.0, -2000.0, 1.0)

    assert result["photo_tile_names"] == ["aerial_photo_0", "aerial_photo_1", "aerial_photo_2", "aerial_photo_3"]
    assert result["photo_extents"] == {f"aerial_photo_{k}": 2000.0 for k in range(4)}
    assert result["layer_variants"] == {"mat_grass": ["mat_grass_t0", "mat_grass_t3"]}
    assert result["variant_parents"]["mat_grass_t3"] == ("mat_grass", "aerial_photo_3")
    new_map, names = result["layer_map"], result["material_names"]
    assert names[new_map[150, 150]] == "mat_grass_t0" and names[new_map[3050, 3050]] == "mat_grass_t3"
    assert names[new_map[10, 10]] == "aerial_photo_0" and names[new_map[3900, 10]] == "aerial_photo_2"
    assert (new_map[:, 4001:] == HOLE).all()
