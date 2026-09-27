"""
Tests for world_to_beamng.io.swissbuildings_dxf: swissBUILDINGS3D 2.0 polyface meshes become the same building dicts
as LoD2 CityGML (walls/roofs as planar polygons), with the overhang the data already models kept as it is.
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.io import lod2
from world_to_beamng.io.swissbuildings_dxf import is_dxf_source, load_swissbuildings, read_polyface_meshes

OX, OY = 2687000.0, 1155000.0  # LV95: large coordinates, like the real data


def _oriented(points, outward):
    """Polygon points ordered so that their normal points to `outward`."""
    points = np.asarray(points, dtype=float)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    return points if float(normal @ np.asarray(outward, dtype=float)) > 0 else points[::-1]


def _dxf(objects):
    """ASCII DXF with one polyface mesh per (layer, handle, [(polygon points, outward direction), ...])."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for layer, handle, polygons in objects:
        points, faces = [], []
        for polygon, outward in polygons:
            ring = _oriented(polygon, outward) + np.array([OX, OY, 0.0])
            first = len(points) + 1
            points.extend(ring.tolist())
            for i in range(1, len(ring) - 1):  # fan triangles, first edge hidden like in the real data (negative index)
                faces.append((-first, first + i, first + i + 1))
        out += ["0", "POLYLINE", "5", handle, "8", layer, "66", "1", "10", "0.0", "20", "0.0", "30", "0.0", "70", "64",
                "71", str(len(points)), "72", str(len(faces))]
        for x, y, z in points:
            out += ["0", "VERTEX", "8", layer, "10", f"{x:.3f}", "20", f"{y:.3f}", "30", f"{z:.3f}", "70", "192"]
        for a, b, c in faces:
            out += ["0", "VERTEX", "8", layer, "10", "0.0", "20", "0.0", "30", "0.0", "70", "128",
                    "71", str(a), "72", str(b), "73", str(c)]
        out += ["0", "SEQEND", "8", layer]
    out += ["0", "ENDSEC", "0", "EOF"]
    return ("\n".join(out) + "\n").encode("utf-8")


def _gabled_house(overhang: bool):
    """10 x 8 m, ground at 100 m, ridge along x at y = 4 (109 m), slope 0.75. With `overhang`: the roof reaches 0.6 m
    beyond the eaves and 0.3 m beyond the gables, 0.2 m thick, with its underside (soffit) and eave fascias."""
    z = lambda y: 109.0 - 0.75 * abs(y - 4.0)
    eave, verge, thick = (0.6, 0.3, 0.2) if overhang else (0.0, 0.0, 0.0)
    x0, x1, s, n = -verge, 10.0 + verge, -eave, 8.0 + eave
    wall_top = z(0.0) - thick  # the walls end under the roof slab
    polygons = [
        ([(0, 0, 100), (10, 0, 100), (10, 8, 100), (0, 8, 100)], (0, 0, -1)),  # ground plate
        ([(0, 0, 100), (10, 0, 100), (10, 0, wall_top), (0, 0, wall_top)], (0, -1, 0)),
        ([(0, 8, 100), (10, 8, 100), (10, 8, wall_top), (0, 8, wall_top)], (0, 1, 0)),
        ([(0, 0, 100), (0, 8, 100), (0, 8, wall_top), (0, 4, z(4.0) - thick), (0, 0, wall_top)], (-1, 0, 0)),
        ([(10, 0, 100), (10, 8, 100), (10, 8, wall_top), (10, 4, z(4.0) - thick), (10, 0, wall_top)], (1, 0, 0)),
        ([(x0, s, z(s)), (x1, s, z(s)), (x1, 4, z(4.0)), (x0, 4, z(4.0))], (0, -0.6, 0.8)),  # roof south
        ([(x0, n, z(n)), (x1, n, z(n)), (x1, 4, z(4.0)), (x0, 4, z(4.0))], (0, 0.6, 0.8)),  # roof north
    ]
    if overhang:
        polygons += [
            ([(x0, s, z(s) - thick), (x1, s, z(s) - thick), (x1, s, z(s)), (x0, s, z(s))], (0, -1, 0)),  # fascias
            ([(x0, n, z(n) - thick), (x1, n, z(n) - thick), (x1, n, z(n)), (x0, n, z(n))], (0, 1, 0)),
            ([(x0, s, z(s) - thick), (x1, s, z(s) - thick), (x1, 0, wall_top), (x0, 0, wall_top)], (0, 0.6, -0.8)),  # soffits
            ([(x0, n, z(n) - thick), (x1, n, z(n) - thick), (x1, 8, wall_top), (x0, 8, wall_top)], (0, -0.6, -0.8)),
            ([(x0, 0, wall_top), (0, 0, wall_top), (0, 4, z(4.0) - thick), (x0, 4, z(4.0) - thick)], (0, 0.6, -0.8)),
        ]
    return polygons


def _load(tmp_path, objects, name="swissbuildings3d_2_2023-05_1251-24_2056_5728.dxf"):
    path = tmp_path / name
    path.write_bytes(_dxf(objects))
    return load_swissbuildings(path)


def test_polyface_meshes_are_read_with_their_layer_handle_and_triangles():
    meshes = read_polyface_meshes(_dxf([("Gebaeude Einzelhaus", "60", _gabled_house(overhang=False))]))

    (mesh,) = meshes
    assert mesh["layer"] == "Gebaeude Einzelhaus" and mesh["handle"] == "60"
    assert mesh["triangles"].min() >= 0  # negative (hidden edge) indices read as their absolute value
    assert mesh["vertices"][:, 0].min() == pytest.approx(OX)


def test_triangles_are_merged_into_planar_walls_and_roofs(tmp_path):
    (house,) = _load(tmp_path, [("Gebaeude Einzelhaus", "60", _gabled_house(overhang=True))])

    assert len(house["walls"]) == 4  # two eave walls, two gable walls - the ground plate is dropped
    assert len(house["roofs"]) == 2
    assert sorted(len(ring) - 1 for ring, _ in house["walls"]) == [4, 4, 5, 5]
    assert house["id"] == "ch_1251-24_60"


def test_a_modelled_overhang_is_recognised_with_its_soffit_and_fascias(tmp_path):
    (house,) = _load(tmp_path, [("Gebaeude Einzelhaus", "60", _gabled_house(overhang=True))])

    assert house["overhang_modeled"] is True
    assert len(house["fascias"]) == 2  # 20 cm edges at the eaves, not facade walls
    assert house["soffits"]
    assert max(ring[:, 0].max() for ring, _ in house["roofs"]) == pytest.approx(OX + 10.3)  # as in the data


def test_a_house_without_overhang_is_marked_for_the_computed_one(tmp_path):
    (house,) = _load(tmp_path, [("Gebaeude Einzelhaus", "61", _gabled_house(overhang=False))])

    assert house["overhang_modeled"] is False
    assert not house["soffits"] and not house["fascias"]


def test_layers_decide_what_is_built(tmp_path):
    box = [([(0, 0, 0), (1, 0, 0), (1, 0, 2), (0, 0, 2)], (0, -1, 0)), ([(0, 0, 2), (1, 0, 2), (1, 0.4, 2), (0, 0.4, 2)], (0, 0, 1))]
    buildings = _load(tmp_path, [
        ("Mauer gross", "1", box),
        ("Unterirdisches Gebaeude", "2", _gabled_house(overhang=False)),
        ("Gebaeude unsichtbar", "3", _gabled_house(overhang=False)),
        ("Unbekannter Layer", "4", _gabled_house(overhang=False)),
    ])

    kinds = {b["id"]: b["kind"] for b in buildings}
    assert kinds == {"ch_1251-24_1": "wall", "ch_1251-24_4": "building"}
    wall = next(b for b in buildings if b["kind"] == "wall")
    assert wall["stone"] and not wall["walls"] and not wall["roofs"]


def test_the_format_is_recognised_from_the_contents(tmp_path):
    dxf_zip = tmp_path / "buildings.zip"
    with zipfile.ZipFile(dxf_zip, "w") as archive:
        archive.writestr("SWISSBUILDINGS3D_2_0_CHLV95LN02_1251-24.dxf", _dxf([("Gebaeude Einzelhaus", "60", _gabled_house(True))]))
    gml_zip = tmp_path / "LoD2_32_1_2_2_bw.zip"
    with zipfile.ZipFile(gml_zip, "w") as archive:
        archive.writestr("LoD2.gml", "<CityModel/>")

    assert is_dxf_source(dxf_zip) and not is_dxf_source(gml_zip)
    assert [b["id"] for b in lod2.load_buildings_from_file(dxf_zip)] == ["ch_1251-24_60"]


def test_the_pipeline_keeps_the_overhang_and_shifts_it_into_local_coordinates(tmp_path):
    from world_to_beamng.geometry import coordinates

    coordinates.set_source_crs(2056)
    buildings_dir = tmp_path / "buildings"
    buildings_dir.mkdir()
    (buildings_dir / "SWISSBUILDINGS3D_2_0_CHLV95LN02_1251-24.dxf").write_bytes(
        _dxf([("Gebaeude Einzelhaus", "60", _gabled_house(True))])
    )
    from pyproj import Transformer

    to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    min_lon, min_lat = to_wgs.transform(OX - 100, OY - 100)
    max_lon, max_lat = to_wgs.transform(OX + 100, OY + 100)

    cache_file = lod2.cache_lod2_buildings(
        str(buildings_dir), (min_lat, min_lon, max_lat, max_lon), (OX, OY, 0.0), str(tmp_path / "cache"), "area"
    )
    (house,) = lod2.load_buildings_from_cache(cache_file)

    assert house["overhang_modeled"] is True and house["kind"] == "building"
    soffit_x = np.vstack([verts for verts, _ in house["soffits"]])[:, 0]
    assert soffit_x.min() == pytest.approx(-0.3)  # shifted like the walls and roofs
    assert house["bounds"][0] == pytest.approx(-0.3)


def test_changed_building_files_give_a_new_cache(tmp_path):
    buildings_dir = tmp_path / "buildings"
    buildings_dir.mkdir()
    first = lod2.lod2_cache_file(buildings_dir, tmp_path, "area")
    (buildings_dir / "a.dxf").write_bytes(_dxf([("Gebaeude Einzelhaus", "60", _gabled_house(True))]))

    assert lod2.lod2_cache_file(buildings_dir, tmp_path, "area") != first


def test_one_modelled_overhang_switches_the_computed_one_off_for_the_whole_import(tmp_path):
    from world_to_beamng.workflow.building_workflow import overhangs_modeled

    modelled, plain = _load(tmp_path, [
        ("Gebaeude Einzelhaus", "60", _gabled_house(True)), ("Gebaeude Einzelhaus", "61", _gabled_house(False)),
    ])

    assert overhangs_modeled([modelled, plain]) is True
    assert overhangs_modeled([plain]) is False  # e.g. LoD2 from Baden-Württemberg: computed as before


def test_without_computed_overhangs_the_roofs_are_taken_as_they_are(tmp_path):
    from world_to_beamng.builders.mesh_builders import BuildingMeshBuilder
    from world_to_beamng.facade.material_names import ROOF_MATERIAL, ROOF_TRIM_MATERIAL

    def roof_x_range(building, compute_overhang):
        mesh = BuildingMeshBuilder(compute_overhang).with_buildings([building]).build()[0]
        roof = np.asarray(mesh["vertices"])[np.unique(np.ravel(mesh["faces"][ROOF_MATERIAL]))]
        return roof[:, 0].min() - OX, roof[:, 0].max() - OX, mesh

    modelled, plain = _load(tmp_path, [
        ("Gebaeude Einzelhaus", "60", _gabled_house(True)), ("Gebaeude Einzelhaus", "61", _gabled_house(False)),
    ])

    low, high, mesh = roof_x_range(modelled, compute_overhang=False)
    assert (low, high) == (pytest.approx(-0.3), pytest.approx(10.3))  # no second overhang on top
    assert mesh["faces"][ROOF_TRIM_MATERIAL]  # the data's soffit and fascias
    low, high, _ = roof_x_range(plain, compute_overhang=False)
    assert (low, high) == (pytest.approx(0.0), pytest.approx(10.0))  # the same import: none computed here either
    low, high, _ = roof_x_range(plain, compute_overhang=True)
    assert low < -0.2 and high > 10.2  # computed like for LoD2 from Baden-Württemberg (30 cm at the gables)


def test_a_free_standing_wall_is_built_in_rubble_stone(tmp_path):
    from world_to_beamng import config
    from world_to_beamng.builders.mesh_builders import BuildingMeshBuilder

    box = [([(0, 0, 0), (4, 0, 0), (4, 0, 2), (0, 0, 2)], (0, -1, 0)), ([(0, 0, 2), (4, 0, 2), (4, 0.4, 2), (0, 0.4, 2)], (0, 0, 1))]
    (wall,) = _load(tmp_path, [("Mauer gross", "1", box)])

    (mesh,) = BuildingMeshBuilder().with_buildings([wall]).build()

    assert set(mesh["faces"]) == {config.WALL_MATERIAL_NAME}
    assert np.ptp(mesh["uvs"], axis=0).max() > 1.0  # metric UVs: a 4 m wall repeats the stone texture


def test_the_ground_level_per_wall_comes_from_the_terrain(tmp_path):
    from world_to_beamng.io.swissbuildings_dxf import attach_wall_ground

    (house,) = _load(tmp_path, [("Gebaeude Einzelhaus", "60", _gabled_house(True))])
    lod2_house = {"id": "bw", "walls": house["walls"], "roofs": house["roofs"]}  # LoD2: walls start on the terrain

    # terrain rising to the north: 103 m at y = 0, 105 m at y = 8 (the body reaches down to 100 m)
    count = attach_wall_ground([house, lod2_house], lambda xy: 103.0 + 0.25 * (np.asarray(xy)[:, 1] - OY))

    assert count == 1 and "wall_ground" not in lod2_house
    grounds = list(zip((ring for ring, _ in house["walls"]), house["wall_ground"]))
    eaves = {round(float(ring[0, 1] - OY)): ground for ring, ground in grounds if np.ptp(ring[:, 1]) < 1e-6}
    gables = [ground for ring, ground in grounds if np.ptp(ring[:, 0]) < 1e-6]
    assert eaves == {0: pytest.approx(103.0), 8: pytest.approx(105.0)}
    assert gables == [pytest.approx(103.0)] * 2  # the lowest terrain along their bottom edge
