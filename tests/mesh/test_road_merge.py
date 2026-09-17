"""Tests für world_to_beamng.mesh.road_merge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.mesh.road_merge import merge_road_exports


def _make_prepared(vertex_count, face_offset_material, uv_count):
    """Baut ein minimales prepare_road_export()-Ergebnis für Tile-Tests."""
    vertices = np.arange(vertex_count * 3, dtype=np.float64).reshape(vertex_count, 3)
    faces = [[0, 1, 2]] if vertex_count >= 3 else []
    materials_per_face = [face_offset_material] if faces else []
    unique_materials = {face_offset_material: {"internal_name": face_offset_material}}
    uvs = [(float(i), float(i)) for i in range(uv_count)]
    uv_indices = {0: [0, 1, 2 % uv_count]} if faces else {}
    return {
        "vertices": vertices,
        "faces": faces,
        "materials_per_face": materials_per_face,
        "unique_materials": unique_materials,
        "uv_indices": uv_indices,
        "uvs": uvs,
    }


def test_merge_offsets_vertex_and_face_indices():
    tile_a = _make_prepared(vertex_count=3, face_offset_material="mat_a", uv_count=3)
    tile_b = _make_prepared(vertex_count=4, face_offset_material="mat_b", uv_count=3)

    merged = merge_road_exports([tile_a, tile_b])

    assert merged["vertices"].shape == (7, 3)
    # Tile A: Face [0,1,2] unverändert
    assert merged["faces"][0] == [0, 1, 2]
    # Tile B: Face [0,1,2] um 3 Vertices (Tile A) verschoben -> [3,4,5]
    assert merged["faces"][1] == [3, 4, 5]
    assert merged["materials_per_face"] == ["mat_a", "mat_b"]


def test_merge_offsets_uv_indices_and_concatenates_uvs():
    tile_a = _make_prepared(vertex_count=3, face_offset_material="mat_a", uv_count=3)
    tile_b = _make_prepared(vertex_count=3, face_offset_material="mat_b", uv_count=2)

    merged = merge_road_exports([tile_a, tile_b])

    assert len(merged["uvs"]) == 5  # 3 von Tile A + 2 von Tile B
    # Face 0 (Tile A) referenziert UV-Indizes 0..2 (unverändert)
    assert merged["uv_indices"][0] == [0, 1, 2]
    # Face 1 (Tile B, face_offset=1) referenziert UV-Indizes ab Offset 3
    assert merged["uv_indices"][1] == [3, 4, 3]


def test_merge_deduplicates_unique_materials_by_name():
    tile_a = _make_prepared(vertex_count=3, face_offset_material="mat_shared", uv_count=3)
    tile_b = _make_prepared(vertex_count=3, face_offset_material="mat_shared", uv_count=3)

    merged = merge_road_exports([tile_a, tile_b])

    assert list(merged["unique_materials"].keys()) == ["mat_shared"]


def test_merge_empty_list_returns_empty_structure():
    merged = merge_road_exports([])

    assert merged["vertices"].shape == (0, 3)
    assert merged["faces"] == []
    assert merged["materials_per_face"] == []
    assert merged["unique_materials"] == {}
    assert merged["uv_indices"] == {}
    assert merged["uvs"] == []


if __name__ == "__main__":
    test_merge_offsets_vertex_and_face_indices()
    print("[OK] test_merge_offsets_vertex_and_face_indices")
    test_merge_offsets_uv_indices_and_concatenates_uvs()
    print("[OK] test_merge_offsets_uv_indices_and_concatenates_uvs")
    test_merge_deduplicates_unique_materials_by_name()
    print("[OK] test_merge_deduplicates_unique_materials_by_name")
    test_merge_empty_list_returns_empty_structure()
    print("[OK] test_merge_empty_list_returns_empty_structure")
    print("Alle Tests bestanden.")
