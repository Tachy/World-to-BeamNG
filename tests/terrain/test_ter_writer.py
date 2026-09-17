"""Tests für world_to_beamng.terrain.ter_writer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.ter_writer import (
    write_ter,
    read_ter,
    encode_heights_to_u16,
    decode_heights_from_u16,
)


def test_round_trip_small_terrain(tmp_path):
    size = 128
    heightmap = np.random.randint(0, 65536, size=(size, size), dtype=np.uint16)
    layer_map = np.random.randint(0, 3, size=(size, size), dtype=np.uint8)
    material_names = ["tile_0_0", "mat_forest", "mat_grass"]

    ter_path = tmp_path / "test.ter"
    write_ter(ter_path, heightmap, layer_map, material_names)

    read_heightmap, read_layer_map, read_names = read_ter(ter_path)

    assert np.array_equal(read_heightmap, heightmap)
    assert np.array_equal(read_layer_map, layer_map)
    assert read_names == material_names


def test_invalid_size_rejected(tmp_path):
    heightmap = np.zeros((100, 100), dtype=np.uint16)
    layer_map = np.zeros((100, 100), dtype=np.uint8)

    try:
        write_ter(tmp_path / "bad.ter", heightmap, layer_map, [])
        assert False, "sollte ValueError werfen (100 ist keine Zweierpotenz)"
    except ValueError as e:
        assert "Zweierpotenz" in str(e)


def test_height_encode_decode_round_trip():
    heights_m = np.array([0.0, 100.0, 256.0, 1023.5], dtype=np.float64)
    z_min = 0.0
    max_height = 1024.0

    encoded = encode_heights_to_u16(heights_m, z_min, max_height)
    decoded = decode_heights_from_u16(encoded, z_min, max_height)

    # Präzision: max_height / 65536 = 1024/65536 = 0.015625m pro Schritt
    assert np.allclose(decoded, heights_m, atol=0.02)


def test_material_name_length_limit(tmp_path):
    heightmap = np.zeros((128, 128), dtype=np.uint16)
    layer_map = np.zeros((128, 128), dtype=np.uint8)
    too_many = [f"mat_{i}" for i in range(255)]

    try:
        write_ter(tmp_path / "bad.ter", heightmap, layer_map, too_many)
        assert False, "sollte ValueError werfen (>254 Materialien)"
    except ValueError as e:
        assert "254" in str(e)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_round_trip_small_terrain(tmp_path)
        print("[OK] test_round_trip_small_terrain")
        test_invalid_size_rejected(tmp_path)
        print("[OK] test_invalid_size_rejected")
        test_height_encode_decode_round_trip()
        print("[OK] test_height_encode_decode_round_trip")
        test_material_name_length_limit(tmp_path)
        print("[OK] test_material_name_length_limit")
        print("Alle Tests bestanden.")
