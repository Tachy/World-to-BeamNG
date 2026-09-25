"""
Schreibt/liest BeamNG .ter Terrain-Dateien (Binärformat Version 9).

Format (empirisch verifiziert gegen eine echte .ter-Datei aus der BeamNG-
Installation, content/levels/GridMap.zip -> GridMap.ter):

    u8       version              (= 9)
    u32 LE   size                 (Kantenlänge, Zweierpotenz, 128-8192)
    u16[] LE heightmap            (size*size Werte, row-major)
    u8[]     layer_map            (size*size Werte, 255 = leer/Hole)
    u32 LE   material_count
    für jedes Material:
        u8   name_length
        ...  name (ASCII, name_length Bytes, kein Terminator)
"""

import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np

TER_VERSION = 9
VALID_SIZES = {128, 256, 512, 1024, 2048, 4096, 8192}
EMPTY_LAYER_VALUE = 255


def write_ter(
    path: Path,
    heightmap: np.ndarray,
    layer_map: np.ndarray,
    material_names: List[str],
) -> None:
    """
    Schreibt eine .ter-Datei.

    Args:
        path: Zielpfad der .ter-Datei
        heightmap: 2D uint16-Array, shape (size, size), row-major
        layer_map: 2D uint8-Array, gleiche Shape wie heightmap
        material_names: Materialnamen, Index entspricht layer_map-Werten
                        (max. 254 Einträge, Index 255 ist für "leer" reserviert)

    Raises:
        ValueError: bei ungültiger Größe, Shape-Mismatch oder zu vielen Materialien
    """
    if heightmap.shape != layer_map.shape:
        raise ValueError(f"heightmap shape {heightmap.shape} != layer_map shape {layer_map.shape}")

    if heightmap.ndim != 2 or heightmap.shape[0] != heightmap.shape[1]:
        raise ValueError(f"heightmap must be square, but is {heightmap.shape}")

    size = heightmap.shape[0]
    if size not in VALID_SIZES:
        raise ValueError(f"size must be a power of two between 128 and 8192 (spec: .ter format), is {size}")
    if len(material_names) > 254:
        raise ValueError(f"at most 254 materials allowed (255 is reserved for holes), {len(material_names)} given")

    heightmap_u16 = heightmap.astype("<u2", copy=False)
    layer_map_u8 = layer_map.astype("u1", copy=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<B", TER_VERSION))
        f.write(struct.pack("<I", size))
        f.write(heightmap_u16.tobytes(order="C"))
        f.write(layer_map_u8.tobytes(order="C"))
        f.write(struct.pack("<I", len(material_names)))
        for name in material_names:
            name_bytes = name.encode("ascii")
            if len(name_bytes) > 255:
                raise ValueError(f"Material name too long (>255 bytes): {name}")
            f.write(struct.pack("<B", len(name_bytes)))
            f.write(name_bytes)


def read_ter(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Liest eine .ter-Datei zurück (für Tests/Validierung).

    Returns:
        (heightmap, layer_map, material_names) - gleiche Typen wie write_ter's Input
    """
    with open(path, "rb") as f:
        data = f.read()

    version = data[0]
    if version != TER_VERSION:
        raise ValueError(f"Unexpected .ter version: {version} (expected {TER_VERSION})")

    size = struct.unpack_from("<I", data, 1)[0]
    offset = 5

    heightmap = np.frombuffer(data, dtype="<u2", count=size * size, offset=offset).reshape(size, size).copy()
    offset += size * size * 2

    layer_map = np.frombuffer(data, dtype="u1", count=size * size, offset=offset).reshape(size, size).copy()
    offset += size * size

    material_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    material_names: List[str] = []
    for _ in range(material_count):
        name_length = data[offset]
        offset += 1
        name = data[offset : offset + name_length].decode("ascii")
        offset += name_length
        material_names.append(name)

    return heightmap, layer_map, material_names


def encode_heights_to_u16(heights_m: np.ndarray, z_min: float, max_height: float) -> np.ndarray:
    """
    Wandelt absolute Höhenwerte (Meter) in das u16-Format der .ter-Heightmap um.

    Formel (siehe Spec Abschnitt 8): heightMeters = storedHeight * (maxHeight / 65536)
    -> storedHeight = (heightMeters - z_min) / maxHeight * 65536

    Args:
        heights_m: beliebige Shape, absolute Höhenwerte in Metern
        z_min: Höhe (Meter), die u16-Wert 0 entspricht
        max_height: Höhenbereich (Meter), den u16-Wert 65535 entspricht

    Returns:
        Gleiche Shape wie heights_m, dtype uint16, auf [0, 65535] geclampt
    """
    relative = (heights_m - z_min) / max_height * 65536.0
    clamped = np.clip(relative, 0, 65535)
    return clamped.astype(np.uint16)
