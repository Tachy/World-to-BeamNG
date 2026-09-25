"""
Writes/reads BeamNG .ter terrain files (binary format version 9).

Format (empirically verified against a real .ter file from the BeamNG
installation, content/levels/GridMap.zip -> GridMap.ter):

    u8       version              (= 9)
    u32 LE   size                 (edge length, power of two, 128-8192)
    u16[] LE heightmap            (size*size values, row-major)
    u8[]     layer_map            (size*size values, 255 = empty/hole)
    u32 LE   material_count
    for each material:
        u8   name_length
        ...  name (ASCII, name_length bytes, no terminator)
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
    Writes a .ter file.

    Args:
        path: Target path of the .ter file
        heightmap: 2D uint16 array, shape (size, size), row-major
        layer_map: 2D uint8 array, same shape as heightmap
        material_names: Material names, index corresponds to layer_map values
                        (max. 254 entries, index 255 is reserved for "empty")

    Raises:
        ValueError: on invalid size, shape mismatch or too many materials
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
    used = np.unique(layer_map)
    unknown = used[(used != 255) & (used >= len(material_names))]
    if unknown.size:
        raise ValueError(f"layer_map references material index {int(unknown[0])}, but only {len(material_names)} materials given")

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
    Reads a .ter file back (for tests/validation).

    Returns:
        (heightmap, layer_map, material_names) - same types as write_ter's input
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
    Converts absolute elevation values (meters) to the u16 format of the .ter heightmap.

    Formula (see spec section 8): heightMeters = storedHeight * (maxHeight / 65536)
    -> storedHeight = (heightMeters - z_min) / maxHeight * 65536

    Args:
        heights_m: any shape, absolute elevation values in meters
        z_min: Height (meters) that corresponds to u16 value 0
        max_height: Height range (meters) that corresponds to u16 value 65535

    Returns:
        Same shape as heights_m, dtype uint16, clamped to [0, 65535]
    """
    relative = (heights_m - z_min) / max_height * 65536.0
    clamped = np.clip(relative, 0, 65535)
    return clamped.astype(np.uint16)
