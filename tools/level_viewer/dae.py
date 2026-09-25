"""
COLLADA (.dae) reader for the meshes the pipeline writes (bridges, tunnels, walls, buildings, horizon).

Vectorized with numpy: a DAE indexes positions and UVs separately per triangle corner, so every distinct
(position index, uv index) pair becomes one output vertex. Only <triangles> primitives are read (the exporter
writes nothing else); node transforms are not applied because the exported meshes are already in level coordinates.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from lxml import etree

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


@dataclass
class DaeMesh:
    name: str
    positions: np.ndarray  # (n, 3) float32
    uvs: Optional[np.ndarray]  # (n, 2) float32 or None
    triangles: Dict[str, np.ndarray] = field(default_factory=dict)  # material -> (m, 3) int64 indices into positions


def _float_array(source) -> Optional[np.ndarray]:
    array = source.find("c:float_array", NS)
    if array is None or not array.text:
        return None
    accessor = source.find("c:technique_common/c:accessor", NS)
    stride = int(accessor.get("stride", "1")) if accessor is not None else 1
    data = np.array(array.text.split(), dtype=np.float32)
    return data.reshape(-1, stride) if stride > 1 and data.size % stride == 0 else data


def _parse_geometry(geometry) -> Optional[DaeMesh]:
    mesh = geometry.find("c:mesh", NS)
    if mesh is None:
        return None
    sources = {src.get("id"): _float_array(src) for src in mesh.findall("c:source", NS)}
    # <vertices id=...><input semantic="POSITION" source="#..."/></vertices>
    vertices_alias = {}
    for vertices in mesh.findall("c:vertices", NS):
        for inp in vertices.findall("c:input", NS):
            if inp.get("semantic") == "POSITION":
                vertices_alias[vertices.get("id")] = inp.get("source", "").lstrip("#")

    corners_v, corners_t, groups = [], [], []
    positions = uvs = None
    for tris in mesh.findall("c:triangles", NS):
        p = tris.find("c:p", NS)
        if p is None or not p.text:
            continue
        offsets, stride = {}, 0
        for inp in tris.findall("c:input", NS):
            offset = int(inp.get("offset", "0"))
            stride = max(stride, offset + 1)
            semantic = inp.get("semantic")
            source_id = inp.get("source", "").lstrip("#")
            if semantic == "VERTEX":
                offsets["v"] = offset
                positions = sources.get(vertices_alias.get(source_id, source_id))
            elif semantic == "TEXCOORD" and "t" not in offsets:
                offsets["t"] = offset
                uvs = sources.get(source_id)
        if "v" not in offsets or positions is None:
            continue
        idx = np.array(p.text.split(), dtype=np.int64)
        idx = idx[: len(idx) - len(idx) % (3 * stride)].reshape(-1, stride)
        corners_v.append(idx[:, offsets["v"]])
        corners_t.append(idx[:, offsets["t"]] if "t" in offsets and uvs is not None else np.zeros(len(idx), dtype=np.int64))
        groups.append((tris.get("material", "unknown"), len(idx) // 3))

    if not corners_v:
        return None
    v = np.concatenate(corners_v)
    t = np.concatenate(corners_t)
    pairs, inverse = np.unique(np.column_stack([v, t]), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    out_uvs = uvs[pairs[:, 1]].astype(np.float32) if uvs is not None and uvs.ndim == 2 else None

    triangles: Dict[str, List[np.ndarray]] = {}
    start = 0
    for material, count in groups:
        faces = inverse[start * 3 : (start + count) * 3].reshape(-1, 3)
        triangles.setdefault(material, []).append(faces)
        start += count
    return DaeMesh(
        name=geometry.get("name") or geometry.get("id") or "mesh",
        positions=positions[pairs[:, 0]].astype(np.float32),
        uvs=out_uvs,
        triangles={m: np.vstack(f) for m, f in triangles.items()},
    )


def read_dae(path) -> List[DaeMesh]:
    """All <geometry> meshes of a DAE file."""
    parser = etree.XMLParser(huge_tree=True, resolve_entities=False)
    root = etree.parse(str(path), parser).getroot()
    meshes = []
    for geometry in root.iterfind(".//c:library_geometries/c:geometry", NS):
        mesh = _parse_geometry(geometry)
        if mesh is not None:
            meshes.append(mesh)
    return meshes


def load_dae_tile(filepath) -> dict:
    """
    Compatibility view for tools/test_export_integrity.py: all meshes of a DAE merged into one vertex array.

    Returns {"vertices": (n, 3), "faces": list of [i0, i1, i2], "materials": material per face,
    "materials_per_face": {material: [face indices]}, "filepath": str}.
    """
    vertices, faces, materials = [], [], []
    offset = 0
    for mesh in read_dae(filepath):
        vertices.append(mesh.positions)
        for material, tris in mesh.triangles.items():
            faces.extend((tris + offset).tolist())
            materials.extend([material] * len(tris))
        offset += len(mesh.positions)
    per_material: Dict[str, List[int]] = {}
    for i, material in enumerate(materials):
        per_material.setdefault(material, []).append(i)
    return {
        "vertices": np.vstack(vertices) if vertices else np.zeros((0, 3), dtype=np.float32),
        "faces": faces,
        "materials": materials,
        "materials_per_face": per_material,
        "filepath": str(filepath),
    }
