"""
swissBUILDINGS3D 2.0 (swisstopo) from DXF: one polyface mesh per object (POLYLINE flag 64 with its VERTEX records),
triangles only, closed bodies including the ground plate, LV95/LN02. There is no semantic split into wall and roof
surfaces - only the layer (object type) - so the triangles are classified by their normal and the coplanar ones are
merged into planar polygons: the same building dict as the CityGML reader (io/lod2.py) gives, so the facade, roof and
church tower code works unchanged.

Unlike LoD2 from Baden-Württemberg, the roofs already reach beyond the walls: the underside of that overhang (soffit)
and its edge (fascia) are part of the mesh. They are kept as "soffits"/"fascias" and the building is marked
"overhang_modeled", so no second overhang is computed on top.
"""

import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

WALL_MAX_NZ = 0.1  # |normal z| below this: vertical (wall)
PLANE_ANGLE_DEG = 1.0  # triangles within this angle and ...
PLANE_DISTANCE_M = 0.02  # ... this distance lie in the same plane
EAVE_TOLERANCE_M = 0.5  # downward faces at most this far below the lowest roof point are the overhang's underside
FASCIA_MAX_HEIGHT_M = 0.5  # vertical faces up to this height at the eave are the overhang's edge, not a facade
SIMPLIFY_M = 0.001  # drops collinear points left over from merging the triangles

VERTEX_FLAG_FACE = 128
VERTEX_FLAG_POLYFACE_POINT = 64


def is_dxf_source(path: Path) -> bool:
    """A .dxf file or a ZIP that contains one."""
    path = Path(path)
    if path.suffix.lower() == ".dxf":
        return True
    if path.suffix.lower() != ".zip":
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.lower().endswith(".dxf") for name in archive.namelist())
    except zipfile.BadZipFile:
        return False


def _read_text(path: Path) -> List[Tuple[str, str]]:
    """(name, text) of the DXF file itself or of every DXF inside the ZIP."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            return [(name, archive.read(name)) for name in archive.namelist() if name.lower().endswith(".dxf")]
    return [(path.name, path.read_bytes())]


def read_polyface_meshes(data: bytes) -> List[Dict]:
    """
    All polyface meshes of an ASCII DXF: [{"handle", "layer", "vertices" (N, 3), "triangles" (K, 3) int}, ...].
    Face records may be quads (fourth index != 0: split into two triangles); negative indices only mark invisible
    edges. Binary DXF is not supported.
    """
    if data.startswith(b"AutoCAD Binary DXF"):
        raise ValueError("binary DXF is not supported - export swissBUILDINGS3D as ASCII DXF")
    lines = data.decode("utf-8", errors="replace").splitlines()

    meshes, mesh, entity, in_entities = [], None, None, False

    def close_entity():
        nonlocal mesh
        if entity is None:
            return
        kind = entity.get("0")
        if kind == "POLYLINE":
            mesh = {"handle": entity.get("5", ""), "layer": entity.get("8", ""), "points": [], "faces": []}
            if int(entity.get("70", "0")) & 64:
                meshes.append(mesh)
        elif kind == "VERTEX" and mesh is not None:
            flags = int(entity.get("70", "0"))
            if flags & VERTEX_FLAG_FACE and not flags & VERTEX_FLAG_POLYFACE_POINT:
                indices = [abs(int(entity.get(code, "0"))) for code in ("71", "72", "73", "74")]
                indices = [i - 1 for i in indices if i != 0]
                if len(indices) >= 3:
                    mesh["faces"].append(indices[:3])
                if len(indices) == 4:
                    mesh["faces"].append([indices[0], indices[2], indices[3]])
            else:
                mesh["points"].append((float(entity.get("10", "0")), float(entity.get("20", "0")), float(entity.get("30", "0"))))
        elif kind == "SEQEND":
            mesh = None

    for i in range(0, len(lines) - 1, 2):
        code, value = lines[i].strip(), lines[i + 1].strip()
        if not in_entities:
            in_entities = code == "2" and value == "ENTITIES"
            continue
        if code == "0":
            close_entity()
            if value == "ENDSEC":
                entity = None
                break
            entity = {"0": value}
        elif entity is not None:
            entity.setdefault(code, value)
    close_entity()

    result = []
    for mesh in meshes:
        points = np.asarray(mesh["points"], dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(mesh["faces"], dtype=np.int64).reshape(-1, 3)
        faces = faces[(faces < len(points)).all(axis=1)] if len(faces) else faces
        if len(points) and len(faces):
            result.append({"handle": mesh["handle"], "layer": mesh["layer"], "vertices": points, "triangles": faces})
    return result


def _sheet_of(name: str) -> str:
    """Map sheet of a swissBUILDINGS3D file ("1251-24"), else the file stem - part of the building ids. The last
    match: the download names carry the release date ("2023-05") before the sheet."""
    matches = re.findall(r"\d{4}-\d{2}", name)
    return matches[-1] if matches else Path(name).stem


def _plane_groups(triangles: np.ndarray, normals: np.ndarray, offsets: np.ndarray, members: Sequence[int]) -> List[List[int]]:
    """Triangle indices grouped by plane (normal within PLANE_ANGLE_DEG, offset within PLANE_DISTANCE_M)."""
    cos_limit = np.cos(np.radians(PLANE_ANGLE_DEG))
    groups: List[Tuple[np.ndarray, float, List[int]]] = []
    for index in members:
        for normal, offset, group in groups:
            if float(normal @ normals[index]) >= cos_limit and abs(offset - offsets[index]) <= PLANE_DISTANCE_M:
                group.append(index)
                break
        else:
            groups.append((normals[index], float(offsets[index]), [index]))
    return [group for _, _, group in groups]


def _merge_plane(corners: np.ndarray, normal: np.ndarray) -> List[np.ndarray]:
    """
    Planar polygons (closed (N, 3) rings, outer boundary only) covered by triangles `corners` (K, 3, 3) that share
    one plane with unit `normal`: union in the plane, back to 3D. Ring order follows the normal (counterclockwise
    seen from the side it points to).
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, normal)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    origin = corners[0, 0]
    local = np.einsum("kpd,ed->kpe", corners - origin, np.stack([e1, e2]))
    union = unary_union([Polygon(tri) for tri in local if Polygon(tri).area > 1e-8])
    parts = [union] if union.geom_type == "Polygon" else list(getattr(union, "geoms", []))
    rings = []
    for part in parts:
        if part.is_empty or part.area < 1e-6:
            continue
        part = part.simplify(SIMPLIFY_M)
        if part.is_empty or part.geom_type != "Polygon":
            continue
        ring2d = np.asarray(part.exterior.coords)
        if part.exterior.is_ccw is False:
            ring2d = ring2d[::-1]
        rings.append(origin + ring2d[:, :1] * e1 + ring2d[:, 1:2] * e2)
    return rings


def _fan(ring: np.ndarray) -> np.ndarray:
    """Fan triangulation of a closed ring (same form as the CityGML reader gives)."""
    count = len(ring) - 1
    return np.array([[0, i, i + 1] for i in range(1, count - 1)], dtype=np.int64).reshape(-1, 3)


def mesh_to_building(mesh: Dict, sheet: str, kind: str = "building") -> Optional[Dict]:
    """
    Building dict of one polyface mesh: "walls"/"roofs" as planar polygons [(ring (N, 3), faces), ...] like
    parse_citygml_buildings(), plus "soffits"/"fascias" (the overhang the data already models, raw triangles),
    "overhang_modeled" and "kind". A "wall" kind (large free-standing walls) keeps all triangles as "stone".
    """
    vertices, triangles = mesh["vertices"], mesh["triangles"]
    corners = vertices[triangles]  # (K, 3, 3)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    keep = area2 > 1e-9
    corners, cross, area2 = corners[keep], cross[keep], area2[keep]
    if not len(corners):
        return None
    normals = cross / area2[:, None]
    # plane offsets from a point of the building, not from the CRS origin: at LV95 coordinates (~2.7e6 m) a normal
    # that differs by rounding only would move the offset by meters and split one plane into many
    offsets = np.einsum("kd,kd->k", normals, corners[:, 0] - corners[0, 0])

    building = {
        "id": f"ch_{sheet}_{mesh['handle']}",
        "kind": kind,
        "walls": [],
        "roofs": [],
        "soffits": [],
        "fascias": [],
        "stone": [],
        "overhang_modeled": False,
        # swissBUILDINGS3D bodies reach a few meters into the ground (~3 m below the lowest terrain corner in the
        # Ticino data): their ground level comes from the terrain, see attach_wall_ground()
        "base_below_ground": True,
    }
    if kind == "wall":
        building["stone"] = [(tri.copy(), np.array([[0, 1, 2]])) for tri in corners]
    else:
        up = normals[:, 2] >= WALL_MAX_NZ
        down = normals[:, 2] <= -WALL_MAX_NZ
        vertical = ~up & ~down
        eave_z = float(corners[up][:, :, 2].min()) if up.any() else float(corners[:, :, 2].max())
        soffit = down & (corners[:, :, 2].mean(axis=1) >= eave_z - EAVE_TOLERANCE_M)  # the rest of `down`: ground

        for group in _plane_groups(corners, normals, offsets, np.flatnonzero(up)):
            for ring in _merge_plane(corners[group], normals[group[0]]):
                building["roofs"].append((ring, _fan(ring)))
        for group in _plane_groups(corners, normals, offsets, np.flatnonzero(vertical)):
            for ring in _merge_plane(corners[group], normals[group[0]]):
                height = float(ring[:, 2].max() - ring[:, 2].min())
                at_eave = float(ring[:, 2].min()) >= eave_z - EAVE_TOLERANCE_M
                if at_eave and height <= FASCIA_MAX_HEIGHT_M and up.any():
                    building["fascias"].append((ring, _fan(ring)))
                else:
                    building["walls"].append((ring, _fan(ring)))
        building["soffits"] = [(tri.copy(), np.array([[0, 1, 2]])) for tri in corners[soffit]]
        building["overhang_modeled"] = bool(soffit.any())

    all_points = corners.reshape(-1, 3)
    building["bounds"] = (
        float(all_points[:, 0].min()), float(all_points[:, 1].min()), float(all_points[:, 2].min()),
        float(all_points[:, 0].max()), float(all_points[:, 1].max()), float(all_points[:, 2].max()),
    )
    return building


GROUND_SAMPLE_STEP_M = 1.0  # terrain samples along the bottom edge of a wall


def attach_wall_ground(buildings: Sequence[Dict], height_at) -> int:
    """
    For buildings whose body reaches into the ground ("base_below_ground"): "wall_ground" = terrain height per wall
    (the lowest along its bottom edge, from `height_at(xy (N, 2)) -> z (N,)` on the finished terrain) - the level the
    facade counts storeys from and puts doors and basement windows on, like the wall base of LoD2 that starts on the
    terrain. Returns the number of buildings handled.
    """
    count = 0
    for building in buildings:
        if not building.get("base_below_ground") or not building.get("walls"):
            continue
        ground = []
        for verts, _ in building["walls"]:
            ring = np.asarray(verts, dtype=np.float64)
            bottom = ring[ring[:, 2] <= ring[:, 2].min() + 0.05][:, :2]
            if len(bottom) >= 2:
                start, end = bottom[0], bottom[np.argmax(np.linalg.norm(bottom - bottom[0], axis=1))]
                steps = max(1, int(np.ceil(np.linalg.norm(end - start) / GROUND_SAMPLE_STEP_M)))
                bottom = start + (end - start) * np.linspace(0.0, 1.0, steps + 1)[:, None]
            ground.append(float(np.min(height_at(bottom))))
        building["wall_ground"] = ground
        count += 1
    return count


def load_swissbuildings(path: Path) -> List[Dict]:
    """Building dicts (source CRS, absolute heights) of one swissBUILDINGS3D DXF or ZIP; layers mapped with
    config.SWISSBUILDINGS_LAYER_KINDS (None = left out, unknown layers count as "building")."""
    buildings, skipped = [], {}
    for name, data in _read_text(path):
        sheet = _sheet_of(name)
        for mesh in read_polyface_meshes(data):
            kind = config.SWISSBUILDINGS_LAYER_KINDS.get(mesh["layer"], "building")
            if kind is None:
                skipped[mesh["layer"]] = skipped.get(mesh["layer"], 0) + 1
                continue
            building = mesh_to_building(mesh, sheet, kind)
            if building is not None:
                buildings.append(building)
    if skipped:
        logger.debug(f"  [i] {Path(path).name}: left out {skipped}")
    return buildings
