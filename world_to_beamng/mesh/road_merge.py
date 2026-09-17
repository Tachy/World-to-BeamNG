"""
Führt die Straßen-Export-Daten mehrerer DGM1-Kacheln (aus
TerrainWorkflow.prepare_road_export()) zu einem einzigen Vertex-/Face-Satz
zusammen, damit alle Straßen als EINE DAE statt einer 500m-Kachelung
exportiert werden können.

Reine Datenzusammenführung (keine I/O): Vertex-/Face-/UV-Indizes jeder
Kachel werden um die bis dahin laufende Gesamtzahl verschoben, damit sie
weiterhin auf das jeweils richtige Element im zusammengeführten Array
zeigen. Materialnamen sind bereits pro Face als String aufgelöst (nicht als
Kachel-lokale road_id), daher gibt es dabei keine Kollisionsgefahr.
"""

from typing import Dict, List

import numpy as np


def merge_road_exports(prepared_list: List[Dict]) -> Dict:
    """
    Args:
        prepared_list: Liste von Dicts aus TerrainWorkflow.prepare_road_export(),
            je {"vertices": (N,3) ndarray, "faces": List[[i0,i1,i2]],
                "materials_per_face": List[str], "unique_materials": Dict,
                "uv_indices": Dict[int, List[int]], "uvs": List[Tuple[float,float]]}

    Returns:
        Dict mit denselben Keys, zusammengeführt über alle Kacheln.
    """
    combined_vertices: List[np.ndarray] = []
    combined_faces: List[List[int]] = []
    combined_materials_per_face: List[str] = []
    combined_unique_materials: Dict = {}
    combined_uv_indices: Dict[int, List[int]] = {}
    combined_uvs: List = []

    vertex_offset = 0
    face_offset = 0
    uv_offset = 0

    for prepared in prepared_list:
        vertices = prepared["vertices"]
        faces = prepared["faces"]
        uvs = prepared["uvs"]

        combined_vertices.append(vertices)
        combined_faces.extend(
            [[i0 + vertex_offset, i1 + vertex_offset, i2 + vertex_offset] for i0, i1, i2 in faces]
        )
        combined_materials_per_face.extend(prepared["materials_per_face"])
        combined_unique_materials.update(prepared["unique_materials"])

        for local_face_idx, uv_idx_triplet in prepared["uv_indices"].items():
            combined_uv_indices[local_face_idx + face_offset] = [i + uv_offset for i in uv_idx_triplet]
        combined_uvs.extend(uvs)

        vertex_offset += len(vertices)
        face_offset += len(faces)
        uv_offset += len(uvs)

    merged_vertices = np.vstack(combined_vertices) if combined_vertices else np.zeros((0, 3))

    return {
        "vertices": merged_vertices,
        "faces": combined_faces,
        "materials_per_face": combined_materials_per_face,
        "unique_materials": combined_unique_materials,
        "uv_indices": combined_uv_indices,
        "uvs": combined_uvs,
    }
