"""
DAEExporter - Zentrale DAE/Collada Export-Klasse.

Konsolidiert DAE-Export für:
- Terrain (multi-tile meshes)
- Buildings (LoD2)
- Horizon (distant terrain)
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class DAEExporter:
    """
    Zentrale Klasse für DAE/Collada 1.4.1 Export.

    Features:
    - Gemeinsame XML-Struktur (Asset, Materials, Effects, Geometries)
    - Automatische UV-Generierung
    - Material-Binding
    - Optimierte NumPy-Integration
    """

    def __init__(self, level_name: str = "World_to_BeamNG"):
        """
        Initialisiere DAE Exporter.

        Args:
            level_name: Name des BeamNG Levels
        """
        self.level_name = level_name
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def _write_header(self, f) -> None:
        """Schreibe DAE XML Header und Asset."""
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<COLLADA version="1.4.1" xmlns="http://www.collada.org/2005/11/COLLADASchema">\n')
        f.write("  <asset>\n")
        f.write(f"    <created>{self.timestamp}</created>\n")
        f.write(f"    <modified>{self.timestamp}</modified>\n")
        f.write('    <unit name="meter" meter="1"/>\n')
        f.write("    <up_axis>Z_UP</up_axis>\n")
        f.write("  </asset>\n")

    def _write_footer(self, f, scene_id: str = "Scene") -> None:
        """Schreibe DAE XML Footer (Scene)."""
        f.write("  <scene>\n")
        f.write(f'    <instance_visual_scene url="#{scene_id}"/>\n')
        f.write("  </scene>\n")
        f.write("</COLLADA>\n")

    def _write_image_library(self, f, material_textures: Dict[str, str]) -> None:
        """Schreibe library_images mit Textur-Pfaden.

        Args:
            f: File handle
            material_textures: Dict {mat_name: texture_path}
        """
        if not material_textures:
            return

        f.write("  <library_images>\n")
        for mat_name in sorted(material_textures.keys()):
            texture_path = material_textures[mat_name]
            f.write(f'    <image id="{mat_name}_image" name="{mat_name}_image">\n')
            f.write(f"      <init_from>{texture_path}</init_from>\n")
            f.write("    </image>\n")
        f.write("  </library_images>\n")

    def _write_material_library(self, f, material_names: List[str]) -> None:
        """Schreibe library_materials mit Effects."""
        f.write("  <library_materials>\n")
        for mat_name in sorted(material_names):
            f.write(f'    <material id="{mat_name}" name="{mat_name}">\n')
            f.write(f'      <instance_effect url="#effect_{mat_name}"/>\n')
            f.write("    </material>\n")
        f.write("  </library_materials>\n")

    def _write_effect_library(
        self,
        f,
        material_names: List[str],
        colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
        material_textures: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Schreibe library_effects.

        Args:
            f: File handle
            material_names: Liste von Material-Namen
            colors: Optional Dict {mat_name: (r, g, b)} für diffuse colors
            material_textures: Optional Dict {mat_name: texture_path} für Texturen
        """
        f.write("  <library_effects>\n")
        for mat_name in sorted(material_names):
            f.write(f'    <effect id="effect_{mat_name}">\n')
            f.write("      <profile_COMMON>\n")

            # Wenn Textur vorhanden: newparam + sampler2D
            if material_textures and mat_name in material_textures:
                surface_id = f"{mat_name}_surface"
                sampler_id = f"{mat_name}_sampler"

                # Surface (verweist auf Image)
                f.write(f'        <newparam sid="{surface_id}">\n')
                f.write('          <surface type="2D">\n')
                f.write(f"            <init_from>{mat_name}_image</init_from>\n")
                f.write("          </surface>\n")
                f.write("        </newparam>\n")

                # Sampler2D (verweist auf Surface)
                f.write(f'        <newparam sid="{sampler_id}">\n')
                f.write("          <sampler2D>\n")
                f.write(f"            <source>{surface_id}</source>\n")
                f.write("          </sampler2D>\n")
                f.write("        </newparam>\n")

            f.write('        <technique sid="common">\n')
            f.write("          <phong>\n")
            f.write("            <diffuse>\n")

            # Textur oder Color
            if material_textures and mat_name in material_textures:
                sampler_id = f"{mat_name}_sampler"
                f.write(f'              <texture texture="{sampler_id}" texcoord="UVSET0"/>\n')
            elif colors and mat_name in colors:
                r, g, b = colors[mat_name]
                f.write(f'              <color sid="diffuse">{r:.3f} {g:.3f} {b:.3f} 1</color>\n')
            else:
                f.write('              <color sid="diffuse">0.8 0.8 0.8 1</color>\n')

            f.write("            </diffuse>\n")
            f.write("          </phong>\n")
            f.write("        </technique>\n")
            f.write("      </profile_COMMON>\n")
            f.write("    </effect>\n")
        f.write("  </library_effects>\n")

    def _write_vertices_source(self, f, source_id: str, vertices: np.ndarray) -> None:
        """
        Schreibe <source> für Vertices (XYZ).

        Args:
            f: File handle
            source_id: ID für <source>
            vertices: (N, 3) NumPy Array
        """
        f.write(f'        <source id="{source_id}">')
        f.write(f'\n          <float_array id="{source_id}_array" count="{len(vertices) * 3}">')

        # Alle Vertex-Werte in einer Zeile
        vertex_str = " ".join(f"{v[0]:.2f} {v[1]:.2f} {v[2]:.2f}" for v in vertices)
        f.write(f"\n{vertex_str}")

        f.write("\n          </float_array>\n")
        f.write("          <technique_common>\n")
        f.write(f'            <accessor source="#{source_id}_array" count="{len(vertices)}" stride="3">\n')
        f.write('              <param name="X" type="float"/>\n')
        f.write('              <param name="Y" type="float"/>\n')
        f.write('              <param name="Z" type="float"/>\n')
        f.write("            </accessor>\n")
        f.write("          </technique_common>\n")
        f.write("        </source>\n")

    def _write_uv_source(self, f, source_id: str, uv_coords: np.ndarray) -> None:
        """
        Schreibe <source> für UV-Koordinaten.

        Args:
            f: File handle
            source_id: ID für <source>
            uv_coords: (N, 2) NumPy Array
        """
        f.write(f'        <source id="{source_id}">')
        f.write(f'\n          <float_array id="{source_id}_array" count="{len(uv_coords) * 2}">')

        # Alle UV-Werte in einer Zeile
        uv_str = " ".join(f"{uv[0]:.6f} {uv[1]:.6f}" for uv in uv_coords)
        f.write(f"\n{uv_str}")

        f.write("\n          </float_array>\n")
        f.write("          <technique_common>\n")
        f.write(f'            <accessor source="#{source_id}_array" count="{len(uv_coords)}" stride="2">\n')
        f.write('              <param name="S" type="float"/>\n')
        f.write('              <param name="T" type="float"/>\n')
        f.write("            </accessor>\n")
        f.write("          </technique_common>\n")
        f.write("        </source>\n")

    def _compute_smooth_normals(self, vertices: np.ndarray, faces: list) -> np.ndarray:
        """
        Berechne Smooth Normals (gemittelt von angrenzenden Face-Normals).

        Args:
            vertices: (N, 3) NumPy Array der Vertex-Positionen
            faces: List von Face-Indizes oder (M, 3) Array

        Returns:
            (N, 3) Array mit Vertex-Normals (normalisiert)
        """
        if not faces:
            # Falls keine Faces, gebe default Normals zurück
            return np.array([[0, 0, 1]] * len(vertices), dtype=np.float32)

        faces = np.array(faces)

        # Initialisiere Normal-Akkumulator
        vertex_normals = np.zeros_like(vertices)

        # Berechne Face-Normals (Cross Product)
        v0 = vertices[faces[:, 0]]  # Erste Vertex jedes Dreiecks
        v1 = vertices[faces[:, 1]]  # Zweite Vertex
        v2 = vertices[faces[:, 2]]  # Dritte Vertex

        # Kanten
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Face-Normals (nicht normalisiert - Fläche wirkt als Gewicht)
        face_normals = np.cross(edge1, edge2)

        # Addiere Face-Normal zu jedem beteiligten Vertex
        for i, face_idx_set in enumerate(faces):
            for vertex_idx in face_idx_set:
                vertex_normals[vertex_idx] += face_normals[i]

        # Normalisiere alle Vertex-Normals
        # Berechne Längen
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        # Verhindere Division durch Null
        lengths[lengths == 0] = 1.0
        # Normalisiere
        vertex_normals = vertex_normals / lengths

        return vertex_normals

    def _write_normals_source(self, f, source_id: str, normals: np.ndarray) -> None:
        """
        Schreibe <source> für Normals (XYZ).

        Args:
            f: File handle
            source_id: ID für <source>
            normals: (N, 3) NumPy Array
        """
        f.write(f'        <source id="{source_id}">')
        f.write(f'\n          <float_array id="{source_id}_array" count="{len(normals) * 3}">')

        # Alle Normal-Werte in einer Zeile
        normal_str = " ".join(f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}" for n in normals)
        f.write(f"\n{normal_str}")

        f.write("\n          </float_array>\n")
        f.write("          <technique_common>\n")
        f.write(f'            <accessor source="#{source_id}_array" count="{len(normals)}" stride="3">\n')
        f.write('              <param name="X" type="float"/>\n')
        f.write('              <param name="Y" type="float"/>\n')
        f.write('              <param name="Z" type="float"/>\n')
        f.write("            </accessor>\n")
        f.write("          </technique_common>\n")
        f.write("        </source>\n")

    def _compute_uv_normalized(
        self,
        vertices: np.ndarray,
        uv_offset: Tuple[float, float] = (0.0, 0.0),
        uv_scale: Tuple[float, float] = (1.0, 1.0),
        tile_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """
        Berechne UV-Koordinaten (normalisiert auf Mesh-Bounds oder Tile-Bounds).

        Args:
            vertices: (N, 3) Vertices
            uv_offset: (u_offset, v_offset)
            uv_scale: (u_scale, v_scale)
            tile_bounds: Optional (x_min, x_max, y_min, y_max) - Wenn gesetzt, nutze diese
                        Bounds statt der Vertex-Bounds (wichtig für Multi-Tile UV-Mapping)

        Returns:
            (N, 2) UV-Koordinaten
        """
        # Nutze Tile-Bounds wenn verfügbar, sonst Vertex-Bounds
        if tile_bounds is not None:
            x_min, x_max, y_min, y_max = tile_bounds
        else:
            x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
            y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()

        # Normalisiere auf 0..1
        u = (vertices[:, 0] - x_min) / (x_max - x_min) if x_max > x_min else 0.0
        v = (vertices[:, 1] - y_min) / (y_max - y_min) if y_max > y_min else 0.0

        # Apply offset und scale
        u = uv_offset[0] + u * uv_scale[0]
        v = uv_offset[1] + v * uv_scale[1]

        return np.column_stack([u, v])

    def _write_triangles_with_normals(
        self,
        f,
        material_name: str,
        faces: Dict[str, np.ndarray],
        vertices_id: str,
        normals_id: str,
        uv_id: Optional[str] = None,
    ) -> None:
        """
        Schreibe <triangles> Block mit Normals.

        Args:
            f: File handle
            material_name: Material-Symbol
            faces: Dict {mat_name: (M, 3)} Face-Indizes
            vertices_id: ID des <vertices> Elements
            normals_id: ID der Normals <source>
            uv_id: Optional ID der UV <source>
        """
        # Sammle alle Faces aus allen Materialien
        all_faces = []
        for mat_name, face_array in faces.items():
            if mat_name == material_name:
                all_faces = face_array
                break

        if len(all_faces) == 0:
            return

        f.write(f'        <triangles material="{material_name}" count="{len(all_faces)}">\n')
        f.write(f'          <input semantic="VERTEX" source="#{vertices_id}" offset="0"/>\n')
        f.write(f'          <input semantic="NORMAL" source="#{normals_id}" offset="1"/>\n')

        if uv_id:
            f.write(f'          <input semantic="TEXCOORD" source="#{uv_id}" offset="2" set="0"/>\n')

        f.write("          <p>")

        # Alle Indizes mit Normals
        # Format mit offsets 0,1,2: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2
        # Normals-Index = Vertex-Index (per-vertex normals)
        # UV-Index = Vertex-Index (per-vertex UVs)
        if uv_id:
            # Mit Normals + UV: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2
            indices_str = " ".join(f"{v0} {v0} {v0} {v1} {v1} {v1} {v2} {v2} {v2}" for v0, v1, v2 in all_faces)
        else:
            # Nur Normals: v0 n0 v1 n1 v2 n2
            indices_str = " ".join(f"{v0} {v0} {v1} {v1} {v2} {v2}" for v0, v1, v2 in all_faces)

        f.write(f"\n{indices_str}")
        f.write("\n          </p>\n")
        f.write("        </triangles>\n")

    def _write_triangles(
        self,
        f,
        material_name: str,
        faces: List[Tuple[int, int, int]],
        vertices_id: str,
        normal_id: Optional[str] = None,
        uv_id: Optional[str] = None,
    ) -> None:
        """
        Schreibe <triangles> Block mit optionalen Normals und UVs.

        WICHTIG: Terrain-Tiles vs. Road-Materialien nutzen unterschiedliche TEXCOORD-Semantics:
        - Terrain Tiles (tile_*): semantic="TEXCOORD" (Original-Mapping)
        - Road Materials: semantic="TEXCOORD0" (für PBR-Shader-Support)

        Args:
            f: File handle
            material_name: Material-Symbol
            faces: Liste von (v0, v1, v2) Face-Indizes
            vertices_id: ID des <vertices> Elements
            normal_id: Optional ID der Normals <source>
            uv_id: Optional ID der UV <source>
        """
        f.write(f'        <triangles material="{material_name}" count="{len(faces)}">\n')
        f.write(f'          <input semantic="VERTEX" source="#{vertices_id}" offset="0"/>\n')

        offset = 1
        if normal_id:
            f.write(f'          <input semantic="NORMAL" source="#{normal_id}" offset="{offset}"/>\n')
            offset += 1

        if uv_id:
            # TEXCOORD (BeamNG-Standard für alle Materialien)
            f.write(f'          <input semantic="TEXCOORD" source="#{uv_id}" offset="{offset}" set="0"/>\n')
            offset += 1

        f.write("          <p>")

        # Alle Indizes in einer Zeile
        if normal_id and uv_id:
            # Mit Normals + UV: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2
            # Alle Indizes sind identisch, da 1:1 Mapping
            indices_str = " ".join(
                f"{face[0]} {face[0]} {face[0]} {face[1]} {face[1]} {face[1]} {face[2]} {face[2]} {face[2]}"
                for face in faces
            )
        elif normal_id:
            # Mit Normals nur: v0 n0 v1 n1 v2 n2
            indices_str = " ".join(f"{face[0]} {face[0]} {face[1]} {face[1]} {face[2]} {face[2]}" for face in faces)
        elif uv_id:
            # Mit UV nur: v0 uv0 v1 uv1 v2 uv2
            indices_str = " ".join(f"{face[0]} {face[0]} {face[1]} {face[1]} {face[2]} {face[2]}" for face in faces)
        else:
            # Ohne UV/Normals: v0 v1 v2
            indices_str = " ".join(f"{face[0]} {face[1]} {face[2]}" for face in faces)

        f.write(f"\n{indices_str}")
        f.write("\n          </p>\n")
        f.write("        </triangles>\n")

    def export_single_mesh(
        self,
        output_path: str,
        vertices: np.ndarray,
        faces: List[Tuple[int, int, int]],
        material_name: str,
        mesh_id: str = "mesh",
        with_uv: bool = False,
        uv_offset: Tuple[float, float] = (0.0, 0.0),
        uv_scale: Tuple[float, float] = (1.0, 1.0),
    ) -> str:
        """
        Exportiere Single-Mesh DAE (z.B. Horizon).

        Args:
            output_path: Ziel-Dateipfad
            vertices: (N, 3) Vertices
            faces: Liste von (v0, v1, v2)
            material_name: Material-Name
            mesh_id: Mesh-ID
            with_uv: UV-Koordinaten generieren?
            uv_offset: UV-Offset
            uv_scale: UV-Skalierung

        Returns:
            output_path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            self._write_header(f)

            # Materials
            self._write_material_library(f, [material_name])
            self._write_effect_library(f, [material_name], colors=None, material_textures=None)

            # Geometries
            f.write("  <library_geometries>\n")
            f.write(f'    <geometry id="{mesh_id}_geometry" name="{mesh_id}">\n')
            f.write("      <mesh>\n")

            # Vertices Source
            vert_src_id = f"{mesh_id}_vertices"
            self._write_vertices_source(f, vert_src_id, vertices)

            # Normals Source (NEW: Smooth Normals für BeamNG)
            normal_src_id = f"{mesh_id}_normals"
            smooth_normals = self._compute_smooth_normals(vertices, faces)
            self._write_normals_source(f, normal_src_id, smooth_normals)

            # UV Source (optional)
            if with_uv:
                uv_src_id = f"{mesh_id}_uvs"
                # Single-Mesh Export: keine tile_bounds (None)
                uv_coords = self._compute_uv_normalized(vertices, uv_offset, uv_scale, None)
                self._write_uv_source(f, uv_src_id, uv_coords)
            else:
                uv_src_id = None

            # Vertices Element
            vert_elem_id = f"{mesh_id}_vertices_input"
            f.write(f'        <vertices id="{vert_elem_id}">\n')
            f.write(f'          <input semantic="POSITION" source="#{vert_src_id}"/>\n')
            f.write("        </vertices>\n")

            # Triangles mit Normals
            f.write(f'        <triangles material="{material_name}" count="{len(faces)}">\n')
            f.write(f'          <input semantic="VERTEX" source="#{vert_elem_id}" offset="0"/>\n')
            f.write(f'          <input semantic="NORMAL" source="#{normal_src_id}" offset="1"/>\n')

            if uv_src_id:
                f.write(f'          <input semantic="TEXCOORD" source="#{uv_src_id}" offset="2" set="0"/>\n')

            f.write("          <p>")

            if uv_src_id:
                # Mit Normals + UV: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2
                indices_str = " ".join(
                    f"{face[0]} {face[0]} {face[0]} {face[1]} {face[1]} {face[1]} {face[2]} {face[2]} {face[2]}"
                    for face in faces
                )
            else:
                # Mit Normals nur: v0 n0 v1 n1 v2 n2
                indices_str = " ".join(f"{face[0]} {face[0]} {face[1]} {face[1]} {face[2]} {face[2]}" for face in faces)

            f.write(f"\n{indices_str}")
            f.write("\n          </p>\n")
            f.write("        </triangles>\n")
            f.write("        </triangles>\n")

            f.write("      </mesh>\n")
            f.write("    </geometry>\n")
            f.write("  </library_geometries>\n")

            # Visual Scene
            f.write("  <library_visual_scenes>\n")
            f.write('    <visual_scene id="Scene" name="Scene">\n')
            f.write(f'      <node id="{mesh_id}_node" name="{mesh_id}" type="NODE">\n')
            f.write(f'        <instance_geometry url="#{mesh_id}_geometry">\n')
            f.write("          <bind_material>\n")
            f.write("            <technique_common>\n")
            f.write(f'              <instance_material symbol="{material_name}" target="#{material_name}"/>\n')
            f.write("            </technique_common>\n")
            f.write("          </bind_material>\n")
            f.write("        </instance_geometry>\n")
            f.write("      </node>\n")
            f.write("    </visual_scene>\n")
            f.write("  </library_visual_scenes>\n")

            self._write_footer(f)

        return output_path

    def export_multi_mesh(
        self,
        output_path: str,
        meshes: List[Dict[str, Any]],
        with_uv: bool = False,
        material_textures: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Exportiere Multi-Mesh DAE (z.B. Terrain-Tiles, Buildings).

        Args:
            output_path: Ziel-Dateipfad
            meshes: Liste von Dicts mit:
                - 'id': Mesh-ID
                - 'vertices': (N, 3) NumPy Array
                - 'faces': Liste von (v0, v1, v2) oder Dict {mat_name: faces_list}
                - 'material' (optional): Material-Name (falls faces eine Liste)
                - 'uv_offset' (optional): UV-Offset
                - 'uv_scale' (optional): UV-Skalierung
            with_uv: UV-Koordinaten generieren?
            material_textures: Optional Dict {mat_name: texture_path} für Textur-Bindings

        Returns:
            output_path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Sammle alle Material-Namen
        material_names = set()
        for mesh_data in meshes:
            faces = mesh_data.get("faces", [])
            if isinstance(faces, dict):
                material_names.update(faces.keys())
            elif "material" in mesh_data:
                material_names.add(mesh_data["material"])

        with open(output_path, "w", encoding="utf-8") as f:
            self._write_header(f)

            # Images (Texturen)
            if material_textures:
                self._write_image_library(f, material_textures)

            # Materials
            self._write_material_library(f, list(material_names))

            # Colors für Buildings (wall/roof)
            colors = {}
            if "lod2_wall_white" in material_names:
                colors["lod2_wall_white"] = (0.95, 0.95, 0.95)
            if "lod2_roof_red" in material_names:
                colors["lod2_roof_red"] = (0.6, 0.2, 0.1)

            self._write_effect_library(f, list(material_names), colors, material_textures)

            # Geometries
            f.write("  <library_geometries>\n")

            for mesh_data in meshes:
                mesh_id = mesh_data["id"]
                vertices = mesh_data["vertices"]
                faces = mesh_data.get("faces", [])
                uv_offset = mesh_data.get("uv_offset", (0.0, 0.0))
                uv_scale = mesh_data.get("uv_scale", (1.0, 1.0))

                f.write(f'    <geometry id="{mesh_id}_geometry" name="{mesh_id}">\n')
                f.write("      <mesh>\n")

                # Vertices Source
                vert_src_id = f"{mesh_id}_vertices"
                self._write_vertices_source(f, vert_src_id, vertices)

                # Normals Source (NEW: Smooth Normals für BeamNG)
                normal_src_id = f"{mesh_id}_normals"
                provided_normals = mesh_data.get("normals")
                if provided_normals is not None:
                    smooth_normals = provided_normals
                else:
                    if isinstance(faces, dict):
                        # Kombiniere alle Faces aus allen Materialen für Normal-Berechnung
                        all_faces = []
                        for mat_faces in faces.values():
                            all_faces.extend(mat_faces)
                        smooth_normals = self._compute_smooth_normals(vertices, all_faces)
                    else:
                        smooth_normals = self._compute_smooth_normals(vertices, faces)
                self._write_normals_source(f, normal_src_id, smooth_normals)

                # UV Source (optional)
                if with_uv:
                    uv_src_id = f"{mesh_id}_uvs"
                    # Prüfe zuerst ob explizite UVs im mesh_data vorhanden sind
                    # Diese stammen aus Mesh.face_uvs und sind bereits richtig berechnet
                    if "uvs" in mesh_data and mesh_data["uvs"] is not None:
                        uv_coords = mesh_data["uvs"]
                    else:
                        # Fallback: berechne UVs automatisch (nur wenn nicht vorhanden)
                        tile_bounds = mesh_data.get("tile_bounds", None)
                        uv_offset = mesh_data.get("uv_offset", (0.0, 0.0))
                        uv_scale = mesh_data.get("uv_scale", (1.0, 1.0))
                        uv_coords = self._compute_uv_normalized(vertices, uv_offset, uv_scale, tile_bounds)
                    self._write_uv_source(f, uv_src_id, uv_coords)
                else:
                    uv_src_id = None

                # Vertices Element
                vert_elem_id = f"{mesh_id}_vertices_input"
                f.write(f'        <vertices id="{vert_elem_id}">\n')
                f.write(f'          <input semantic="POSITION" source="#{vert_src_id}"/>\n')
                f.write("        </vertices>\n")

                # Triangles (pro Material wenn faces ein Dict ist)
                if isinstance(faces, dict):
                    for mat_name, mat_faces in faces.items():
                        if len(mat_faces) > 0:
                            self._write_triangles(f, mat_name, mat_faces, vert_elem_id, normal_src_id, uv_src_id)
                else:
                    mat_name = mesh_data.get("material", "default")
                    if len(faces) > 0:
                        self._write_triangles(f, mat_name, faces, vert_elem_id, normal_src_id, uv_src_id)

                f.write("      </mesh>\n")
                f.write("    </geometry>\n")

            f.write("  </library_geometries>\n")

            # Visual Scene
            f.write("  <library_visual_scenes>\n")
            f.write('    <visual_scene id="Scene" name="Scene">\n')

            for mesh_data in meshes:
                mesh_id = mesh_data["id"]
                faces = mesh_data.get("faces", [])

                f.write(f'      <node id="{mesh_id}_node" name="{mesh_id}" type="NODE">\n')
                f.write(f'        <instance_geometry url="#{mesh_id}_geometry">\n')

                # Material Bindings
                if isinstance(faces, dict) and faces:
                    f.write("          <bind_material>\n")
                    f.write("            <technique_common>\n")
                    for mat_name in sorted(faces.keys()):
                        f.write(f'              <instance_material symbol="{mat_name}" target="#{mat_name}"/>\n')
                    f.write("            </technique_common>\n")
                    f.write("          </bind_material>\n")

                f.write("        </instance_geometry>\n")
                f.write("      </node>\n")

            f.write("    </visual_scene>\n")
            f.write("  </library_visual_scenes>\n")

            self._write_footer(f)

        return output_path

    def __repr__(self) -> str:
        return f"DAEExporter(level={self.level_name})"
