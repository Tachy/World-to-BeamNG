"""
DAEExporter - central DAE/Collada export class.

Consolidates the DAE export for:
- Terrain (multi-tile meshes)
- Buildings (LoD2)
- Horizon (distant terrain)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..facade.material_names import DAE_EFFECT_COLORS


class DAEExporter:
    """
    Central class for DAE/Collada 1.4.1 export.

    Features:
    - Shared XML structure (asset, materials, effects, geometries)
    - Automatic UV generation
    - Material binding
    - Optimized NumPy integration
    - Integration with MaterialManager (no local material copies)
    """

    def __init__(self, material_manager: Optional["MaterialManager"] = None, level_name: str = "World_to_BeamNG"):
        """
        Initialize the DAE exporter.

        Args:
            material_manager: Reference to the MaterialManager (optional for compatibility)
            level_name: Name of the BeamNG level
        """
        self.material_manager = material_manager
        self.level_name = level_name
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def _write_header(self, f) -> None:
        """Write the DAE XML header and asset."""
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<COLLADA version="1.4.1" xmlns="http://www.collada.org/2005/11/COLLADASchema">\n')
        f.write("  <asset>\n")
        f.write(f"    <created>{self.timestamp}</created>\n")
        f.write(f"    <modified>{self.timestamp}</modified>\n")
        f.write('    <unit name="meter" meter="1"/>\n')
        f.write("    <up_axis>Z_UP</up_axis>\n")
        f.write("  </asset>\n")

    def _write_footer(self, f, scene_id: str = "Scene") -> None:
        """Write the DAE XML footer (scene)."""
        f.write("  <scene>\n")
        f.write(f'    <instance_visual_scene url="#{scene_id}"/>\n')
        f.write("  </scene>\n")
        f.write("</COLLADA>\n")

    def _write_image_library(self, f, material_textures: Dict[str, str]) -> None:
        """Write library_images with texture paths.

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
        """Write library_materials with effects."""
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
        Write library_effects.

        Args:
            f: File handle
            material_names: List of material names
            colors: Optional dict {mat_name: (r, g, b)} for diffuse colors
            material_textures: Optional dict {mat_name: texture_path} for textures
        """
        f.write("  <library_effects>\n")
        for mat_name in sorted(material_names):
            f.write(f'    <effect id="effect_{mat_name}">\n')
            f.write("      <profile_COMMON>\n")

            # If a texture is present: newparam + sampler2D
            if material_textures and mat_name in material_textures:
                surface_id = f"{mat_name}_surface"
                sampler_id = f"{mat_name}_sampler"

                # Surface (references the image)
                f.write(f'        <newparam sid="{surface_id}">\n')
                f.write('          <surface type="2D">\n')
                f.write(f"            <init_from>{mat_name}_image</init_from>\n")
                f.write("          </surface>\n")
                f.write("        </newparam>\n")

                # Sampler2D (references the surface)
                f.write(f'        <newparam sid="{sampler_id}">\n')
                f.write("          <sampler2D>\n")
                f.write(f"            <source>{surface_id}</source>\n")
                f.write("          </sampler2D>\n")
                f.write("        </newparam>\n")

            f.write('        <technique sid="common">\n')
            f.write("          <phong>\n")
            f.write("            <diffuse>\n")

            # Texture or color
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
        Write <source> for vertices (XYZ).

        Args:
            f: File handle
            source_id: ID for <source>
            vertices: (N, 3) NumPy array
        """
        f.write(f'        <source id="{source_id}">')
        f.write(f'\n          <float_array id="{source_id}_array" count="{len(vertices) * 3}">')

        # All vertex values on one line
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
        Write <source> for UV coordinates.

        Args:
            f: File handle
            source_id: ID for <source>
            uv_coords: (N, 2) NumPy array
        """
        f.write(f'        <source id="{source_id}">')
        f.write(f'\n          <float_array id="{source_id}_array" count="{len(uv_coords) * 2}">')

        # All UV values on one line
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
        Compute smooth normals (averaged from adjacent face normals).

        Args:
            vertices: (N, 3) NumPy array of the vertex positions
            faces: List of face indices or (M, 3) array

        Returns:
            (N, 3) array with vertex normals (normalized)
        """
        if not faces:
            # If there are no faces, return default normals
            return np.array([[0, 0, 1]] * len(vertices), dtype=np.float32)

        faces = np.array(faces)

        # Initialize the normal accumulator
        vertex_normals = np.zeros_like(vertices)

        # Compute face normals (cross product)
        v0 = vertices[faces[:, 0]]  # First vertex of each triangle
        v1 = vertices[faces[:, 1]]  # Second vertex
        v2 = vertices[faces[:, 2]]  # Third vertex

        # Edges
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Face normals (not normalized - the face area acts as weight)
        face_normals = np.cross(edge1, edge2)

        # Add the face normal to every participating vertex
        for i, face_idx_set in enumerate(faces):
            for vertex_idx in face_idx_set:
                vertex_normals[vertex_idx] += face_normals[i]

        # Normalize all vertex normals
        # Compute lengths
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        # Prevent division by zero
        lengths[lengths == 0] = 1.0
        # Normalize
        vertex_normals = vertex_normals / lengths

        return vertex_normals

    def _write_normals_source(self, f, source_id: str, normals: np.ndarray) -> None:
        """
        Write <source> for normals (XYZ).

        Args:
            f: File handle
            source_id: ID for <source>
            normals: (N, 3) NumPy array
        """
        f.write(f'        <source id="{source_id}">')
        f.write(f'\n          <float_array id="{source_id}_array" count="{len(normals) * 3}">')

        # All normal values on one line
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

    def _write_triangles(
        self,
        f,
        material_name: str,
        faces: List[Tuple[int, int, int]],
        vertices_id: str,
        normal_id: Optional[str] = None,
        uv_id: Optional[str] = None,
        uv_indices: Optional[List[Tuple[int, int, int]]] = None,  # NEW: separate UV indices
    ) -> None:
        """
        Write a <triangles> block with optional normals and UVs.

        IMPORTANT: Terrain tiles vs. road materials use different TEXCOORD semantics:
        - Terrain tiles (tile_*): semantic="TEXCOORD" (original mapping)
        - Road materials: semantic="TEXCOORD0" (for PBR shader support)

        Args:
            f: File handle
            material_name: Material symbol
            faces: List of (v0, v1, v2) face indices
            vertices_id: ID of the <vertices> element
            normal_id: Optional ID of the normals <source>
            uv_id: Optional ID of the UV <source>
            uv_indices: Optional separate UV indices (list of (uv0, uv1, uv2) tuples)
        """
        f.write(f'        <triangles material="{material_name}" count="{len(faces)}">\n')
        f.write(f'          <input semantic="VERTEX" source="#{vertices_id}" offset="0"/>\n')

        offset = 1
        if normal_id:
            f.write(f'          <input semantic="NORMAL" source="#{normal_id}" offset="{offset}"/>\n')
            offset += 1

        if uv_id:
            # TEXCOORD (BeamNG default for all materials)
            f.write(f'          <input semantic="TEXCOORD" source="#{uv_id}" offset="{offset}" set="0"/>\n')
            offset += 1

        f.write("          <p>")

        # All indices on one line
        if normal_id and uv_id:
            # With normals + UV: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2
            if uv_indices:
                # Separate UV indices present (e.g. for roads)
                indices_str = " ".join(
                    f"{face[0]} {face[0]} {uv_ids[0]} {face[1]} {face[1]} {uv_ids[1]} {face[2]} {face[2]} {uv_ids[2]}"
                    for face, uv_ids in zip(faces, uv_indices)
                )
            else:
                # 1:1 mapping (e.g. for terrain)
                indices_str = " ".join(
                    f"{face[0]} {face[0]} {face[0]} {face[1]} {face[1]} {face[1]} {face[2]} {face[2]} {face[2]}"
                    for face in faces
                )
        elif normal_id:
            # With normals only: v0 n0 v1 n1 v2 n2
            indices_str = " ".join(f"{face[0]} {face[0]} {face[1]} {face[1]} {face[2]} {face[2]}" for face in faces)
        elif uv_id:
            # With UV only: v0 uv0 v1 uv1 v2 uv2
            if uv_indices:
                indices_str = " ".join(
                    f"{face[0]} {uv_ids[0]} {face[1]} {uv_ids[1]} {face[2]} {uv_ids[2]}"
                    for face, uv_ids in zip(faces, uv_indices)
                )
            else:
                indices_str = " ".join(f"{face[0]} {face[0]} {face[1]} {face[1]} {face[2]} {face[2]}" for face in faces)
        else:
            # Without UV/normals: v0 v1 v2
            indices_str = " ".join(f"{face[0]} {face[1]} {face[2]}" for face in faces)

        f.write(f"\n{indices_str}")
        f.write("\n          </p>\n")
        f.write("        </triangles>\n")

    def export_multi_mesh(
        self,
        output_path: str,
        meshes: List[Dict[str, Any]],
        with_uv: bool = False,
        material_textures: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Export a multi-mesh DAE (e.g. terrain tiles, buildings).

        Args:
            output_path: Target file path
            meshes: List of dicts with:
                - 'id': Mesh ID
                - 'vertices': (N, 3) NumPy array
                - 'faces': List of (v0, v1, v2) or dict {mat_name: faces_list}
                - 'material' (optional): Material name (if faces is a list)
                - 'uv_offset' (optional): UV offset
                - 'uv_scale' (optional): UV scale
            with_uv: Generate UV coordinates?
            material_textures: Optional dict {mat_name: texture_path} for texture bindings

        Returns:
            output_path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Collect all material names
        material_names = set()
        for mesh_data in meshes:
            faces = mesh_data.get("faces", [])
            if isinstance(faces, dict):
                material_names.update(faces.keys())
            elif "material" in mesh_data:
                material_names.add(mesh_data["material"])

        with open(output_path, "w", encoding="utf-8") as f:
            self._write_header(f)

            # Images (textures)
            if material_textures:
                self._write_image_library(f, material_textures)

            # Materials
            self._write_material_library(f, list(material_names))

            # Colors for buildings (wall/roof)
            colors = {name: color for name, color in DAE_EFFECT_COLORS.items() if name in material_names}

            self._write_effect_library(f, list(material_names), colors, material_textures)

            # Geometries
            f.write("  <library_geometries>\n")

            for mesh_data in meshes:
                mesh_id = mesh_data["id"]
                vertices = mesh_data["vertices"]
                faces = mesh_data.get("faces", [])

                f.write(f'    <geometry id="{mesh_id}_geometry" name="{mesh_id}">\n')
                f.write("      <mesh>\n")

                # Vertices Source
                vert_src_id = f"{mesh_id}_vertices"
                self._write_vertices_source(f, vert_src_id, vertices)

                # Normals source (NEW: smooth normals for BeamNG)
                normal_src_id = f"{mesh_id}_normals"
                provided_normals = mesh_data.get("normals")
                if provided_normals is not None:
                    smooth_normals = provided_normals
                else:
                    if isinstance(faces, dict):
                        # Combine all faces of all materials for the normal computation
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
                    # First check whether explicit UVs are present in mesh_data
                    # UVs must be present - from mesh_data["global_uvs"]
                    if "uvs" in mesh_data and mesh_data["uvs"] is not None:
                        uv_coords = mesh_data["uvs"]
                    else:
                        raise ValueError(f"Tile {mesh_id} has with_uv=True but no UVs in mesh_data!")
                    self._write_uv_source(f, uv_src_id, uv_coords)
                else:
                    uv_src_id = None

                # Vertices Element
                vert_elem_id = f"{mesh_id}_vertices_input"
                f.write(f'        <vertices id="{vert_elem_id}">\n')
                f.write(f'          <input semantic="POSITION" source="#{vert_src_id}"/>\n')
                f.write("        </vertices>\n")

                # Triangles (per material if faces is a dict)
                uv_indices_dict = mesh_data.get("uv_indices", {})  # NEW: get UV indices

                if isinstance(faces, dict):
                    for mat_name, mat_faces in faces.items():
                        if len(mat_faces) > 0:
                            # Get the UV indices for this material (if present)
                            mat_uv_indices = uv_indices_dict.get(mat_name, None)
                            self._write_triangles(
                                f, mat_name, mat_faces, vert_elem_id, normal_src_id, uv_src_id, mat_uv_indices
                            )
                else:
                    mat_name = mesh_data.get("material", "default")
                    if len(faces) > 0:
                        mat_uv_indices = (
                            uv_indices_dict.get(mat_name, None) if isinstance(uv_indices_dict, dict) else None
                        )
                        self._write_triangles(
                            f, mat_name, faces, vert_elem_id, normal_src_id, uv_src_id, mat_uv_indices
                        )

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
