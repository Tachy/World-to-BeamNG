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
    
    def _write_material_library(self, f, material_names: List[str]) -> None:
        """Schreibe library_materials mit Effects."""
        f.write("  <library_materials>\n")
        for mat_name in sorted(material_names):
            f.write(f'    <material id="{mat_name}" name="{mat_name}">\n')
            f.write(f'      <instance_effect url="#effect_{mat_name}"/>\n')
            f.write("    </material>\n")
        f.write("  </library_materials>\n")
    
    def _write_effect_library(self, f, material_names: List[str], colors: Optional[Dict[str, Tuple[float, float, float]]] = None) -> None:
        """
        Schreibe library_effects.
        
        Args:
            f: File handle
            material_names: Liste von Material-Namen
            colors: Optional Dict {mat_name: (r, g, b)} für diffuse colors
        """
        f.write("  <library_effects>\n")
        for mat_name in sorted(material_names):
            f.write(f'    <effect id="effect_{mat_name}">\n')
            f.write("      <profile_COMMON>\n")
            f.write('        <technique sid="common">\n')
            f.write("          <phong>\n")
            f.write("            <diffuse>\n")
            
            # Color aus Dict oder Fallback grau
            if colors and mat_name in colors:
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
    
    def _write_vertices_source(
        self, 
        f, 
        source_id: str, 
        vertices: np.ndarray
    ) -> None:
        """
        Schreibe <source> für Vertices (XYZ).
        
        Args:
            f: File handle
            source_id: ID für <source>
            vertices: (N, 3) NumPy Array
        """
        f.write(f'        <source id="{source_id}">\n')
        f.write(f'          <float_array id="{source_id}_array" count="{len(vertices) * 3}">')
        
        # Optimiert: Batch-Write
        for v in vertices:
            f.write(f"\n{v[0]:.2f} {v[1]:.2f} {v[2]:.2f}")
        
        f.write("\n          </float_array>\n")
        f.write("          <technique_common>\n")
        f.write(f'            <accessor source="#{source_id}_array" count="{len(vertices)}" stride="3">\n')
        f.write('              <param name="X" type="float"/>\n')
        f.write('              <param name="Y" type="float"/>\n')
        f.write('              <param name="Z" type="float"/>\n')
        f.write("            </accessor>\n")
        f.write("          </technique_common>\n")
        f.write("        </source>\n")
    
    def _write_uv_source(
        self,
        f,
        source_id: str,
        uv_coords: np.ndarray
    ) -> None:
        """
        Schreibe <source> für UV-Koordinaten.
        
        Args:
            f: File handle
            source_id: ID für <source>
            uv_coords: (N, 2) NumPy Array
        """
        f.write(f'        <source id="{source_id}">\n')
        f.write(f'          <float_array id="{source_id}_array" count="{len(uv_coords) * 2}">')
        
        for uv in uv_coords:
            f.write(f"\n{uv[0]:.6f} {uv[1]:.6f}")
        
        f.write("\n          </float_array>\n")
        f.write("          <technique_common>\n")
        f.write(f'            <accessor source="#{source_id}_array" count="{len(uv_coords)}" stride="2">\n')
        f.write('              <param name="S" type="float"/>\n')
        f.write('              <param name="T" type="float"/>\n')
        f.write("            </accessor>\n")
        f.write("          </technique_common>\n")
        f.write("        </source>\n")
    
    def _compute_uv_normalized(
        self,
        vertices: np.ndarray,
        uv_offset: Tuple[float, float] = (0.0, 0.0),
        uv_scale: Tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        """
        Berechne UV-Koordinaten (normalisiert auf Mesh-Bounds).
        
        Args:
            vertices: (N, 3) Vertices
            uv_offset: (u_offset, v_offset)
            uv_scale: (u_scale, v_scale)
        
        Returns:
            (N, 2) UV-Koordinaten
        """
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        
        # Normalisiere auf 0..1
        u = (vertices[:, 0] - x_min) / (x_max - x_min) if x_max > x_min else 0.0
        v = (vertices[:, 1] - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        
        # Apply offset und scale
        u = uv_offset[0] + u * uv_scale[0]
        v = uv_offset[1] + v * uv_scale[1]
        
        return np.column_stack([u, v])
    
    def _write_triangles(
        self,
        f,
        material_name: str,
        faces: List[Tuple[int, int, int]],
        vertices_id: str,
        uv_id: Optional[str] = None
    ) -> None:
        """
        Schreibe <triangles> Block.
        
        Args:
            f: File handle
            material_name: Material-Symbol
            faces: Liste von (v0, v1, v2) Face-Indizes
            vertices_id: ID des <vertices> Elements
            uv_id: Optional ID der UV <source>
        """
        f.write(f'        <triangles material="{material_name}" count="{len(faces)}">\n')
        f.write(f'          <input semantic="VERTEX" source="#{vertices_id}" offset="0"/>\n')
        
        if uv_id:
            f.write(f'          <input semantic="TEXCOORD" source="#{uv_id}" offset="1" set="0"/>\n')
        
        f.write("          <p>")
        
        # Schreibe Indizes
        for face in faces:
            if uv_id:
                # Mit UV: v0 uv0 v1 uv1 v2 uv2
                f.write(f"\n{face[0]} {face[0]} {face[1]} {face[1]} {face[2]} {face[2]}")
            else:
                # Ohne UV: v0 v1 v2
                f.write(f"\n{face[0]} {face[1]} {face[2]}")
        
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
        uv_scale: Tuple[float, float] = (1.0, 1.0)
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
            self._write_effect_library(f, [material_name])
            
            # Geometries
            f.write("  <library_geometries>\n")
            f.write(f'    <geometry id="{mesh_id}_geometry" name="{mesh_id}">\n')
            f.write("      <mesh>\n")
            
            # Vertices Source
            vert_src_id = f"{mesh_id}_vertices"
            self._write_vertices_source(f, vert_src_id, vertices)
            
            # UV Source (optional)
            if with_uv:
                uv_src_id = f"{mesh_id}_uvs"
                uv_coords = self._compute_uv_normalized(vertices, uv_offset, uv_scale)
                self._write_uv_source(f, uv_src_id, uv_coords)
            else:
                uv_src_id = None
            
            # Vertices Element
            vert_elem_id = f"{mesh_id}_vertices_input"
            f.write(f'        <vertices id="{vert_elem_id}">\n')
            f.write(f'          <input semantic="POSITION" source="#{vert_src_id}"/>\n')
            f.write("        </vertices>\n")
            
            # Triangles
            self._write_triangles(f, material_name, faces, vert_elem_id, uv_src_id)
            
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
        with_uv: bool = False
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
            
            # Materials
            self._write_material_library(f, list(material_names))
            
            # Colors für Buildings (wall/roof)
            colors = {}
            if "lod2_wall_white" in material_names:
                colors["lod2_wall_white"] = (0.95, 0.95, 0.95)
            if "lod2_roof_red" in material_names:
                colors["lod2_roof_red"] = (0.6, 0.2, 0.1)
            
            self._write_effect_library(f, list(material_names), colors)
            
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
                
                # UV Source (optional)
                if with_uv:
                    uv_src_id = f"{mesh_id}_uvs"
                    uv_coords = self._compute_uv_normalized(vertices, uv_offset, uv_scale)
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
                            self._write_triangles(f, mat_name, mat_faces, vert_elem_id, uv_src_id)
                else:
                    mat_name = mesh_data.get("material", "default")
                    if len(faces) > 0:
                        self._write_triangles(f, mat_name, faces, vert_elem_id, uv_src_id)
                
                f.write("      </mesh>\n")
                f.write("    </geometry>\n")
            
            f.write("  </library_geometries>\n")
            
            # Visual Scene
            f.write("  <library_visual_scenes>\n")
            f.write('    <visual_scene id="Scene" name="Scene">\n')
            
            for mesh_data in meshes:
                mesh_id = mesh_data["id"]
                f.write(f'      <node id="{mesh_id}_node" name="{mesh_id}" type="NODE">\n')
                f.write(f'        <instance_geometry url="#{mesh_id}_geometry"/>\n')
                f.write("      </node>\n")
            
            f.write("    </visual_scene>\n")
            f.write("  </library_visual_scenes>\n")
            
            self._write_footer(f)
        
        return output_path
    
    def __repr__(self) -> str:
        return f"DAEExporter(level={self.level_name})"
