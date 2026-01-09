"""
Debug Network Exporter - Sammelt und exportiert Debug-Visualisierungsdaten.

Singleton-Pattern zum Sammeln von Debug-Daten während des Workflows:
- Junctions (Positionen, Verbindungen)
- Roads (Centerlines, Metadaten)
- Boundary-Polygone (Stitching-Visualisierung)
- Component-Linien (Connected Components aus Stitching)
- Grid-Farben (für Viewer)

Usage:
    exporter = DebugNetworkExporter.get_instance()
    exporter.add_road(road_data)
    exporter.add_junction(junction_data)
    exporter.add_component_line(coords, color, label)
    exporter.export(cache_dir)
"""

import json
import os
from typing import Dict, List, Optional, Any


class DebugNetworkExporter:
    """Sammelt Debug-Daten für Visualisierung im DAE Viewer (Singleton)."""

    _instance: Optional["DebugNetworkExporter"] = None

    def __init__(self):
        """Private Constructor - verwende get_instance() stattdessen."""
        if DebugNetworkExporter._instance is not None:
            raise RuntimeError("DebugNetworkExporter ist ein Singleton - verwende get_instance()")

        self.roads: List[Dict[str, Any]] = []
        self.junctions: List[Dict[str, Any]] = []
        self.boundary_polygons: List[Dict[str, Any]] = []
        self.component_lines: List[Dict[str, Any]] = []

        # Grid-Farben für Viewer (standardmäßig)
        self.grid_colors = {
            "terrain": {
                "face": [0.8, 0.95, 0.8],
                "edge": [0.2, 0.5, 0.2],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "road": {
                "face": [1.0, 1.0, 1.0],
                "edge": [1.0, 0.0, 0.0],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "building_wall": {
                "face": [0.95, 0.95, 0.95],
                "edge": [0.3, 0.3, 0.3],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "building_roof": {
                "face": [0.6, 0.2, 0.1],
                "edge": [0.3, 0.1, 0.05],
                "face_opacity": 0.5,
                "edge_opacity": 1.0,
            },
            "junction": {
                "color": [0.0, 0.0, 1.0],
                "opacity": 0.5,
            },
            "centerline": {
                "color": [0.0, 0.0, 1.0],
                "line_width": 2.0,
                "opacity": 1.0,
            },
            "boundary": {
                "color": [1.0, 0.0, 1.0],
                "line_width": 2.0,
                "opacity": 1.0,
            },
            "component_terrain": {
                "color": [0.2, 0.8, 0.2],
                "line_width": 3.0,
                "opacity": 1.0,
            },
            "component_road": {
                "color": [0.8, 0.2, 0.2],
                "line_width": 3.0,
                "opacity": 1.0,
            },
        }

    @classmethod
    def get_instance(cls) -> "DebugNetworkExporter":
        """Hole die Singleton-Instanz (erstellt sie bei Bedarf)."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.roads = []
            cls._instance.junctions = []
            cls._instance.boundary_polygons = []
            cls._instance.component_lines = []
            cls._instance.grid_colors = {
                "terrain": {
                    "face": [0.8, 0.95, 0.8],
                    "edge": [0.2, 0.5, 0.2],
                    "face_opacity": 0.5,
                    "edge_opacity": 1.0,
                },
                "road": {
                    "face": [1.0, 1.0, 1.0],
                    "edge": [1.0, 0.0, 0.0],
                    "face_opacity": 0.5,
                    "edge_opacity": 1.0,
                },
                "building_wall": {
                    "face": [0.95, 0.95, 0.95],
                    "edge": [0.3, 0.3, 0.3],
                    "face_opacity": 0.5,
                    "edge_opacity": 1.0,
                },
                "building_roof": {
                    "face": [0.6, 0.2, 0.1],
                    "edge": [0.3, 0.1, 0.05],
                    "face_opacity": 0.5,
                    "edge_opacity": 1.0,
                },
                "junction": {
                    "color": [0.0, 0.0, 1.0],
                    "opacity": 0.5,
                },
                "centerline": {
                    "color": [0.0, 0.0, 1.0],
                    "line_width": 2.0,
                    "opacity": 1.0,
                },
                "boundary": {
                    "color": [1.0, 0.0, 1.0],
                    "line_width": 2.0,
                    "opacity": 1.0,
                },
                "component_terrain": {
                    "color": [0.2, 0.8, 0.2],
                    "line_width": 3.0,
                    "opacity": 1.0,
                },
                "component_road": {
                    "color": [0.8, 0.2, 0.2],
                    "line_width": 3.0,
                    "opacity": 1.0,
                },
            }
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Setze Singleton-Instanz zurück (für neuen Export-Lauf)."""
        cls._instance = None

    def add_road(self, road_data: Dict[str, Any]) -> None:
        """
        Füge eine Road zur Debug-Visualisierung hinzu.

        Args:
            road_data: Dict mit Road-Informationen:
                - road_id: Eindeutige Road-ID
                - coords: Liste von (x, y, z) Centerline-Punkten
                - num_points: Anzahl der Punkte
                - junction_start_id: Junction am Anfang (optional)
                - junction_end_id: Junction am Ende (optional)
                - junction_buffer_start: Buffer-Distanz am Start (optional)
                - junction_buffer_end: Buffer-Distanz am Ende (optional)
        """
        # Konvertiere NumPy Arrays zu Listen für JSON-Serialisierung
        road_copy = road_data.copy()
        if "coords" in road_copy:
            coords = road_copy["coords"]
            if hasattr(coords, "tolist"):  # NumPy Array
                road_copy["coords"] = coords.tolist()
            elif isinstance(coords, list) and len(coords) > 0:
                # Prüfe ob Elemente NumPy Arrays sind
                if hasattr(coords[0], "tolist"):
                    road_copy["coords"] = [c.tolist() if hasattr(c, "tolist") else list(c) for c in coords]

        # Füge Rendering-Eigenschaften hinzu
        road_copy.setdefault("color", [0.0, 0.0, 1.0])
        road_copy.setdefault("line_width", 2.0)
        road_copy.setdefault("opacity", 1.0)

        self.roads.append(road_copy)

    def add_junction(self, junction_data: Dict[str, Any]) -> None:
        """
        Füge eine Junction zur Debug-Visualisierung hinzu.

        Args:
            junction_data: Dict mit Junction-Informationen:
                - position: (x, y, z) Junction-Koordinaten
                - road_indices: Liste von Road-Indizes die hier verbunden sind
                - connection_types: Dict mapping road_idx -> ["start"/"end"]
        """
        # Konvertiere NumPy Arrays zu Listen
        junction_copy = junction_data.copy()
        if "position" in junction_copy:
            pos = junction_copy["position"]
            if hasattr(pos, "tolist"):
                junction_copy["position"] = pos.tolist()

        self.junctions.append(junction_copy)

    def add_boundary(self, boundary_data: Dict[str, Any]) -> None:
        """
        Füge ein Boundary-Polygon zur Debug-Visualisierung hinzu.

        Args:
            boundary_data: Dict mit Boundary-Informationen:
                - type: "boundary_polygon" oder "search_circle"
                - coords: Liste von (x, y, z) Polygon-Punkten
                - centerline_point: (x, y, z) Centerline-Sample-Punkt
                - search_radius: Suchradius in Metern
        """
        # Konvertiere NumPy Arrays zu Listen
        boundary_copy = boundary_data.copy()
        if "coords" in boundary_copy:
            coords = boundary_copy["coords"]
            if hasattr(coords, "tolist"):
                boundary_copy["coords"] = coords.tolist()
            elif isinstance(coords, list) and len(coords) > 0:
                if hasattr(coords[0], "tolist"):
                    boundary_copy["coords"] = [c.tolist() if hasattr(c, "tolist") else list(c) for c in coords]

        if "centerline_point" in boundary_copy:
            pt = boundary_copy["centerline_point"]
            if hasattr(pt, "tolist"):
                boundary_copy["centerline_point"] = pt.tolist()

        self.boundary_polygons.append(boundary_copy)

    def add_component_line(self, coords, color=None, label=None, line_width=3.0):
        """
        Füge eine Connected Component Linie hinzu (z.B. Terrain-Kante oder Straßen-Kante).

        Args:
            coords: Liste von (x, y, z) Koordinaten
            color: RGB-Farbe [r, g, b] (0.0-1.0), default: grün
            label: Optionales Label (z.B. "terrain", "road")
            line_width: Linienbreite in Pixeln
        """
        # Konvertiere NumPy Arrays zu Listen
        if hasattr(coords, "tolist"):
            coords = coords.tolist()
        elif isinstance(coords, list) and len(coords) > 0:
            if hasattr(coords[0], "tolist"):
                coords = [c.tolist() if hasattr(c, "tolist") else list(c) for c in coords]

        # Default-Farbe: Grün
        if color is None:
            color = [0.2, 0.8, 0.2]

        component_data = {
            "type": "component_line",
            "coords": [[float(c[0]), float(c[1]), float(c[2])] for c in coords],
            "color": color,
            "line_width": line_width,
            "opacity": 1.0,
        }

        if label:
            component_data["label"] = label

        self.component_lines.append(component_data)

    def add_roads_batch(self, roads: List[Dict[str, Any]]) -> None:
        """Füge mehrere Roads auf einmal hinzu."""
        for road in roads:
            self.add_road(road)

    def add_junctions_batch(self, junctions: List[Dict[str, Any]]) -> None:
        """Füge mehrere Junctions auf einmal hinzu."""
        for junction in junctions:
            self.add_junction(junction)

    def add_boundaries_batch(self, boundaries: List[Dict[str, Any]]) -> None:
        """Füge mehrere Boundary-Polygone auf einmal hinzu."""
        for boundary in boundaries:
            self.add_boundary(boundary)

    def merge(self, other: "DebugNetworkExporter") -> None:
        """
        Merge Daten aus einem anderen Exporter (für Multi-Tile Support).

        Args:
            other: Anderer DebugNetworkExporter
        """
        road_offset = len(self.roads)
        junction_offset = len(self.junctions)

        # Roads direkt hinzufügen
        self.roads.extend(other.roads)

        # Junctions mit angepassten road_indices hinzufügen
        for junction in other.junctions:
            junction_copy = junction.copy()
            # Passe road_indices an
            if "road_indices" in junction_copy:
                junction_copy["road_indices"] = [idx + road_offset for idx in junction_copy["road_indices"]]
            # Passe connection_types an
            if "connection_types" in junction_copy:
                new_conn_types = {}
                for idx, types in junction_copy["connection_types"].items():
                    new_conn_types[int(idx) + road_offset] = types
                junction_copy["connection_types"] = new_conn_types
            self.junctions.append(junction_copy)

        # Boundary-Polygone direkt hinzufügen
        self.boundary_polygons.extend(other.boundary_polygons)

        # Component-Linien direkt hinzufügen
        self.component_lines.extend(other.component_lines)

    def export(self, cache_dir: str, filename: str = "debug_network.json") -> None:
        """
        Exportiere gesammelte Debug-Daten in JSON-Datei.

        Args:
            cache_dir: Zielverzeichnis für Export
            filename: Dateiname (default: debug_network.json)
        """
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, filename)

        data = {
            "roads": self.roads,
            "junctions": self.junctions,
            "boundary_polygons": self.boundary_polygons,
            "component_lines": self.component_lines,
            "grid_colors": self.grid_colors,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(
            f"  [Debug] Exportiert: {len(self.roads)} Roads, {len(self.junctions)} Junctions, "
            f"{len(self.boundary_polygons)} Boundaries, {len(self.component_lines)} Component-Linien"
        )
        print(f"  [Debug] Datei: {output_path}")

    def clear(self) -> None:
        """Lösche alle gesammelten Daten."""
        self.roads.clear()
        self.junctions.clear()
        self.boundary_polygons.clear()
        self.component_lines.clear()

    def __repr__(self) -> str:
        return (
            f"DebugNetworkExporter(roads={len(self.roads)}, "
            f"junctions={len(self.junctions)}, "
            f"boundaries={len(self.boundary_polygons)}, "
            f"components={len(self.component_lines)})"
        )
