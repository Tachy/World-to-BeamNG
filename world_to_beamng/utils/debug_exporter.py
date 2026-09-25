"""
Debug Network Exporter - collects and exports debug visualization data.

Singleton pattern for collecting debug data during the workflow:
- Junctions (positions, connections)
- Roads (centerlines, metadata)
- Boundary polygons (stitching visualization)
- Component lines (connected components from stitching)
- Universal primitives (labels, circles, polygons, lines, points, arrows, vectors)
- Grid colors (for the viewer)

Usage:
    exporter = DebugNetworkExporter.get_instance()

    # Universal primitives:
    exporter.add_label("Debug Label", position=(100, 100, 0))
    exporter.add_circle(50, center=(100, 100, 0))
    exporter.add_polygon([(0, 0, 0), (10, 0, 0), (10, 10, 0)])
    exporter.add_line([(0, 0, 0), (100, 100, 0)])
    exporter.add_point((100, 100, 0))
    exporter.add_arrow((0, 0, 0), (100, 0, 0))
    exporter.add_vector((50, 50, 0), (1, 0, 0), scale=20)

    exporter.export(cache_dir)
"""

import json
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Utility functions for coordinate handling
# ============================================================================


def _normalize_coordinate(coord: Union[Tuple, List, np.ndarray]) -> List[float]:
    """
    Convert various coordinate formats to a list [x, y, z].

    Args:
        coord: tuple, list or NumPy array with 3 coordinates

    Returns:
        [x, y, z] as a list of floats
    """
    if hasattr(coord, "tolist"):
        coord = coord.tolist()
    elif not isinstance(coord, (list, tuple)):
        raise TypeError(f"Coordinate must be a tuple/list/array, not {type(coord)}")

    if len(coord) != 3:
        raise ValueError(f"Coordinate must have 3 components, not {len(coord)}")

    return [float(c) for c in coord]


def _normalize_coordinates(coords: Union[List, np.ndarray]) -> List[List[float]]:
    """
    Convert a list of coordinates to [[x, y, z], ...].

    Args:
        coords: list of tuples/lists/arrays or a NumPy 2D array

    Returns:
        list of [x, y, z] lists
    """
    if hasattr(coords, "tolist"):
        # NumPy array
        coords = coords.tolist()

    if not isinstance(coords, (list, tuple)):
        raise TypeError(f"Coordinates must be a list, not {type(coords)}")

    result = []
    for coord in coords:
        result.append(_normalize_coordinate(coord))

    return result


def _get_default_color(color_type: str = "standard") -> List[float]:
    """
    Return the default color for a type.

    Args:
        color_type: "standard" (blue), "positive" (green), "negative" (red), etc.

    Returns:
        RGB color as [r, g, b] with values 0.0-1.0
    """
    colors = {
        "standard": [0.0, 0.0, 1.0],  # Blue
        "positive": [0.2, 0.8, 0.2],  # Green
        "negative": [1.0, 0.2, 0.2],  # Red
        "warning": [1.0, 0.8, 0.0],  # Yellow
        "neutral": [0.7, 0.7, 0.7],  # Gray
        "highlight": [1.0, 0.0, 1.0],  # Magenta
        "outline": [0.0, 0.0, 0.0],  # Black
    }
    return colors.get(color_type, colors["standard"])


class DebugNetworkExporter:
    """Collects debug data for visualization in the DAE viewer (singleton)."""

    _instance: Optional["DebugNetworkExporter"] = None

    def __init__(self):
        """Private constructor - use get_instance() instead."""
        if DebugNetworkExporter._instance is not None:
            raise RuntimeError("DebugNetworkExporter is a singleton - use get_instance()")

        self.primitives: List[Dict[str, Any]] = []  # labels, circles, polygons, lines, etc.

        # Grid colors for the viewer (defaults)
        self.grid_colors = self._get_default_grid_colors()

    @staticmethod
    def _get_default_grid_colors() -> Dict[str, Any]:
        """Return the default grid colors."""
        return {
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
        """Get the singleton instance (creates it if needed)."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.primitives = []
            cls._instance.grid_colors = cls._get_default_grid_colors()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for a new export run)."""
        cls._instance = None

    # ========================================================================
    # UNIVERSAL PRIMITIVES - The heart of the new system
    # ========================================================================

    def add_line(
        self,
        coords: Union[List, np.ndarray],
        color: Optional[Union[List, Tuple]] = None,
        width: float = 2.0,
        label: Optional[str] = None,
    ) -> None:
        """
        Add a line.

        Args:
            coords: list of (x, y, z) coordinates
            color: RGB color [r, g, b] 0.0-1.0, default: blue
            width: line width in pixels
        """
        if color is None:
            color = _get_default_color("standard")

        primitive = {
            "type": "line",
            "coords": _normalize_coordinates(coords),
            "color": list(color),
            "line_width": float(width),
            "opacity": 1.0,
        }
        if label:
            primitive["label"] = str(label)
        self.primitives.append(primitive)

    # ========================================================================
    # BATCH & UTILITY METHODS
    # ========================================================================

    def export(self, cache_dir: str, filename: str = "debug_network.json") -> None:
        """
        Export the collected debug data to a JSON file.

        Args:
            cache_dir: target directory for the export
            filename: file name (default: debug_network.json)
        """
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(cache_dir) / filename

        data = {
            "primitives": self.primitives,
            "grid_colors": self.grid_colors,
        }

        # Without indent: json.dump with indent uses the slow Python encoder (several seconds for a ~10 MB debug
        # network), the file is only read by machine (tools/dae_viewer.py).
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data))

        logger.debug(f"  [Debug] Exported: {len(self.primitives)} primitives")
        logger.debug(f"  [Debug] File: {output_path}")

    def clear(self) -> None:
        """Delete all collected data."""
        self.primitives.clear()

    def __repr__(self) -> str:
        return f"DebugNetworkExporter(primitives={len(self.primitives)})"
