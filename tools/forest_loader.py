"""
Forest Layer Loader for dae_viewer

Loads forest.forest4.json and creates point instances for trees.
All trees are rendered in ONE actor (optimized for 30k+ instances).
"""

import json
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()
import numpy as np
import pyvista as pv
from pathlib import Path


# Color mapping for tree types
TREE_TYPE_COLORS = {
    "oak": (0.2, 0.6, 0.2),  # Dark green
    "birch": (0.8, 0.7, 0.5),  # Light brown
    "spruce": (0.1, 0.4, 0.1),  # Dark green
    "ash": (0.4, 0.6, 0.3),  # Medium green
    "pine": (0.15, 0.45, 0.15),  # Fir green
    "beech": (0.25, 0.55, 0.25),  # Dark green
    "default": (0.3, 0.8, 0.3),  # Light green
}


def load_forest_layer(viewer, forest_json_path: Path):
    """
    Load forest data from forest.forest4.json and add ALL trees to the scene as a single actor.

    Optimized for 30,000+ instances: all points in ONE PolyData mesh.

    Args:
        viewer: DAETileViewer instance
        forest_json_path: Path to forest.forest4.json
    """

    if not forest_json_path.exists():
        logger.error(f"  [!] Forest JSON not found: {forest_json_path}")
        return None

    try:
        with open(forest_json_path, "r", encoding="utf-8") as f:
            forest_data = json.load(f)
    except Exception as e:
        logger.error(f"  [!] Error loading the forest JSON: {e}")
        return None

    # Forest.json format: {"formatVersion": 1, "trees": [...]}
    if not forest_data or "trees" not in forest_data:
        logger.error(f"  [!] No trees in forest.forest4.json")
        return None

    tree_instances = forest_data["trees"]
    if not tree_instances:
        logger.error(f"  [!] Tree instances are empty")
        return None

    logger.info(f"  [Forest] Loading {len(tree_instances)} tree instances into ONE actor...")

    try:
        # Extract positions and types
        positions = []
        colors = []

        tree_types = {}
        for inst in tree_instances:
            pos = inst.get("pos")
            if pos and len(pos) >= 3:
                positions.append(pos[:3])

                # Determine color based on tree type
                tree_type = inst.get("type", "default")
                tree_types[tree_type] = tree_types.get(tree_type, 0) + 1
                color = TREE_TYPE_COLORS.get(tree_type.lower(), TREE_TYPE_COLORS["default"])
                colors.append(color)

        if not positions:
            logger.error(f"  [!] No valid positions found")
            return None

        positions_array = np.array(positions, dtype=np.float32)
        colors_array = np.array(colors, dtype=np.float32)

        # Create ONE single PolyData with ALL trees
        point_cloud = pv.PolyData(positions_array)

        # Add colors to the points
        point_cloud["Colors"] = colors_array

        # Render all trees in ONE actor!
        actor = viewer.plotter.add_mesh(
            point_cloud,
            scalars="Colors",
            point_size=5.0,
            opacity=0.85,
            label="Trees",
            render_points_as_spheres=True,
            rgb=True,  # colors are RGB
        )

        # Store actor reference for toggling
        viewer.forest_actors.append(actor)

        logger.info(f"  [✓] {len(tree_instances)} trees loaded into ONE actor")
        logger.info(f"      Types: {tree_types}")

        return actor

    except Exception as e:
        logger.error(f"  [!] Error processing the forest data: {e}")
        import traceback

        traceback.print_exc()
        return None
