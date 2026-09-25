"""
Forest Instance Generator: creates the final tree instances.

Generates complete tree instances from (x, y, z) positions with:
- Tree type (based on tree_distribution)
- Rotation (quaternion around the Z axis)
- Scale (from the average_height range)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ForestInstanceGenerator:
    """
    Generates final tree instances with type, rotation and scale.

    Format per instance (BeamNG .forest4.json schema):
    {
        "type": "oak",
        "pos": [x, y, z],
        "rotationMatrix": [r00, r01, r02, r10, r11, r12, r20, r21, r22],  # 3x3 rotation matrix
        "scale": 1.15
    }
    """

    def __init__(self, registered_trees: Optional[Dict] = None):
        """
        Args:
            registered_trees: optional - dict of available tree species (from AssetScanner)
        """
        self.registered_trees = registered_trees or {}

    def generate_instances(
        self, points_3d: List[Tuple[float, float, float]], forest_type: str, forest_properties: Dict
    ) -> List[Dict]:
        """
        Generate tree instances for a forest polygon.

        Args:
            points_3d: list of (x, y, z) positions
            forest_type: forest type (e.g. "deciduous_dense")
            forest_properties: properties from forest_types (tree_distribution, average_height, etc.)

        Returns:
            List of instance dicts
        """
        if not points_3d:
            return []

        instances = []

        # Tree distribution (percentage shares) - from preferred_trees
        tree_distribution = forest_properties.get("preferred_trees", {})
        if not tree_distribution:
            logger.warning(f"No preferred_trees for {forest_type}, skipping")
            return []

        # Average Height Range
        avg_height_range = forest_properties.get("average_height", [15.0, 25.0])
        min_height = avg_height_range[0]
        max_height = avg_height_range[1]

        # Choose tree types for all points
        tree_types = self._select_tree_types(len(points_3d), tree_distribution)

        # Generate instances
        for i, (x, y, z) in enumerate(points_3d):
            tree_type = tree_types[i]

            # Rotation (random around the Z axis)
            rotation_matrix = self._generate_rotation_matrix()

            # Scale (based on average_height)
            scale = self._generate_scale(min_height, max_height)

            instance = {
                "type": tree_type,
                "pos": [float(x), float(y), float(z)],
                "rotationMatrix": rotation_matrix,
                "scale": float(scale),
            }

            instances.append(instance)

        logger.debug(f"  Generated: {len(instances)} instances for {forest_type}")

        return instances

    def _select_tree_types(self, count: int, tree_distribution: Dict[str, float]) -> List[str]:
        """
        Choose tree types based on the distribution.

        Args:
            count: number of tree types to generate
            tree_distribution: dict tree_name → probability (0.0-1.0)

        Returns:
            List of tree type names
        """
        # Extract tree names and probabilities
        tree_names = list(tree_distribution.keys())
        probabilities = list(tree_distribution.values())

        # Normalize probabilities (if the sum != 1.0)
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            # Fallback: uniform distribution
            probabilities = [1.0 / len(tree_names)] * len(tree_names)

        # Filter to available tree species only
        if self.registered_trees:
            available_trees = []
            available_probs = []
            for name, prob in zip(tree_names, probabilities):
                if name in self.registered_trees:
                    available_trees.append(name)
                    available_probs.append(prob)

            if not available_trees:
                logger.warning(f"None of the tree types available: {tree_names}")
                # Fallback: use the first available tree
                if self.registered_trees:
                    fallback = list(self.registered_trees.keys())[0]
                    return [fallback] * count
                else:
                    return ["oak"] * count  # hard fallback

            tree_names = available_trees
            probabilities = available_probs

            # Re-normalize
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]

        # Choose tree types according to the distribution
        tree_types = np.random.choice(tree_names, size=count, p=probabilities)

        return tree_types.tolist()

    def _generate_rotation_matrix(self) -> List[float]:
        """
        Generate a random rotation around the Z axis as a 3x3 rotation matrix (row-major).

        BeamNG's .forest4.json expects "rotationMatrix" as 9 values, not a quaternion.

        Returns:
            [r00, r01, r02, r10, r11, r12, r20, r21, r22]
        """
        # Random angle around the Z axis (0 - 2π)
        angle = np.random.uniform(0, 2 * np.pi)

        c = float(np.cos(angle))
        s = float(np.sin(angle))

        return [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]

    def _generate_scale(self, min_height: float, max_height: float) -> float:
        """
        Generate a random scale from the height range.

        Args:
            min_height: minimum tree height
            max_height: maximum tree height

        Returns:
            Scale factor
        """
        # Assumption: the base tree height is ~20m, scale is relative to it
        base_height = 20.0

        # Random height from the range
        target_height = np.random.uniform(min_height, max_height)

        # Compute the scale
        scale = target_height / base_height

        # Clamp to sensible values
        scale = max(0.5, min(2.0, scale))

        return scale

    def generate_instances_for_forests(
        self,
        forest_points_3d: Dict[int, List[Tuple[float, float, float]]],
        forests: List[Dict],
        forest_properties_map: Dict[str, Dict],
        fitter=None,
    ) -> List[Dict]:
        """
        Generate instances for multiple forest polygons.

        Args:
            forest_points_3d: dict forest_index → list of (x, y, z) points
            forests: list of forest dicts (from the normalizer) with "type"
            forest_properties_map: dict forest_type → properties
            fitter: optional - TrunkFitter: checks the trunks of each instance against the exclusion zone and ground

        Returns:
            List of all generated instances (flat list)
        """
        all_instances = []
        dropped = 0

        for forest_idx, points_3d in forest_points_3d.items():
            if forest_idx >= len(forests):
                logger.warning(f"Forest index {forest_idx} out of range, skipping")
                continue

            forest = forests[forest_idx]
            forest_type = forest.get("type")

            if not forest_type:
                logger.warning(f"Forest polygon {forest_idx} without type, skipping")
                continue

            # Get properties
            properties = forest_properties_map.get(forest_type, {})

            # Generate instances
            instances = self.generate_instances(
                points_3d=points_3d, forest_type=forest_type, forest_properties=properties
            )
            if fitter is not None:
                before = len(instances)
                instances = fitter.fit(
                    instances, properties.get("preferred_trees", {}), row=bool(properties.get("row_spacing"))
                )
                dropped += before - len(instances)

            all_instances.extend(instances)

        logger.info(f"✓ {len(all_instances)} tree instances generated")
        if dropped:
            logger.info(f"  [Trunk] {dropped} instances dropped (no type fits: trunk on a path or floating in the air)")

        return all_instances
