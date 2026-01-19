"""
Forest Instance Generator: Erzeugt finale Baum-Instanzen.

Generiert aus (x, y, z) Positionen vollständige Baum-Instances mit:
- Tree-Type (basierend auf tree_distribution)
- Rotation (Quaternion um Z-Achse)
- Scale (aus average_height Range)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ForestInstanceGenerator:
    """
    Generiert finale Baum-Instanzen mit Type, Rotation und Scale.

    Format pro Instance:
    {
        "type": "oak",
        "pos": [x, y, z],
        "rot": [rx, ry, rz, rw],  # Quaternion
        "scale": 1.15
    }
    """

    def __init__(self, registered_trees: Optional[Dict] = None):
        """
        Args:
            registered_trees: Optional - Dict von verfügbaren Baumarten (aus AssetScanner)
        """
        self.registered_trees = registered_trees or {}

    def generate_instances(
        self, points_3d: List[Tuple[float, float, float]], forest_type: str, forest_properties: Dict
    ) -> List[Dict]:
        """
        Generiere Baum-Instanzen für ein Waldpolygon.

        Args:
            points_3d: Liste von (x, y, z) Positionen
            forest_type: Forest-Type (z.B. "deciduous_dense")
            forest_properties: Properties aus forest_types (tree_distribution, average_height, etc.)

        Returns:
            Liste von Instance-Dicts
        """
        if not points_3d:
            return []

        instances = []

        # Tree Distribution (prozentuale Anteile)
        tree_distribution = forest_properties.get("tree_distribution", {})
        if not tree_distribution:
            logger.warning(f"Keine tree_distribution für {forest_type}, überspringe")
            return []

        # Average Height Range
        avg_height_range = forest_properties.get("average_height", [15.0, 25.0])
        min_height = avg_height_range[0]
        max_height = avg_height_range[1]

        # Wähle Tree-Types für alle Punkte
        tree_types = self._select_tree_types(len(points_3d), tree_distribution)

        # Generiere Instances
        for i, (x, y, z) in enumerate(points_3d):
            tree_type = tree_types[i]

            # Rotation (zufällig um Z-Achse)
            rotation = self._generate_rotation()

            # Scale (basierend auf average_height)
            scale = self._generate_scale(min_height, max_height)

            instance = {
                "type": tree_type,
                "pos": [float(x), float(y), float(z)],
                "rot": rotation,
                "scale": float(scale),
            }

            instances.append(instance)

        logger.debug(f"  Generiert: {len(instances)} Instanzen für {forest_type}")

        return instances

    def _select_tree_types(self, count: int, tree_distribution: Dict[str, float]) -> List[str]:
        """
        Wähle Tree-Types basierend auf Verteilung.

        Args:
            count: Anzahl zu generierender Tree-Types
            tree_distribution: Dict tree_name → probability (0.0-1.0)

        Returns:
            Liste von Tree-Type-Namen
        """
        # Extrahiere Tree-Names und Probabilities
        tree_names = list(tree_distribution.keys())
        probabilities = list(tree_distribution.values())

        # Normalisiere Probabilities (falls Summe != 1.0)
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            # Fallback: Gleichverteilung
            probabilities = [1.0 / len(tree_names)] * len(tree_names)

        # Filtere nur verfügbare Baumarten
        if self.registered_trees:
            available_trees = []
            available_probs = []
            for name, prob in zip(tree_names, probabilities):
                if name in self.registered_trees:
                    available_trees.append(name)
                    available_probs.append(prob)

            if not available_trees:
                logger.warning(f"Keine der Tree-Types verfügbar: {tree_names}")
                # Fallback: Nutze ersten verfügbaren Baum
                if self.registered_trees:
                    fallback = list(self.registered_trees.keys())[0]
                    return [fallback] * count
                else:
                    return ["oak"] * count  # Hard Fallback

            tree_names = available_trees
            probabilities = available_probs

            # Re-normalisiere
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]

        # Wähle Tree-Types nach Verteilung
        tree_types = np.random.choice(tree_names, size=count, p=probabilities)

        return tree_types.tolist()

    def _generate_rotation(self) -> List[float]:
        """
        Generiere zufällige Rotation um Z-Achse (Quaternion).

        Returns:
            [rx, ry, rz, rw] Quaternion
        """
        # Zufälliger Winkel um Z-Achse (0 - 2π)
        angle = np.random.uniform(0, 2 * np.pi)

        # Quaternion für Rotation um Z-Achse:
        # q = [0, 0, sin(angle/2), cos(angle/2)]
        half_angle = angle / 2.0

        rx = 0.0
        ry = 0.0
        rz = float(np.sin(half_angle))
        rw = float(np.cos(half_angle))

        return [rx, ry, rz, rw]

    def _generate_scale(self, min_height: float, max_height: float) -> float:
        """
        Generiere zufällige Skalierung aus Height-Range.

        Args:
            min_height: Minimale Baumhöhe
            max_height: Maximale Baumhöhe

        Returns:
            Scale-Faktor
        """
        # Annahme: Basis-Baumhöhe ist ~20m, Scale skaliert relativ dazu
        base_height = 20.0

        # Zufällige Höhe aus Range
        target_height = np.random.uniform(min_height, max_height)

        # Scale berechnen
        scale = target_height / base_height

        # Clamp zu vernünftigen Werten
        scale = max(0.5, min(2.0, scale))

        return scale

    def generate_instances_for_forests(
        self,
        forest_points_3d: Dict[int, List[Tuple[float, float, float]]],
        forests: List[Dict],
        forest_properties_map: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Generiere Instanzen für mehrere Waldpolygone.

        Args:
            forest_points_3d: Dict forest_index → Liste von (x, y, z) Punkten
            forests: Liste von Forest-Dicts (aus Normalizer) mit "type"
            forest_properties_map: Dict forest_type → properties

        Returns:
            Liste aller generierten Instanzen (flache Liste)
        """
        all_instances = []

        for forest_idx, points_3d in forest_points_3d.items():
            if forest_idx >= len(forests):
                logger.warning(f"Forest-Index {forest_idx} außerhalb Bereich, überspringe")
                continue

            forest = forests[forest_idx]
            forest_type = forest.get("type")

            if not forest_type:
                logger.warning(f"Waldpolygon {forest_idx} ohne type, überspringe")
                continue

            # Hole Properties
            properties = forest_properties_map.get(forest_type, {})

            # Generiere Instances
            instances = self.generate_instances(
                points_3d=points_3d, forest_type=forest_type, forest_properties=properties
            )

            all_instances.extend(instances)

        logger.info(f"✓ {len(all_instances)} Baum-Instanzen generiert")

        return all_instances
