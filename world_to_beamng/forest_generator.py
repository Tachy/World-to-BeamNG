"""
Dynamischer Forest-Type-Generator

Generiert zur Runtime forest_types basierend auf:
1. Registrierten Tree-Assets (aus MaterialManager-Singleton)
2. Waldtyp-Templates aus osm_to_beamng.json
3. Intelligenter Baum-Verteilung auf verfügbare Assets
"""

from typing import Dict, List, Set, Any
import logging
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()


class ForestGenerator:
    """Generiert dynamisch forest_types aus Templates und verfügbaren Assets."""

    def __init__(self, registered_trees: Dict[str, Any], template_config: Dict[str, Any]):
        """
        Args:
            registered_trees: Dict der registrierten Baumarten (aus ForestAssetScanner)
            template_config: Dictionary mit forest_type_templates aus osm_to_beamng.json
        """
        self.registered_trees = registered_trees
        self.templates = template_config
        self.available_trees = self._get_available_trees()

    def _get_available_trees(self) -> Set[str]:
        """
        Hole verfügbare Tree-Types aus registered_trees.

        Returns:
            Set von Tree-Type-Namen die registriert sind
        """
        available = set(self.registered_trees.keys())
        logger.info(f"Verfügbare Baum-Typen: {sorted(available)}")
        return available

    def _get_available_trees_for_template(self, template_name: str) -> List[str]:
        """
        Findet verfügbare Bäume für einen Waldtyp.

        Args:
            template_name: Name des Waldtyps (z.B. "deciduous_dense")

        Returns:
            Liste von verfügbaren Baum-Typen für diesen Waldtyp
        """
        if template_name not in self.templates:
            return []

        template = self.templates[template_name]
        preferred = template.get("preferred_trees", [])
        fallback = template.get("fallback_trees", [])

        # Filtere nur verfügbare Bäume
        available_preferred = [t for t in preferred if t in self.available_trees]
        available_fallback = [t for t in fallback if t in self.available_trees]

        # Bevorzugte zuerst, dann Fallback
        result = available_preferred + available_fallback

        if not result:
            logger.warning(
                f"Keine Baum-Typen für '{template_name}' verfügbar. " f"Bevorzugt: {preferred}, Fallback: {fallback}"
            )

        return result

    def _distribute_trees(self, available_trees: List[str]) -> Dict[str, float]:
        """
        Verteilt verfügbare Bäume gleichmäßig auf Walddichte.

        Args:
            available_trees: Liste verfügbarer Baum-Typen

        Returns:
            Dict mit Baum-Typen und deren Gewichtung (Wahrscheinlichkeit)
        """
        if not available_trees:
            return {}

        # Gleichmäßige Verteilung
        weight = 1.0 / len(available_trees)
        return {tree: weight for tree in available_trees}

    def generate_forest_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Generiert forest_types mit real verfügbaren Baum-Typen.

        Returns:
            Dictionary mit allen generierten forest_types
        """
        forest_types = {}

        for template_name, template in self.templates.items():
            # Finde verfügbare Bäume für diesen Waldtyp
            available_trees = self._get_available_trees_for_template(template_name)

            # Verteile sie auf die Template-Dichte
            tree_distribution = self._distribute_trees(available_trees)

            # Baue forest_type aus Template
            forest_type = {
                "tree_density": template["tree_density"],
                "tree_distribution": tree_distribution,
                "average_height": template["average_height"],
                "underground_material": template["underground_material"],
                "lod_distance": template["lod_distance"],
                "collision_enabled": template["collision_enabled"],
                "comment": template.get("comment", ""),
            }

            forest_types[template_name] = forest_type

            # Log
            if tree_distribution:
                trees_str = ", ".join(sorted(tree_distribution.keys()))
                logger.info(f"  {template_name}: {trees_str} " f"(Dichte: {template['tree_density']})")
            else:
                logger.info(f"  {template_name}: LEER (keine Bäume verfügbar)")

        return forest_types


def get_forest_types_from_registered_trees(
    registered_trees: Dict[str, Any], template_config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Hilfsfunktion: Generiere forest_types aus registrierten Baumarten.

    Args:
        registered_trees: Dict der registrierten Trees (aus ForestAssetScanner)
        template_config: Forest-Type-Templates aus osm_to_beamng.json

    Returns:
        Dictionary mit generierten forest_types
    """
    generator = ForestGenerator(registered_trees=registered_trees, template_config=template_config)
    return generator.generate_forest_types()
