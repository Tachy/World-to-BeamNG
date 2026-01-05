"""
Mapping von OSM-Daten zu BeamNG-Material-Parametern.

Zentrale Verwaltung von:
- Straßenbreiten basierend auf OSM-Tags
- Weitere Material-Eigenschaften (zukünftig)
"""

WIDTH_FALLBACK = {
    "secondary": 7.0,
    "tertiary": 6.0,
    "unclassified": 5.0,
    "residential": 5.0,
    "living_street": 4.5,
    "service": 3.5,
    "track": 3.0,
    "path": 2.0,
    "footway": 2.0,
    "steps": 1.5,
}


def get_road_width(tags):
    """
    Bestimmt die Straßenbreite aus OSM-Tags mit mehrstufigem Fallback.

    Args:
        tags: Dict mit OSM-Tags (z.B. {"highway": "residential", "lanes": "2"})

    Returns:
        float: Straßenbreite in Metern

    Fallback-Logik:
        1. Direktes "width"-Tag (Meter, auch als "7.5m" formatierbar)
        2. "lanes"-Tag × 3.25m (Standardspurbreite)
        3. Highway-Typ-Fallback (secondary=7m, residential=5m, etc.)
        4. Ultimativer Default: 4.0m
    """
    if tags is None:
        tags = {}

    # 1. Direktes width-Tag
    if "width" in tags:
        try:
            width_str = str(tags["width"]).replace("m", "").strip()
            return float(width_str)
        except (ValueError, AttributeError):
            pass

    # 2. Lanes-Fallback (typisch 3.25m pro Spur)
    if "lanes" in tags:
        try:
            lanes = int(tags["lanes"])
            return lanes * 3.25
        except (ValueError, TypeError):
            pass

    # 3. Highway-Typ Fallback
    hw_type = tags.get("highway", "unclassified")
    if isinstance(hw_type, str):
        # Vereinfache highway-Typ (manchmal mit Präfixes wie "primary_link")
        base_type = hw_type.split("_")[0]
        return WIDTH_FALLBACK.get(base_type, WIDTH_FALLBACK.get(hw_type, 4.0))

    # 4. Ultimativer Default
    return 4.0
