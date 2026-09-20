"""
Materialnamen der LOD2-Gebäude (eine Quelle für Mesh-Builder, DAE-Exporter und Material-Export).
"""

from .facade_styles import PLASTER_COLORS

WALL_MATERIALS = tuple(f"lod2_wall_plaster_{color.name}" for color in PLASTER_COLORS)  # Index = PLASTER_COLORS
WINDOW_MATERIAL = "lod2_windows"  # Fenster, Türen, Kellerfenster (Sprite-Atlas)
ROOF_MATERIAL = "lod2_roof_red"  # Biberschwanz-Dach
FLAT_ROOF_MATERIAL = "lod2_roof_flat"  # Kiesfläche auf Flachdächern
ROOF_EDGE_MATERIAL = "lod2_roof_edge"  # Blechrand um Flachdächer
ROOF_TRIM_MATERIAL = "lod2_roof_trim"  # Stirnbrett und Untersicht der Dachüberstände

# Diffuse Vorschaufarben im DAE-Effekt (BeamNG nutzt die Texturen aus materials.json)
DAE_EFFECT_COLORS = {
    **{name: tuple(channel / 255.0 for channel in color.rgb) for name, color in zip(WALL_MATERIALS, PLASTER_COLORS)},
    WINDOW_MATERIAL: (0.55, 0.62, 0.70),
    ROOF_MATERIAL: (0.6, 0.2, 0.1),
    FLAT_ROOF_MATERIAL: (0.5, 0.49, 0.47),
    ROOF_EDGE_MATERIAL: (0.62, 0.64, 0.67),
    ROOF_TRIM_MATERIAL: (0.36, 0.26, 0.19),
}
