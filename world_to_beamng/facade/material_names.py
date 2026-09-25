"""
Material names of the LOD2 buildings (single source for mesh builders, DAE exporter and material export).
"""

from .facade_styles import PLASTER_COLORS

WALL_MATERIALS = tuple(f"lod2_wall_plaster_{color.name}" for color in PLASTER_COLORS)  # index = PLASTER_COLORS
WINDOW_MATERIAL = "lod2_windows"  # windows, doors, basement windows (sprite atlas)
ROOF_MATERIAL = "lod2_roof_red"  # beaver-tail tile roof
FLAT_ROOF_MATERIAL = "lod2_roof_flat"  # gravel surface on flat roofs
ROOF_EDGE_MATERIAL = "lod2_roof_edge"  # sheet-metal rim around flat roofs
ROOF_TRIM_MATERIAL = "lod2_roof_trim"  # fascia board and soffit of the roof overhangs

# Diffuse preview colors in the DAE effect (BeamNG uses the textures from materials.json)
DAE_EFFECT_COLORS = {
    **{name: tuple(channel / 255.0 for channel in color.rgb) for name, color in zip(WALL_MATERIALS, PLASTER_COLORS)},
    WINDOW_MATERIAL: (0.55, 0.62, 0.70),
    ROOF_MATERIAL: (0.6, 0.2, 0.1),
    FLAT_ROOF_MATERIAL: (0.5, 0.49, 0.47),
    ROOF_EDGE_MATERIAL: (0.62, 0.64, 0.67),
    ROOF_TRIM_MATERIAL: (0.36, 0.26, 0.19),
}
