"""
Koordinaten-Transformationen (WGS84 ↔ UTM).
"""

from pyproj import Transformer

# Transformer: GPS (WGS84) <-> UTM Zone 32N (Metrisch fuer Mitteleuropa)
transformer_to_utm = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
transformer_to_wgs84 = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)


def apply_local_offset(x, y, z):
    """Konvertiert zu lokalen Koordinaten relativ zum ersten Punkt (Array-fähig)."""
    from .. import config

    if config.LOCAL_OFFSET is None:
        # Setze Offset vom ersten Wert
        if isinstance(x, __import__("numpy").ndarray):
            config.LOCAL_OFFSET = (x[0], y[0], z[0])
        else:
            config.LOCAL_OFFSET = (x, y, z)
    return (
        x - config.LOCAL_OFFSET[0],
        y - config.LOCAL_OFFSET[1],
        z - config.LOCAL_OFFSET[2],
    )
