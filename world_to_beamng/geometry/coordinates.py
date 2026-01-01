"""
Koordinaten-Transformationen für World-to-BeamNG.
Stellt pyproj Transformer bereit für WGS84 <-> UTM Konvertierungen.
"""

from pyproj import Transformer

# Transformer für UTM Zone 32N (Deutschland) -> WGS84
transformer_to_wgs84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

# Transformer für WGS84 -> UTM Zone 32N
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
