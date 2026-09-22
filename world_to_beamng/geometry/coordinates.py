"""
Koordinaten-Transformationen für World-to-BeamNG.
Stellt pyproj Transformer bereit für WGS84 <-> Quell-CRS Konvertierungen.

Das Quell-CRS ist standardmäßig EPSG:25832 (ETRS89/UTM32N, LGL Baden-Württemberg), wird aber nach
dem Kachel-Scan ggf. durch das tatsächlich erkannte CRS der Eingabedaten überschrieben (siehe
set_source_crs()). transformer_to_wgs84/transformer_to_utm sind Lazy-Proxys: der echte
pyproj.Transformer wird erst beim ersten tatsächlichen Attributzugriff gebaut (und bei einem
späteren CRS-Wechsel neu) - das ist nötig, weil mehrere Module diese Objekte per
"from ... import transformer_to_wgs84" ZUR IMPORTZEIT binden (also bevor set_source_crs() in
main() laufen konnte); der Proxy selbst wird nie neu zugewiesen, nur sein Inhalt wechselt lazy.
"""

from pyproj import Transformer

from .. import config

_source_epsg: int | None = None  # None = noch nicht explizit gesetzt -> config.SOURCE_CRS_EPSG gilt


def set_source_crs(epsg: int) -> None:
    """
    Setzt die erkannte/konfigurierte Quell-CRS für die aktuelle Pipeline-Ausführung.

    Muss VOR der ersten Koordinatentransformation aufgerufen werden (world_to_beamng.py::main(),
    direkt nach dem Kachel-Scan, vor compute_global_center()/export_complete_level()).
    """
    global _source_epsg
    _source_epsg = int(epsg)


def get_source_crs_epsg() -> int:
    """Liefert die aktuell gesetzte Quell-CRS, oder config.SOURCE_CRS_EPSG als Fallback."""
    return _source_epsg if _source_epsg is not None else config.SOURCE_CRS_EPSG


class _LazyTransformer:
    """
    Baut den echten pyproj.Transformer erst beim ersten Attributzugriff, und neu, sobald sich
    get_source_crs_epsg() seitdem geändert hat.

    Reicht ALLE Attributzugriffe durch (nicht nur .transform()), da manche Aufrufer auch
    .source_crs/.target_crs direkt abfragen (siehe workflow/forest_workflow.py).
    """

    def __init__(self, src_crs_fn, dst_crs_fn):
        self._src_crs_fn = src_crs_fn
        self._dst_crs_fn = dst_crs_fn
        self._built = None
        self._built_epsg = None

    def _ensure(self):
        epsg = get_source_crs_epsg()
        if self._built is None or self._built_epsg != epsg:
            self._built = Transformer.from_crs(self._src_crs_fn(), self._dst_crs_fn(), always_xy=True)
            self._built_epsg = epsg
        return self._built

    def __getattr__(self, name):
        return getattr(self._ensure(), name)


# Transformer für Quell-CRS -> WGS84
transformer_to_wgs84 = _LazyTransformer(lambda: f"EPSG:{get_source_crs_epsg()}", lambda: "EPSG:4326")

# Transformer für WGS84 -> Quell-CRS
transformer_to_utm = _LazyTransformer(lambda: "EPSG:4326", lambda: f"EPSG:{get_source_crs_epsg()}")
