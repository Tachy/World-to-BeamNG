"""
Coordinate transformations for World-to-BeamNG.
Provides pyproj transformers for WGS84 <-> source CRS conversions.

The source CRS is EPSG:25832 by default (ETRS89/UTM32N, LGL Baden-Württemberg), but after the tile
scan it may be overridden by the CRS actually detected in the input data (see
set_source_crs()). transformer_to_wgs84/transformer_to_utm are lazy proxies: the real
pyproj.Transformer is only built on the first actual attribute access (and rebuilt after a
later CRS change) - this is necessary because several modules bind these objects via
"from ... import transformer_to_wgs84" AT IMPORT TIME (i.e. before set_source_crs() could run in
main()); the proxy itself is never reassigned, only its content changes lazily.
"""

from pyproj import Transformer

from .. import config

_source_epsg: int | None = None  # None = not explicitly set yet -> config.SOURCE_CRS_EPSG applies


def set_source_crs(epsg: int) -> None:
    """
    Sets the detected/configured source CRS for the current pipeline run.

    Must be called BEFORE the first coordinate transformation (world_to_beamng.py::main(),
    right after the tile scan, before compute_global_center()/export_complete_level()).
    """
    global _source_epsg
    _source_epsg = int(epsg)


def get_source_crs_epsg() -> int:
    """Returns the currently set source CRS, or config.SOURCE_CRS_EPSG as a fallback."""
    return _source_epsg if _source_epsg is not None else config.SOURCE_CRS_EPSG


class _LazyTransformer:
    """
    Builds the real pyproj.Transformer only on the first attribute access, and rebuilds it as soon as
    get_source_crs_epsg() has changed since.

    Passes ALL attribute accesses through (not only .transform()), since some callers also query
    .source_crs/.target_crs directly (see workflow/forest_workflow.py).
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


# Transformer for source CRS -> WGS84
transformer_to_wgs84 = _LazyTransformer(lambda: f"EPSG:{get_source_crs_epsg()}", lambda: "EPSG:4326")

# Transformer for WGS84 -> source CRS
transformer_to_utm = _LazyTransformer(lambda: "EPSG:4326", lambda: f"EPSG:{get_source_crs_epsg()}")
