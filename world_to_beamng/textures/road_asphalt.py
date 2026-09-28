"""
Asphalt roads with a PBR texture from ambientCG instead of BeamNG's stock asphalt.

A DecalRoad maps its texture ONCE across the full carriageway width (u 0..1) and repeats it along the road every
config.ROAD_DECAL_TEXTURE_LENGTH meters; the carriageways on bridges, in tunnels and galleries use the same UV layout
(see config.ROAD_DECAL_TEXTURE_LENGTH). A square ambientCG tile stretched across 7 m would look coarse, so the import
builds a road strip: ROAD_ASPHALT_TILES_ACROSS tiles side by side across the road and ROAD_ASPHALT_TILES_ALONG whole
tiles along one texture repeat (whole tiles keep the repeats seamless). The strip is stored as a normal library
texture (data/textures/<name>/, see textures/library.py) and written into the level as DDS like the others.

ambientCG ZIPs hold <asset>_Color.png, _NormalGL.png/_NormalDX.png, _Roughness.png and _AmbientOcclusion.png; the
green-up NormalGL is the one BeamNG expects (config.TEXTURE_NORMAL_GREEN_UP), the ambient occlusion is baked into the
colour (the library has no AO channel).

Without an ambientCG ZIP the strip is built from BeamNG's own stock asphalt (write_stock_road_strip()): that texture is a
terrain detail texture with a grain for ~1.2 m per tile, stretched once across a whole carriageway its chips would look
four times too big. The stock strip is built from the local installation on every export and written into the level as
DDS only - BeamNG assets must not be redistributed, so it never goes into the (public) texture library.
"""

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from .. import config
from . import library

_SUFFIXES = {"color": "_color", "normal": "_normalgl", "roughness": "_roughness", "ao": "_ambientocclusion"}


def _zip_signature(zip_path: Path) -> str:
    """Name and content hash: another download under the same name (even of the same size) is imported again."""
    digest = hashlib.sha1(Path(zip_path).read_bytes()).hexdigest()[:12]
    return f"{Path(zip_path).name} sha1 {digest}"


def read_ambientcg_zip(zip_path: Path) -> Dict[str, np.ndarray]:
    """{"color" (H, W, 3), "normal" (H, W, 3), "roughness" (H, W), "ao" (H, W)} uint8 from an ambientCG ZIP (AO optional:
    white without it)."""
    maps: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            stem = Path(name).stem.lower()
            for key, suffix in _SUFFIXES.items():
                if stem.endswith(suffix) and name.lower().endswith((".png", ".jpg", ".jpeg")):
                    image = Image.open(io.BytesIO(archive.read(name)))
                    mode = "RGB" if key in ("color", "normal") else "L"
                    maps[key] = np.asarray(image.convert(mode), dtype=np.uint8)
    missing = [key for key in ("color", "normal", "roughness") if key not in maps]
    if missing:
        raise ValueError(f"{Path(zip_path).name} is no ambientCG PBR ZIP: {', '.join(missing)} missing")
    maps.setdefault("ao", np.full(maps["color"].shape[:2], 255, dtype=np.uint8))
    return maps


def _resize(array: np.ndarray, px: int) -> np.ndarray:
    image = Image.fromarray(array)
    return np.asarray(image.resize((px, px), Image.Resampling.LANCZOS), dtype=np.uint8)


def _renormalize(normal: np.ndarray) -> np.ndarray:
    """Unit length again after resampling (the filter shortens the vectors)."""
    vectors = normal.astype(np.float64) / 127.5 - 1.0
    vectors /= np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1e-6)
    return np.clip((vectors + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)


def build_road_strip(maps: Dict[str, np.ndarray], tiles_across: int, tiles_along: int, tile_px: int) -> Dict[str, np.ndarray]:
    """
    The road strip of `tiles_across` x `tiles_along` copies of the tile (each tile_px square): columns run across the
    road, rows along it. Returns {"color", "normal", "roughness"} as uint8 RGB for library.store_texture() ("roughness"
    only if the source has one); the ambient occlusion is multiplied into the colour.
    """
    ao = _resize(maps["ao"], tile_px).astype(np.float64) / 255.0
    color = np.clip(_resize(maps["color"], tile_px).astype(np.float64) * ao[..., None] + 0.5, 0, 255).astype(np.uint8)
    normal = _renormalize(_resize(maps["normal"], tile_px))
    tiles = [("color", color), ("normal", normal)]
    if "roughness" in maps:
        tiles.append(("roughness", np.repeat(_resize(maps["roughness"], tile_px)[..., None], 3, axis=2)))
    return {key: np.tile(tile, (tiles_along, tiles_across, 1)) for key, tile in tiles}


def import_road_asphalt(
    library_dir: Optional[Path] = None,
    zip_name: Optional[str] = None,
    name: Optional[str] = None,
    tiles_across: Optional[int] = None,
    tiles_along: Optional[int] = None,
    tile_px: Optional[int] = None,
) -> Path:
    """Imports the ambientCG ZIP from the library folder as the road strip texture (defaults from config)."""
    library_dir = Path(library_dir or config.TEXTURE_LIBRARY_DIR)
    zip_path = library_dir / (zip_name or config.ROAD_ASPHALT_TEXTURE_ZIP)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Asphalt texture: {zip_path} is missing - put the ambientCG ZIP there (e.g. Road012A_2K-PNG.zip from "
            f"ambientcg.com) or set config.ROAD_ASPHALT_TEXTURE_ZIP = None for BeamNG's stock asphalt"
        )
    across = tiles_across or config.ROAD_ASPHALT_TILES_ACROSS
    along = tiles_along or config.ROAD_ASPHALT_TILES_ALONG
    strip = build_road_strip(read_ambientcg_zip(zip_path), across, along, tile_px or config.ROAD_ASPHALT_TILE_PX)
    return library.store_texture(
        name or config.ROAD_ASPHALT_TEXTURE_NAME,
        strip,
        tile_m=config.ROAD_DECAL_TEXTURE_LENGTH,  # one strip = one texture repeat along the road
        source=f"ambientCG {_zip_signature(zip_path)} (CC0), {across} tiles across x {along} along, AO baked into colour",
        library_dir=library_dir,
    )


def road_asphalt_outdated(library_dir: Optional[Path] = None, zip_name: Optional[str] = None, name: Optional[str] = None) -> bool:
    """The imported strip was made from another ZIP (other name or content) than the one in the library folder now."""
    library_dir = Path(library_dir or config.TEXTURE_LIBRARY_DIR)
    zip_path = library_dir / (zip_name or config.ROAD_ASPHALT_TEXTURE_ZIP)
    entry = library.load_manifest(library_dir).get(name or config.ROAD_ASPHALT_TEXTURE_NAME)
    return entry is not None and zip_path.exists() and _zip_signature(zip_path) not in entry.get("source", "")


def use_road_asphalt(surface_types: Dict[str, Dict], surface: str, library_maps: Dict[str, str]) -> None:
    """Gives the surface type (data/osm_to_beamng.json) the library textures instead of the stock ones - the DecalRoad
    material and the structure carriageways both take their textures from there. Stock AO/opacity maps are dropped: the
    AO is baked into the colour, the stock asphalt has no opacity either."""
    surface_types[surface]["textures"] = dict(library_maps)


STOCK_STRIP_NAME = "road_asphalt_stock"
_STOCK_HASH_FILE = "road_asphalt_stock.hash.json"


def _write_dds(pixels: np.ndarray, output_dir: Path, name: str, dds_format: str) -> Path:
    from ..facade import dds_export

    return dds_export.write_dds(pixels, output_dir, name, dds_format, 0)  # full mip chain: tiling texture


def write_stock_road_strip(
    install_dir: Path,
    output_dir: Path,
    source: str,
    tiles_across: int,
    tiles_along: int,
    tile_px: int,
) -> Dict[str, str]:
    """
    Road strip from BeamNG's stock asphalt `source` (path below content/assets/materials without the "_b.color.dds"
    suffix, e.g. "tileable/road/m_asphalt_new_01/t_asphalt_02"), written as DDS into `output_dir` - only when the
    source zip or the tiling changed.

    Returns:
        {"baseColorMap", "normalMap"} relative to the BeamNG user folder (the stock asphalt has no roughness map)
    """
    import json

    from ..facade import dds_export

    materials = Path(install_dir) / "content" / "assets" / "materials"
    prefix = f"assets/materials/{source}"
    names = {"color": f"{prefix}_b.color.dds", "normal": f"{prefix}_nm.normal.dds", "ao": f"{prefix}_ao.data.dds"}
    zip_path = None
    for candidate in sorted(materials.glob("*.zip")):
        with zipfile.ZipFile(candidate) as archive:
            if names["color"] in archive.namelist():
                zip_path = candidate
                break
    if zip_path is None:
        raise FileNotFoundError(f"Stock asphalt {source} not found in {materials}")

    output_dir = Path(output_dir)
    signature = f"{_zip_signature(zip_path)} {source} {tiles_across}x{tiles_along} {tile_px}px"
    hash_path = output_dir / _STOCK_HASH_FILE
    channels = (("color", "baseColorMap", "_b.color", dds_export.COLOR), ("normal", "normalMap", "_nm.normal", dds_export.NORMAL))
    result = {key: str(config.RELATIVE_DIR_TEXTURES / f"{STOCK_STRIP_NAME}{suffix}.dds") for _, key, suffix, _ in channels}
    up_to_date = hash_path.exists() and json.loads(hash_path.read_text(encoding="utf-8")).get("signature") == signature
    if up_to_date and all((output_dir / f"{STOCK_STRIP_NAME}{suffix}.dds").exists() for _, _, suffix, _ in channels):
        return result

    maps = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        for key, member in names.items():
            if member in members:
                mode = "L" if key == "ao" else "RGB"
                maps[key] = np.asarray(Image.open(io.BytesIO(archive.read(member))).convert(mode), dtype=np.uint8)
    maps.setdefault("ao", np.full(maps["color"].shape[:2], 255, dtype=np.uint8))
    strip = build_road_strip(maps, tiles_across, tiles_along, tile_px)
    for channel, _, suffix, dds_format in channels:
        _write_dds(strip[channel], output_dir, f"{STOCK_STRIP_NAME}{suffix}", dds_format)
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(json.dumps({"signature": signature}), encoding="utf-8")
    return result
