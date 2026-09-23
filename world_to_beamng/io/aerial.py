"""
Aerial image processing - Extrahiert und kachelt Luftbildaufnahmen.
"""

import json
import zipfile
import math
from pathlib import Path
from PIL import Image, ImageEnhance
from io import BytesIO
from world_to_beamng.logging_config import LoggerConfig
from .. import config

# Dieses Modul baut selbst große Leinwände aus eigenen, vertrauenswürdigen Geodaten (kein Öffnen
# einer fremden Datei) - PILs Decompression-Bomb-Schutz (Default-Grenze ~89,5 Mio. Pixel) greift
# hier grundlos: schon eine 2 km-Kachel bei feiner Auflösung (z.B. 0.1m/px Schweizer Orthofotos)
# liegt weit darüber.
Image.MAX_IMAGE_PIXELS = None

logger = LoggerConfig.get_logger()


def parse_world_file(tfw_data):
    """
    Parst World File (.tfw) Daten.

    Args:
        tfw_data: Bytes oder String der .tfw-Datei

    Returns:
        Dict mit pixel_size_x, pixel_size_y, x_origin, y_origin
    """
    if isinstance(tfw_data, bytes):
        tfw_data = tfw_data.decode("utf-8")

    lines = tfw_data.strip().split("\n")
    if len(lines) < 6:
        return None

    try:
        pixel_size_x = float(lines[0])
        pixel_size_y = float(lines[3])
        x_origin = float(lines[4])
        y_origin = float(lines[5])

        return {
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "x_origin": x_origin,
            "y_origin": y_origin,
        }
    except (ValueError, IndexError):
        return None


def extract_images_from_zips(aerial_dir=config.AERIAL_DATA_DIR):
    """
    Extrahiert alle Bilder mit Georeferenzierung aus ZIP-Dateien.

    Georeferenzierung kommt entweder aus einer begleitenden .tfw-Datei (world_info["crs_epsg"] bleibt
    dann None - die Quell-CRS wird wie bisher angenommen) ODER, falls keine .tfw da ist, aus
    eingebetteten GeoTIFF-Tags im Bild selbst (world_info["crs_epsg"] gesetzt). Reines JPG/PNG ohne
    .tfw und ohne Geo-Tags bleibt ein Fehlerfall (world_info=None, wird später verworfen).

    Args:
        aerial_dir: Pfad zum Luftbild-Verzeichnis (config.AERIAL_DATA_DIR)

    Returns:
        List von (image_name, image_data_bytes, world_file_info) Tupeln
    """
    aerial_path = Path(aerial_dir)
    images = []

    if not aerial_path.exists():
        logger.error(f"[!] Verzeichnis {aerial_dir} existiert nicht")
        return images

    zip_files = list(aerial_path.glob("*.zip"))

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                file_list = zip_ref.namelist()

                # Finde Bilddateien (TIF, TIFF, JPG, JPEG, PNG)
                image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
                image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in image_extensions)]

                for img_file in image_files:
                    img_data = zip_ref.read(img_file)

                    # Suche passende .tfw-Datei
                    # Ersetze Bildendung mit .tfw (z.B. .tif → .tfw)
                    img_path = Path(img_file)
                    tfw_file = str(img_path.with_suffix(".tfw"))

                    world_info = None
                    if tfw_file in file_list:
                        tfw_data = zip_ref.read(tfw_file)
                        world_info = parse_world_file(tfw_data)
                    else:
                        # Debugging: Suche .tfw mit gleichem Stammnamen (case-insensitive)
                        base_name = img_path.stem.lower()
                        for f in file_list:
                            if f.lower().endswith(".tfw") and Path(f).stem.lower() == base_name:
                                tfw_data = zip_ref.read(f)
                                world_info = parse_world_file(tfw_data)
                                break

                    if world_info is None:
                        # Kein .tfw gefunden - evtl. hat das Bild selbst eine eingebettete
                        # GeoTIFF-Georeferenz (kein .tfw nötig, z.B. viele generische GeoTIFF-Portale)
                        world_info = _read_geotiff_world_info(f"/vsizip/{zip_path}/{img_file}")

                    images.append((img_file, img_data, world_info))

        except Exception as e:
            logger.error(f"[!] Fehler beim Lesen von {zip_path.name}: {e}")

    return images


def _read_geotiff_world_info(path_or_vsi):
    """
    Liest CRS + world_info (pixel_size_x/y, x_origin, y_origin = obere linke Pixelecke) aus einem
    georeferenzierten Raster via rasterio - eingebettete GeoTIFF-Tags, kein .tfw nötig.

    Returns:
        dict wie parse_world_file(), zusätzlich "crs_epsg" (kann None sein, falls das CRS keinen
        EPSG-Code hat - bekannte Grenze), oder None falls kein CRS/keine echte Geotransform da ist.
    """
    import rasterio

    try:
        with rasterio.open(path_or_vsi) as src:
            if src.crs is None or src.transform.is_identity:
                return None
            t = src.transform
            return {
                "pixel_size_x": t.a,
                "pixel_size_y": t.e,
                "x_origin": t.c,
                "y_origin": t.f,
                "crs_epsg": src.crs.to_epsg(),
            }
    except Exception:
        return None


def extract_loose_images(aerial_dir=config.AERIAL_DATA_DIR):
    """
    Lose Rasterdateien direkt im Verzeichnis (*.tif, *.tiff) - nicht in einem ZIP. Georeferenzierung
    wie bei extract_images_from_zips(): eingebettete GeoTIFF-Tags bevorzugt, sonst eine begleitende
    .tfw-Datei gleichen Namens.

    Returns:
        Liste von (image_name, image_path: Path, world_info) - image_path (nicht bytes!), da lose
        GeoTIFFs beliebig groß sein können; siehe _open_image().
    """
    aerial_path = Path(aerial_dir)
    images = []
    if not aerial_path.exists():
        return images

    paths = sorted(aerial_path.glob("*.tif")) + sorted(aerial_path.glob("*.tiff"))
    for path in paths:
        world_info = _read_geotiff_world_info(str(path))
        if world_info is None:
            tfw = path.with_suffix(".tfw")
            if tfw.exists():
                world_info = parse_world_file(tfw.read_bytes())
                if world_info is not None:
                    world_info["crs_epsg"] = None
        images.append((path.name, path, world_info))
    return images


def extract_georeferenced_images(aerial_dir=config.AERIAL_DATA_DIR):
    """
    Kombiniert extract_images_from_zips() (ZIP, bytes-basiert) und extract_loose_images() (lose
    Datei, Path-basiert) zu einer einheitlichen Liste - Quellformat ist danach egal, beides läuft
    über denselben _open_image()/_prepare_image_for_compositing()-Pfad weiter.

    Returns:
        Liste von (image_name, source: bytes|Path, world_info|None)
    """
    return extract_images_from_zips(aerial_dir) + extract_loose_images(aerial_dir)


def _open_image(source):
    """Öffnet ein Quellbild - source ist entweder bytes (aus einem ZIP) oder ein Path (lose Datei)."""
    return Image.open(BytesIO(source)) if isinstance(source, (bytes, bytearray)) else Image.open(source)


def _reproject_image_to_source_crs(source, dst_epsg):
    """
    Reprojiziert ein einzelnes georeferenziertes Bild nach dst_epsg via rasterio (Vorbild: die
    bereits vorhandene Reprojektions-Logik in terrain/horizon_image.py::build_horizon_image()).

    Args:
        source: bytes (aus einem ZIP) oder Path/str (lose Datei)
        dst_epsg: Ziel-EPSG-Code

    Returns:
        (PIL.Image RGB, world_info) im Ziel-CRS
    """
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    def _reproject(src):
        transform, width, height = calculate_default_transform(
            src.crs, f"EPSG:{dst_epsg}", src.width, src.height, *src.bounds
        )
        dst = np.zeros((3, height, width), dtype=src.dtypes[0])
        for band in range(1, min(src.count, 3) + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=dst[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=f"EPSG:{dst_epsg}",
                resampling=Resampling.bilinear,
            )
        image = Image.fromarray(np.moveaxis(dst, 0, -1)).convert("RGB")
        world_info = {
            "pixel_size_x": transform.a,
            "pixel_size_y": transform.e,
            "x_origin": transform.c,
            "y_origin": transform.f,
            "crs_epsg": dst_epsg,
        }
        return image, world_info

    if isinstance(source, (bytes, bytearray)):
        with rasterio.io.MemoryFile(source) as memfile, memfile.open() as src:
            return _reproject(src)
    with rasterio.open(source) as src:
        return _reproject(src)


def _prepare_image_for_compositing(source, world_info, dst_epsg):
    """
    Öffnet ein Quellbild fürs Compositing und reprojiziert es bei Bedarf ins Ziel-CRS. Für
    .tfw-Paare (world_info["crs_epsg"] is None, angenommene Quell-CRS wie bisher) wird NIE
    reprojiziert - kein Verhaltensunterschied für den bestehenden LGL-BW-Pfad.

    Returns:
        (PIL.Image, world_info) - world_info unverändert, außer bei Reprojektion (dann die im
        Ziel-CRS neu berechnete Georeferenz)
    """
    src_epsg = world_info.get("crs_epsg")
    if src_epsg is not None and src_epsg != dst_epsg:
        return _reproject_image_to_source_crs(source, dst_epsg)
    return _open_image(source), world_info


def enhance_dop20_image(image, contrast_factor=1.18, brightness_factor=0.92, color_factor=1.12):
    """
    Verbessert DOP20-Bilder für naturgetreuere Darstellung in BeamNG.

    DOP20 Bilder sind oft zu blass und zu hell - diese Funktion erhöht:
    - Kontrast (mehr Dynamik)
    - Farbnättigung (lebendiger)
    - Reduziert Helligkeit (naturgetreuer)

    Args:
        image: PIL Image
        contrast_factor: Kontrast-Multiplikator (1.18 = +18%, Standard)
        brightness_factor: Helligkeit-Multiplikator (0.92 = -8%, Standard)
        color_factor: Farbnättigung-Multiplikator (1.12 = +12%, Standard)

    Returns:
        Verbessertes PIL Image
    """
    # Stelle sicher, dass Bild RGB ist
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Erhöhe Kontrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Reduziere Helligkeit (dunkler)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Erhöhe Farbnättigung
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    return image


AERIAL_PHOTO_FILENAME = "aerial_photo.png"


def process_aerial_images(aerial_dir, output_dir, grid_bounds, global_offset, target_pixel_size=None):
    """
    Setzt alle Luftbilder zu EINEM zusammenhängenden Foto für die gesamte
    grid_bounds-Fläche zusammen (statt vieler kleiner 500m-Kacheln).

    Hintergrund (Recherche 2026-09-18): BeamNGs v1.5-Terrain-Material-System
    ist laut offizieller Doku für eine KLEINE Anzahl wiederholender Materialien
    ausgelegt ("keep terrain material counts much lower than the technical
    limit"), nicht für viele (16-25) einzigartige 4096px-Texturen. Mit vielen
    großen, einzigartigen Materialien hat BeamNGs Textur-Atlas-Packer beim
    Verpacken einzelne Kacheln verdreht dargestellt, obwohl die Quelldateien
    nachweislich korrekt waren (jede für sich UND als zusammengesetztes Mosaik
    lückenlos). Mit nur einem Material für die gesamte Fläche entfällt dieses
    Packing-Problem komplett.

    Nutzt .tfw World Files zur exakten Positionierung jedes Quellbilds auf
    einer gemeinsamen Leinwand in nativer Auflösung, skaliert das Ergebnis
    danach auf target_pixel_size herunter.

    Args:
        aerial_dir: Verzeichnis mit ZIP-Archiven
        output_dir: Zielverzeichnis für das zusammengesetzte Foto
        grid_bounds: (min_x, max_x, min_y, max_y) in lokalen Koordinaten
        global_offset: (utm_x, utm_y, utm_z) tuple - UTM Offset für Koordinaten-Transformation
        target_pixel_size: Kantenlänge (Pixel) des Ausgabefotos (Default: config.TERRAIN_BASE_TEX_PIXEL_SIZE)

    Returns:
        1 wenn ein Foto gespeichert wurde, sonst 0
    """
    if target_pixel_size is None:
        target_pixel_size = config.TERRAIN_BASE_TEX_PIXEL_SIZE

    from ..geometry.coordinates import get_source_crs_epsg

    dst_epsg = get_source_crs_epsg()

    images = extract_georeferenced_images(aerial_dir)
    if not images:
        logger.debug("  [i] Keine Luftbilder gefunden")
        return 0

    images_with_geo = [(name, data, info) for name, data, info in images if info is not None]
    if not images_with_geo:
        logger.error(f"  [!] Keine Georeferenzierung gefunden (fehlen .tfw-Dateien oder eingebettete GeoTIFF-Tags?)")
        return 0

    logger.debug(f"  [i] {len(images_with_geo)} Luftbilder mit Georeferenzierung gefunden")

    grid_min_x, grid_max_x, grid_min_y, grid_max_y = grid_bounds
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y
    offset_x, offset_y = global_offset[:2]

    # Native Auflösung als Referenz für die Leinwand (alle DOP20-Kacheln einer
    # Region haben dieselbe Pixelgröße, z.B. 0.2m/px).
    native_pixel_size = abs(images_with_geo[0][2]["pixel_size_x"])
    canvas_w = max(1, round(grid_width / native_pixel_size))
    canvas_h = max(1, round(grid_height / native_pixel_size))
    logger.info(
        f"  [i] Baue Gesamt-Luftbild: {grid_width:.0f}m x {grid_height:.0f}m "
        f"@ {native_pixel_size}m/px = {canvas_w}x{canvas_h}px nativ -> {target_pixel_size}x{target_pixel_size}px"
    )

    # Füllfarbe für evtl. Lücken (keine Luftbild-Deckung) - gedecktes Grün statt
    # Schwarz/Magenta, damit fehlende Randbereiche nicht grell auffallen.
    canvas = Image.new("RGB", (canvas_w, canvas_h), (70, 95, 55))

    pasted = 0
    for img_name, img_data, world_info in images_with_geo:
        try:
            image, world_info = _prepare_image_for_compositing(img_data, world_info, dst_epsg)
            image = enhance_dop20_image(image)
            pixel_size = abs(world_info["pixel_size_x"])

            # .tfw-Ursprung ist die obere linke (nordwestliche) Pixelecke.
            img_local_x = world_info["x_origin"] - offset_x
            img_local_y = world_info["y_origin"] - offset_y

            if not math.isclose(pixel_size, native_pixel_size, rel_tol=1e-6):
                scale = pixel_size / native_pixel_size
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )

            # Position auf der Leinwand: Ursprung der Leinwand ist die
            # nordwestliche Ecke (grid_min_x, grid_max_y), Zeile 0 = Norden -
            # Standard-Bildkonvention, keine Kachel-Bucket-Arithmetik mehr nötig.
            px = round((img_local_x - grid_min_x) / native_pixel_size)
            py = round((grid_max_y - img_local_y) / native_pixel_size)

            canvas.paste(image, (px, py))
            pasted += 1
        except Exception as e:
            logger.error(f"  [!] Fehler beim Verarbeiten von {img_name}: {e}")
            continue

    if pasted == 0:
        logger.error("  [!] Keine Luftbilder konnten platziert werden")
        return 0

    canvas = canvas.resize((target_pixel_size, target_pixel_size), Image.Resampling.LANCZOS)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / AERIAL_PHOTO_FILENAME
    canvas.save(filepath, "PNG")

    logger.info(f"  [OK] Gesamt-Luftbild aus {pasted} Quellbildern gespeichert: {filepath}")
    return 1


AERIAL_SIGNATURE_FILENAME = "aerial_photo.json"
AERIAL_SIGNATURE_VERSION = 2
SINGLE_PHOTO_NAME = AERIAL_PHOTO_FILENAME[: -len(".png")]  # "aerial_photo"


def process_aerial_tiles(aerial_dir, output_dir, photos, global_offset, target_pixel_size=None):
    """
    Vier-Bilder-Modus: baut pro Eintrag in `photos` ein eigenes Luftbild (ein Foto je DGM1-Kachel).

    Jedes Quellbild wird nur EINMAL gelesen und verbessert und dann in alle Fotos gesetzt, die es berührt
    (ein Quellbild kann über eine Kachelgrenze reichen). Positioniert wird wie beim Gesamtfoto über die
    .tfw-Georeferenz; jedes Foto wird von der nativen Auflösung (0,2 m/px) auf target_pixel_size skaliert.

    Args:
        photos: [{"name": "aerial_photo_0", "bounds": (x_min, x_max, y_min, y_max)}] in lokalen Koordinaten
        global_offset: (utm_x, utm_y, ...) für die Umrechnung der Quellbild-Ursprünge nach lokal

    Returns:
        Anzahl der gespeicherten Fotos
    """
    if target_pixel_size is None:
        target_pixel_size = config.TERRAIN_BASE_TEX_PIXEL_SIZE

    from ..geometry.coordinates import get_source_crs_epsg

    dst_epsg = get_source_crs_epsg()

    images = [(n, d, i) for n, d, i in extract_georeferenced_images(aerial_dir) if i is not None]
    if not images:
        logger.error("  [!] Keine georeferenzierten Luftbilder gefunden")
        return 0

    offset_x, offset_y = global_offset[:2]
    native = abs(images[0][2]["pixel_size_x"])

    canvases = []
    for photo in photos:
        x_min, x_max, y_min, y_max = photo["bounds"]
        size = (max(1, round((x_max - x_min) / native)), max(1, round((y_max - y_min) / native)))
        canvases.append(Image.new("RGB", size, (70, 95, 55)))  # gedecktes Grün für Lücken
        logger.info(f"  [i] Baue {photo['name']}: {x_max - x_min:.0f}m x {y_max - y_min:.0f}m @ {native}m/px = {size[0]}x{size[1]}px -> {target_pixel_size}px")

    for img_name, img_data, world_info in images:
        try:
            image, world_info = _prepare_image_for_compositing(img_data, world_info, dst_epsg)
            image = enhance_dop20_image(image)
            pixel_size = abs(world_info["pixel_size_x"])
            if not math.isclose(pixel_size, native, rel_tol=1e-6):
                scale = pixel_size / native
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS
                )
            img_x = world_info["x_origin"] - offset_x  # .tfw-Ursprung = obere linke (nordwestliche) Pixelecke
            img_y = world_info["y_origin"] - offset_y
            for photo, canvas in zip(photos, canvases):
                x_min, _, _, y_max = photo["bounds"]
                px = round((img_x - x_min) / native)
                py = round((y_max - img_y) / native)
                if px >= canvas.width or py >= canvas.height or px + image.width <= 0 or py + image.height <= 0:
                    continue  # Quellbild liegt außerhalb dieser Kachel
                canvas.paste(image, (px, py))  # PIL schneidet an den Rändern ab
        except Exception as e:
            logger.error(f"  [!] Fehler beim Verarbeiten von {img_name}: {e}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for photo, canvas in zip(photos, canvases):
        canvas.resize((target_pixel_size, target_pixel_size), Image.Resampling.LANCZOS).save(
            Path(output_dir) / f"{photo['name']}.png", "PNG"
        )
        saved += 1
    logger.info(f"  [OK] {saved} Kachel-Luftbilder gespeichert")
    return saved


def _aerial_source_files(aerial_dir):
    """ZIPs UND lose Rasterdateien (*.tif/*.tiff) - beide gelten als Quellbilder (siehe extract_georeferenced_images())."""
    p = Path(aerial_dir)
    return sorted(p.glob("*.zip")) + sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))


def aerial_photos_signature(aerial_dir, photos, global_offset, target_pixel_size=None):
    """
    Beschreibt, WOFÜR die Luftbilder gebaut wurden: welche Fotos (Name + Fläche), Ursprung, Auflösung, Quellbilder.

    Ohne diese Angabe erkennt der Exporter veraltete Fotos nicht - z.B. das 2-km-Foto einer einzelnen
    DGM1-Kachel, das nach dem Umstellen auf vier Kacheln (4 km) einfach auf die doppelte Fläche gestreckt würde.
    """
    if target_pixel_size is None:
        target_pixel_size = config.TERRAIN_BASE_TEX_PIXEL_SIZE
    sources = [[path.name, path.stat().st_size] for path in _aerial_source_files(aerial_dir)]
    return {
        "version": AERIAL_SIGNATURE_VERSION,
        "photos": [{"name": p["name"], "bounds": [round(float(v), 3) for v in p["bounds"]]} for p in photos],
        "global_offset": [round(float(v), 3) for v in global_offset[:2]],
        "target_pixel_size": int(target_pixel_size),
        "sources": sources,
    }


def write_aerial_photo_signature(output_dir, signature):
    (Path(output_dir) / AERIAL_SIGNATURE_FILENAME).write_text(json.dumps(signature, indent=2), encoding="utf-8")


def aerial_photo_is_current(output_dir, signature):
    """True, wenn ALLE Fotos existieren und genau mit dieser Signatur gebaut wurden (Fotos ohne Signatur gelten als veraltet)."""
    signature_file = Path(output_dir) / AERIAL_SIGNATURE_FILENAME
    if not signature_file.exists():
        return False
    photos = signature.get("photos") or [{"name": SINGLE_PHOTO_NAME}]
    if not all((Path(output_dir) / f"{p['name']}.png").exists() for p in photos):
        return False
    try:
        return json.loads(signature_file.read_text(encoding="utf-8")) == signature
    except (OSError, ValueError):
        return False


def _remove_stale_photos(output_dir, keep_names):
    """Entfernt Fotos des jeweils anderen Modus (aerial_photo.png bzw. aerial_photo_<k>.png) - je ca. 130 MB."""
    import re

    for path in Path(output_dir).glob("aerial_photo*.png"):
        if re.fullmatch(r"aerial_photo(_\d+)?\.png", path.name) and path.stem not in keep_names:
            path.unlink()
            logger.info(f"  [i] Veraltetes Luftbild entfernt: {path.name}")


def ensure_aerial_photos(aerial_dir, output_dir, photos, global_offset, target_pixel_size=None):
    """
    Baut die Luftbilder nur, wenn sie fehlen oder nicht zur aktuellen Fläche/Kachelaufteilung passen.

    Args:
        photos: [{"name", "bounds"}]; ein einziger Eintrag "aerial_photo" = Gesamtfoto, sonst ein Foto je Kachel

    Returns:
        "current" (passt, nichts zu tun), "built" (neu gebaut), "failed" (Bauen fehlgeschlagen)
        oder "none" (keine Quellbilder - bestehende Fotos bleiben unverändert)
    """
    if not Path(aerial_dir).exists() or not _aerial_source_files(aerial_dir):
        return "none"

    signature = aerial_photos_signature(aerial_dir, photos, global_offset, target_pixel_size)
    names = {p["name"] for p in photos}
    if aerial_photo_is_current(output_dir, signature):
        _remove_stale_photos(output_dir, names)
        return "current"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if len(photos) == 1 and photos[0]["name"] == SINGLE_PHOTO_NAME:
        built = process_aerial_images(aerial_dir, output_dir, photos[0]["bounds"], global_offset, target_pixel_size)
    else:
        built = process_aerial_tiles(aerial_dir, output_dir, photos, global_offset, target_pixel_size)
    if built <= 0:
        return "failed"
    write_aerial_photo_signature(output_dir, signature)
    _remove_stale_photos(output_dir, names)
    return "built"


MINIMAP_SUBDIR = "minimap"
MINIMAP_FILENAME = "terrain.png"


def build_minimap_image(textures_dir, output_path, photos, terrain_bounds_local, target_pixel_size=None):
    """
    Baut das BigMap-Vorschaubild (info.json-Feld "minimap") aus den bereits gebauten Luftbild-PNGs
    (aerial_photo*.png in textures_dir, siehe ensure_aerial_photos()) - liest keine Quellbilder erneut ein,
    sondern setzt nur die fertigen Fotos verkleinert auf eine gemeinsame Leinwand.

    Selbe Konvention wie process_aerial_images()/process_aerial_tiles(): Zeile 0 = Norden, Leinwand-Ursprung
    = (terrain_bounds_local[0], terrain_bounds_local[3]) = (x_min, y_max), gedecktes Grün für Lücken.

    Args:
        textures_dir: Verzeichnis mit den fertigen aerial_photo*.png (config.BEAMNG_DIR_TEXTURES)
        output_path: Ziel-PNG-Pfad
        photos: [{"name", "bounds": (x_min, x_max, y_min, y_max)}] - dieselbe Liste wie an ensure_aerial_photos()
        terrain_bounds_local: (x_min, x_max, y_min, y_max) der GESAMTEN Terrain-Fläche in lokalen Koordinaten
        target_pixel_size: Kantenlänge (Pixel) der Minimap (Default: config.MINIMAP_PIXEL_SIZE)

    Returns:
        True bei Erfolg, False wenn ein Quellfoto fehlt (kein Ausnahmefehler - Minimap ist optional)
    """
    if target_pixel_size is None:
        target_pixel_size = config.MINIMAP_PIXEL_SIZE

    x_min, x_max, y_min, y_max = terrain_bounds_local
    width_m, height_m = x_max - x_min, y_max - y_min
    if width_m <= 0 or height_m <= 0:
        return False

    px_per_m_x = target_pixel_size / width_m
    px_per_m_y = target_pixel_size / height_m
    canvas = Image.new("RGB", (target_pixel_size, target_pixel_size), (70, 95, 55))  # gedecktes Grün für Lücken

    for photo in photos:
        source_path = Path(textures_dir) / f"{photo['name']}.png"
        if not source_path.exists():
            return False

        bx_min, bx_max, by_min, by_max = photo["bounds"]
        tile_w = max(1, round((bx_max - bx_min) * px_per_m_x))
        tile_h = max(1, round((by_max - by_min) * px_per_m_y))

        with Image.open(source_path) as source:
            tile = source.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)

        px = round((bx_min - x_min) * px_per_m_x)
        py = round((y_max - by_max) * px_per_m_y)
        canvas.paste(tile, (px, py))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")
    return True


def minimap_info_json_fields(x_min, y_max, size_m, relative_file=None):
    """
    info.json-Felder für die Minimap: "size" (Terrain-Ausdehnung) und "minimap" (Bild + Lage).

    Args:
        x_min, y_max: Nordwest-Ecke der Terrain-Fläche in lokalen Koordinaten (= Bild-Ursprung, Zeile 0 = Norden)
        size_m: Kantenlänge der (quadratischen) Terrain-Fläche in Metern
        relative_file: Pfad relativ zum Level-Root (Default: "{MINIMAP_SUBDIR}/{MINIMAP_FILENAME}")

    Returns:
        {"size": [...], "minimap": [...]} zum Zusammenführen in ItemManager.set_info_json_fields()
    """
    file = relative_file or f"{MINIMAP_SUBDIR}/{MINIMAP_FILENAME}"
    return {
        "size": [size_m, size_m],
        "minimap": [{"file": file, "size": [size_m, size_m], "offset": [x_min, y_max]}],
    }


POI_PREVIEW_SUBDIR = "spawn_previews"


def _photo_containing(photos, xy):
    """Foto-Kachel, deren bounds xy enthalten - sonst die mit dem nächstgelegenen Mittelpunkt (Fallback für
    einen POI hart an der Kachelkante/knapp außerhalb durch Rundung)."""
    x, y = xy
    for photo in photos:
        bx_min, bx_max, by_min, by_max = photo["bounds"]
        if bx_min <= x <= bx_max and by_min <= y <= by_max:
            return photo

    def _center_dist(photo):
        bx_min, bx_max, by_min, by_max = photo["bounds"]
        return math.hypot(x - (bx_min + bx_max) / 2.0, y - (by_min + by_max) / 2.0)

    return min(photos, key=_center_dist) if photos else None


def build_poi_preview_image(textures_dir, output_path, photos, position_xy, crop_size_m=None, target_pixel_size=None):
    """
    Vorschaubild für einen POI-Spawn-Punkt (info.json spawnPoints[].preview, siehe
    lua/ge/extensions/core/levels.lua): quadratischer Ausschnitt aus dem bereits gebauten Luftbild,
    Draufsicht, POI mittig - liest kein Quellbild erneut ein, nur die fertigen aerial_photo*.png
    (siehe ensure_aerial_photos()), dieselbe Konvention wie build_minimap_image() (Zeile 0 = Norden).

    Args:
        textures_dir: Verzeichnis mit den fertigen aerial_photo*.png (config.BEAMNG_DIR_TEXTURES)
        output_path: Ziel-Bildpfad (.jpg)
        photos: [{"name", "bounds": (x_min, x_max, y_min, y_max)}] - dieselbe Liste wie an ensure_aerial_photos()
        position_xy: (x, y) des POI in lokalen Koordinaten
        crop_size_m: Kantenlänge (Meter) des Ausschnitts (Default: config.POI_PREVIEW_CROP_SIZE_M)
        target_pixel_size: Kantenlänge (Pixel) des gespeicherten Bilds (Default: config.POI_PREVIEW_PIXEL_SIZE)

    Returns:
        True bei Erfolg, False wenn keine passende Foto-Kachel gefunden/gelesen werden konnte (das
        Vorschaubild ist optional - BeamNG fällt sonst auf das Level-Vorschaubild zurück)
    """
    if crop_size_m is None:
        crop_size_m = config.POI_PREVIEW_CROP_SIZE_M
    if target_pixel_size is None:
        target_pixel_size = config.POI_PREVIEW_PIXEL_SIZE

    photo = _photo_containing(photos, position_xy)
    if photo is None:
        return False
    source_path = Path(textures_dir) / f"{photo['name']}.png"
    if not source_path.exists():
        return False

    bx_min, bx_max, by_min, by_max = photo["bounds"]
    width_m, height_m = bx_max - bx_min, by_max - by_min
    if width_m <= 0 or height_m <= 0:
        return False

    x, y = position_xy
    half = crop_size_m / 2.0
    try:
        with Image.open(source_path) as source:
            px_per_m_x = source.width / width_m
            px_per_m_y = source.height / height_m

            left = (x - half - bx_min) * px_per_m_x
            right = (x + half - bx_min) * px_per_m_x
            top = (by_max - (y + half)) * px_per_m_y  # Zeile 0 = Norden
            bottom = (by_max - (y - half)) * px_per_m_y

            # An den Bild-Rand klemmen (POI nahe der Kachel-Kante): Ausschnitt bleibt im Bild, ist dann
            # nur nicht mehr exakt mittig - besser als ein leeres/abgeschnittenes Vorschaubild.
            left, right = max(0.0, left), min(float(source.width), right)
            top, bottom = max(0.0, top), min(float(source.height), bottom)
            if right - left < 2 or bottom - top < 2:
                return False

            crop = source.convert("RGB").crop((round(left), round(top), round(right), round(bottom)))
            crop = crop.resize((target_pixel_size, target_pixel_size), Image.Resampling.LANCZOS)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path, "JPEG", quality=85)
    except OSError as exc:
        logger.debug(f"  [POI-Preview] {source_path} übersprungen: {exc}")
        return False
    return True
