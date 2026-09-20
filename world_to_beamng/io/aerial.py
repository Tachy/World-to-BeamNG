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


def extract_images_from_zips(aerial_dir="data/DOP20"):
    """
    Extrahiert alle Bilder mit Georeferenzierung aus ZIP-Dateien.

    Args:
        aerial_dir: Pfad zum DOP20-Verzeichnis

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

                    images.append((img_file, img_data, world_info))

        except Exception as e:
            logger.error(f"[!] Fehler beim Lesen von {zip_path.name}: {e}")

    return images


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

    images = extract_images_from_zips(aerial_dir)
    if not images:
        logger.debug("  [i] Keine Luftbilder gefunden")
        return 0

    images_with_geo = [(name, data, info) for name, data, info in images if info is not None]
    if not images_with_geo:
        logger.error(f"  [!] Keine Georeferenzierung gefunden (fehlen .tfw-Dateien?)")
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
            image = Image.open(BytesIO(img_data))
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

    images = [(n, d, i) for n, d, i in extract_images_from_zips(aerial_dir) if i is not None]
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
            image = enhance_dop20_image(Image.open(BytesIO(img_data)))
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


def aerial_photos_signature(aerial_dir, photos, global_offset, target_pixel_size=None):
    """
    Beschreibt, WOFÜR die Luftbilder gebaut wurden: welche Fotos (Name + Fläche), Ursprung, Auflösung, Quellbilder.

    Ohne diese Angabe erkennt der Exporter veraltete Fotos nicht - z.B. das 2-km-Foto einer einzelnen
    DGM1-Kachel, das nach dem Umstellen auf vier Kacheln (4 km) einfach auf die doppelte Fläche gestreckt würde.
    """
    if target_pixel_size is None:
        target_pixel_size = config.TERRAIN_BASE_TEX_PIXEL_SIZE
    sources = [[path.name, path.stat().st_size] for path in sorted(Path(aerial_dir).glob("*.zip"))]
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
    if not Path(aerial_dir).exists() or not any(Path(aerial_dir).glob("*.zip")):
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
