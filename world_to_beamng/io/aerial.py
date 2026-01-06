"""
Aerial image processing - Extrahiert und kachelt Luftbildaufnahmen.
"""

import zipfile
import math
from pathlib import Path
from PIL import Image
from io import BytesIO


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
        print(f"[!] Verzeichnis {aerial_dir} existiert nicht")
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
            print(f"[!] Fehler beim Lesen von {zip_path.name}: {e}")

    return images


def tile_image(image, tile_size=1000):
    """
    Teilt ein Bild in tile_size x tile_size Pixel große Kacheln auf.

    Args:
        image: PIL Image
        tile_size: Kachelgröße in Pixeln

    Returns:
        List von PIL Image Objekten (Kacheln)
    """
    width, height = image.size
    tiles = []

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    for y_idx in range(tiles_y):
        for x_idx in range(tiles_x):
            x1 = x_idx * tile_size
            y1 = y_idx * tile_size
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)

            tile = image.crop((x1, y1, x2, y2))

            # Falls Kachel kleiner als tile_size, fülle mit schwarzem Rand auf
            if tile.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded

            tiles.append(tile)

    return tiles


def process_aerial_images(aerial_dir, output_dir, tile_size=1000, grid_bounds=None):
    """
    Verarbeitet alle Luftbilder: Extrahiert aus ZIPs, kachelt basierend auf Georeferenzierung.

    Verwendet .tfw World Files zur Bestimmung der Position jedes Luftbilds.
    Tiles werden basierend auf Weltkoordinaten benannt.

    Args:
        aerial_dir: Verzeichnis mit ZIP-Archiven
        output_dir: Zielverzeichnis für Kacheln
        tile_size: Kachelgröße in Pixeln (Standard: 1000)
        grid_bounds: (min_x, max_x, min_y, max_y) in lokalen Koordinaten (optional)

    Returns:
        Anzahl gespeicherter Kacheln
    """
    from world_to_beamng import config

    # Extrahiere Bilder mit Georeferenzierung
    images = extract_images_from_zips(aerial_dir)

    if not images:
        print("  [i] Keine Luftbilder gefunden")
        return 0

    # Filtere Bilder ohne World File Info
    images_with_geo = [(name, data, info) for name, data, info in images if info is not None]

    if not images_with_geo:
        print(f"  [!] Keine Georeferenzierung gefunden (fehlen .tfw-Dateien?)")
        return 0

    print(f"  [i] {len(images_with_geo)} Luftbilder mit Georeferenzierung gefunden")

    # Debug: Zeige erste Luftbild-Info
    if images_with_geo:
        first_name, first_data, first_info = images_with_geo[0]
        first_img = Image.open(BytesIO(first_data))
        print(f"  [DEBUG] Erstes Luftbild: {first_name}")
        print(f"    Größe: {first_img.size[0]}×{first_img.size[1]} Pixel")
        print(f"    Pixel-Größe: {first_info['pixel_size_x']}m/px")
        print(
            f"    Abdeckung: {first_img.size[0] * abs(first_info['pixel_size_x']):.0f}m × {first_img.size[1] * abs(first_info['pixel_size_y']):.0f}m"
        )
        print(f"    UTM Origin: ({first_info['x_origin']:.1f}, {first_info['y_origin']:.1f})")

    # Bestimme Grid-Bounds (in lokalen Koordinaten)
    if grid_bounds is None:
        grid_bounds = config.GRID_BOUNDS_LOCAL

    if grid_bounds is None:
        print("  [!] Keine Grid-Bounds verfügbar")
        return 0

    grid_min_x, grid_max_x, grid_min_y, grid_max_y = grid_bounds
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    # Local Offset (UTM -> lokal)
    if config.LOCAL_OFFSET is None:
        print("  [!] LOCAL_OFFSET nicht gesetzt")
        return 0

    offset_x, offset_y, offset_z = config.LOCAL_OFFSET

    # Tile-Größe in Weltkoordinaten (z.B. 400m)
    tile_world_size = config.TILE_SIZE  # 400m

    print(
        f"  [DEBUG] Grid Bounds (lokal): X=[{grid_min_x:.1f}..{grid_max_x:.1f}], Y=[{grid_min_y:.1f}..{grid_max_y:.1f}]"
    )
    print(f"  [DEBUG] Grid Größe: {grid_width:.1f}m × {grid_height:.1f}m")
    print(
        f"  [DEBUG] Erwartete Tiles: {int(grid_width/tile_world_size)}×{int(grid_height/tile_world_size)} = {int(grid_width/tile_world_size) * int(grid_height/tile_world_size)}"
    )
    print(f"  [DEBUG] LOCAL_OFFSET: ({offset_x:.1f}, {offset_y:.1f}, {offset_z:.1f})")

    # Erstelle Ausgabeverzeichnis
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tile_counter = 0
    unique_tiles = set()  # Tracking für eindeutige Tiles

    for img_name, img_data, world_info in images_with_geo:
        try:
            # Lade Bild
            image = Image.open(BytesIO(img_data))
            width, height = image.size

            # Welt-Koordinaten des Luftbilds (UTM)
            img_utm_x = world_info["x_origin"]
            img_utm_y = world_info["y_origin"]

            # Nutze .tfw Metadaten direkt für Pixel-Größe
            pixel_size = abs(world_info["pixel_size_x"])

            # Transformiere zu lokalen Koordinaten
            # .tfw Ursprung ist oben-links, Y wächst nach unten (negative Richtung in lokalen Koordinaten)
            img_local_x = img_utm_x - offset_x
            img_local_y = img_utm_y - offset_y  # Obere linke Ecke
            # Kachle Bild
            tiles = tile_image(image, tile_size=tile_size)
            print(
                f"    [DEBUG] Luftbild {img_name}: {width}x{height}px -> {len(tiles)} Tiles (tile_size={tile_size}px)"
            )

            # Berechne Anzahl Tiles pro Dimension
            tiles_per_row = (width + tile_size - 1) // tile_size

            # Speichere Kacheln mit Geo-basierter Indizierung
            for tile_idx, tile_img in enumerate(tiles):
                # Lokale Position innerhalb des Luftbilds
                local_x_idx = tile_idx % tiles_per_row
                local_y_idx = tile_idx // tiles_per_row

                # Lokale Position dieser Tile
                # Luftbild hat Ursprung oben-links, Y läuft nach unten in Pixeln
                tile_size_m = tile_size * pixel_size
                tile_local_x = img_local_x + (local_x_idx * tile_size_m)
                tile_local_y = img_local_y - (local_y_idx * tile_size_m)  # Obere Kante dieser Tile

                # Berechne globale Tile-Indizes relativ zum Grid
                # Nutze math.floor() für korrekte Behandlung negativer Zahlen
                global_x_idx = math.floor((tile_local_x - grid_min_x) / tile_world_size)
                global_y_idx = math.floor((tile_local_y - grid_min_y) / tile_world_size)
                if len(unique_tiles) < 8:
                    print(
                        f"      [{img_name[-30:]}] Tile({local_x_idx},{local_y_idx}) @ World({tile_local_x:.0f},{tile_local_y:.0f}) -> Grid-Idx({global_x_idx},{global_y_idx})"
                    )

                # Prüfe ob Tile innerhalb des Grids liegt
                tiles_x = int(grid_width / tile_world_size)
                tiles_y = int(grid_height / tile_world_size)

                if 0 <= global_x_idx < tiles_x and 0 <= global_y_idx < tiles_y:
                    filename = f"tile_{global_x_idx}_{global_y_idx}.jpg"
                    filepath = output_path / filename

                    # Prüfe ob dieses Tile bereits geschrieben wurde (in unique_tiles oder auf Disk)
                    if (global_x_idx, global_y_idx) in unique_tiles:
                        continue  # Skip - Tile existiert bereits

                    # Prüfe ob Datei bereits existiert
                    if filepath.exists():
                        unique_tiles.add((global_x_idx, global_y_idx))
                        continue  # Skip - Tile bereits auf Disk

                    # Konvertiere zu RGB falls nötig
                    if tile_img.mode != "RGB":
                        tile_img = tile_img.convert("RGB")

                    # Speichere als JPG
                    tile_img.save(filepath, "JPEG", quality=85, optimize=True)
                    tile_counter += 1
                    unique_tiles.add((global_x_idx, global_y_idx))

        except Exception as e:
            print(f"  [!] Fehler beim Verarbeiten von {img_name}: {e}")
            continue

    print(f"  [OK] {tile_counter} Kacheln gespeichert ({len(unique_tiles)} eindeutige Tiles)")
    print(
        f"  [i] Tile-Range: X=[{min(t[0] for t in unique_tiles)}..{max(t[0] for t in unique_tiles)}], Y=[{min(t[1] for t in unique_tiles)}..{max(t[1] for t in unique_tiles)}]"
    )
    print(f"  [OK] {tile_counter} Kacheln gespeichert in {output_path}")
    return tile_counter
