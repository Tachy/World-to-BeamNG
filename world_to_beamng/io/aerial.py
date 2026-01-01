"""
Aerial image processing - Extrahiert und kachelt Luftbildaufnahmen.
"""

import zipfile
from pathlib import Path
from PIL import Image
from io import BytesIO


def extract_images_from_zips(aerial_dir="aerial"):
    """
    Extrahiert alle Bilder aus ZIP-Dateien im aerial-Verzeichnis.
    
    Args:
        aerial_dir: Pfad zum aerial-Verzeichnis
        
    Returns:
        List von (image_name, image_data_bytes) Tupeln
    """
    aerial_path = Path(aerial_dir)
    images = []
    
    if not aerial_path.exists():
        print(f"[!] Verzeichnis {aerial_dir} existiert nicht")
        return images
    
    zip_files = list(aerial_path.glob("*.zip"))
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Finde Bilddateien (TIF, TIFF, JPG, JPEG, PNG)
                image_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
                image_files = [
                    f for f in file_list 
                    if any(f.lower().endswith(ext) for ext in image_extensions)
                ]
                
                for img_file in image_files:
                    img_data = zip_ref.read(img_file)
                    images.append((img_file, img_data))
                    
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
                padded = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded
            
            tiles.append(tile)
    
    return tiles


def process_aerial_images(aerial_dir, output_dir, tile_size=1000):
    """
    Verarbeitet alle Luftbilder: Extrahiert aus ZIPs, kachelt und speichert als JPGs.
    
    Args:
        aerial_dir: Verzeichnis mit ZIP-Archiven
        output_dir: Zielverzeichnis für Kacheln
        tile_size: Kachelgröße in Pixeln (Standard: 1000)
        
    Returns:
        Anzahl gespeicherter Kacheln
    """
    # Extrahiere Bilder aus ZIPs
    images = extract_images_from_zips(aerial_dir)
    
    if not images:
        print("  [i] Keine Luftbilder gefunden")
        return 0
    
    print(f"  [i] {len(images)} Luftbilder gefunden")
    
    # Erstelle Ausgabeverzeichnis
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Verarbeite jedes Bild
    tile_counter = 0
    
    for img_name, img_data in images:
        try:
            # Lade Bild
            image = Image.open(BytesIO(img_data))
            
            # Kachle Bild
            tiles = tile_image(image, tile_size=tile_size)
            
            # Speichere Kacheln
            for tile_img in tiles:
                filename = f"tile_{tile_counter:04d}.jpg"
                filepath = output_path / filename
                
                # Konvertiere zu RGB falls nötig
                if tile_img.mode != 'RGB':
                    tile_img = tile_img.convert('RGB')
                
                # Speichere als JPG
                tile_img.save(filepath, 'JPEG', quality=85, optimize=True)
                tile_counter += 1
            
        except Exception as e:
            print(f"  [!] Fehler beim Verarbeiten von {img_name}: {e}")
            continue
    
    print(f"  [OK] {tile_counter} Kacheln gespeichert in {output_path}")
    return tile_counter
