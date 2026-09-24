# data/satellite

Luftbilder/Satellitenbilder (RGB-Orthofotos) für die Terrain-Textur. **Erforderlich** – ohne Daten
hier bekommt das Terrain nur eine grüne Füllfarbe statt einer Foto-Textur, kein brauchbares
Ergebnis.

- LGL Baden-Württemberg: ZIP-Dateien mit TIF + TFW, Name `dop20rgb_32_<x>_<y>_2_bw.zip`.
- Andere Region: ein oder mehrere georeferenzierte Orthofotos (eingebettetes GeoTIFF-Tag oder
  begleitende `.tfw`-Datei), beliebiger Dateiname, lose oder im ZIP.

Muss dieselbe Fläche wie `data/height` abdecken.
