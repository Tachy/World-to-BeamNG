"""
DAE-Viewer Steuerung und Layer-Management - Verbessert

STEUERUNG:
==========

Hauptschalter:
  X = Toggle zwischen Rendering-Ansicht (Texturen) und Grid-Ansicht (Debug-Layer)
      → triggert NEUAUFBAU der Geometrie
      → nur dieser Befehl verursacht einen Neuaufbau!

In Grid-Ansicht (wenn X aus):
  Alle Layer können EINZELN ein-/ausgeblendet werden (OHNE Neuaufbau):
  
  T = Terrain ein-/aus
  S = Straßen (Roads) ein-/aus
  H = Häuser (Buildings) ein-/aus
  D = Debug-Layer ein-/aus (Junctions, Centerlines, Boundary-Polygons)

Kamera und Navigation:
  K = Gespeicherte Kamera-Position laden
  Shift+K = Aktuelle Kamera-Position speichern
  L = DAE-Datei neuladen
  Up/Down-Tasten = Zoom anpassen

Maus:
  Rechtsklick + Ziehen = Kamera rotieren
  Mausrad = Zoom

ARCHITEKTUR:
============

Layer-Management:
  - Jeder Layer hat einen Show/Hide-Flag (show_terrain, show_roads, show_buildings, show_debug)
  - Diese Flags sind PERSISTENT (werden in dae_viewer.cfg gespeichert)
  - Beim Start werden gespeicherte Zustände wiederhergestellt

Sichtbarkeits-Updates:
  - _update_visibility(): Ändert Sichtbarkeit ohne Neuaufbau
  - _update_debug_visibility(): Spezial für Debug-Layer (lädt ihn beim Erst-Toggle)
  - _update_active_layers_text(): Zeigt aktive Layer oben rechts an

Neuaufbau vermeiden:
  - Only update_view() triggert Neuaufbau (für X-Taste: Textur-Toggle)
  - Alle anderen Toggles (T, S, H, D) nutzen nur _update_visibility()
  - Debug-Layer wird einmalig beim Erst-Toggle geladen, dann nur Sichtbarkeit gewechselt

Status-Display:
  - Oben rechts: Zeigt aktive Layer als "T S H X D" (nur wenn aktiv)
  - Oben links: Bedienung und Hinweise
  - Unten links: Kamera-Position, Rotation, Zoom

VERBESSERTE FUNKTIONEN:
=======================

1. Häuser-Toggle (H):
   - Neu hinzugefügt
   - Ermöglicht separate Kontrolle der Gebäudeebene
   - Funktioniert in beiden Ansichten (Rendering + Grid)

2. Kein überflüssiger Neuaufbau:
   - T, S, H, D triggern KEINEN Neuaufbau mehr
   - Nur X (Textur-Toggle) triggert Neuaufbau
   - Performance und Stabilität verbessert

3. Bessere Dokumentation:
   - Klare Steuerungshinweise beim Start
   - Docstring am Anfang zeigt komplette Steuerung
   - Konsistente Benennung (Terrain, Straßen, Häuser, Debug)

4. Debug-Layer Verbesserungen:
   - Nur in Grid-Ansicht verfügbar (wenn X aus)
   - Wird einmalig geladen, dann schnell umgeschaltet
   - Keine Lagz-Probleme bei häufigen Toggles
"""

# VERWENDUNGSBEISPIELE:
# =====================

# Grid-Ansicht aktivieren und nur Straßen + Debug zeigen:
# 1. X drücken (zu Grid wechseln)
# 2. T drücken (Terrain aus)
# 3. H drücken (Häuser aus)
# 4. D drücken (Debug an)
# Result: Nur Straßen und Debug sichtbar

# Alle Layer auf einmal aus:
# 1. X drücken (zu Grid wechseln)
# 2. T drücken (Terrain aus)
# 3. S drücken (Straßen aus)
# 4. H drücken (Häuser aus)
# 5. D drücken (Debug aus)
# Result: Leere Ansicht, gutes Testing für Kamerapositionen

# Rendering-Ansicht mit Texturen:
# X drücken → automatisch zu Rendering gewechselt
# Alle Layer sichtbar, keine Debug-Overlays
