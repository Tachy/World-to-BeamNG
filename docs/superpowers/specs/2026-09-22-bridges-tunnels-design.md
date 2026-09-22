# Design: Brücken und Tunnel

Datum: 2026-09-22
Status: Entwurf (approved in Chat, wartet auf finales Review)

## Motivation

Die Straßen-Pipeline behandelt aktuell jede `highway=*`-Way identisch: die Höhe
jedes Centerline-Punkts kommt aus der rohen DGM-Abtastung an dieser Stelle
(`geometry/polygon.py::get_road_polygons()`), und
`terrain/road_embedding.py::embed_roads_into_heightmap()` stanzt exakt diese
Höhe ins Terrain-Raster. `bridge=*`- und `tunnel=*`-Tags werden komplett
ignoriert.

Für das aktuell laufende Gotthard-Export-Gebiet bedeutet das konkret:

- Der 16,9 km lange Gotthard-Straßentunnel (OSM-Way 49124512, `layer=-3`,
  `tunnel=yes`) bekommt die Geländehöhe der Bergoberfläche darüber zugewiesen
  und wird als Straße ins Terrain eingebettet - vermutlich eine "Phantom"-
  Straße entlang des Bergrückens.
- 71 Brücken-Ways (u.a. "Nuova strada del Passo del San Gottardo" mit
  teils gestapelten Fahrbahnen über `layer=1`/`layer=2`, sowie Abschnitte der
  "Tremola"-Passstraße) bekommen die rohe Talsohlenhöhe statt einer
  plausiblen Brückenhöhe.
- Eine Lawinengalerie (`tunnel=avalanche_protector`, "Galleria artificiale
  Piano dei buoi") wird wie eine normale Straße behandelt.

Ziel dieses Designs: Brücken, Tunnel und (talseitig offene) Galerien anhand
ihrer OSM-Tags erkennen, ihnen ein plausibles Höhenprofil geben, sie aus der
normalen Terrain-Einbettung herausnehmen und als eigene, generische 3D-Objekte
exportieren - allgemein nutzbar für jedes Exportgebiet, nicht nur Gotthard-
spezifisch.

## Nicht-Ziele (v1)

- Keine Terrain-Einschneidung an Tunnel-/Galerie-Portalen (die Heightmap
  bleibt dort unverändert; die Portal-Stirnfläche wird stattdessen an die
  natürliche Hangneigung angepasst, siehe Abschnitt 5).
- Keine Stapel-/Kollisionsauflösung für Brücken, die sich kreuzen
  (`layer`-Tag wird nicht ausgewertet; jede Brücken-Way wird unabhängig
  gebaut).
- Keine Geländer-Geometrie an Brücken (nur Deck-Fascia).
- Keine fotobasierte Beton-Textur; eine prozedurale reicht für v1.

## 1. Klassifizierung (gemeinsam für Brücke/Tunnel/Galerie)

Neues Modul `world_to_beamng/geometry/road_structures.py`:

```python
def classify_structure(osm_tags: dict) -> str:
    """"surface" | "bridge" | "tunnel" | "gallery""""
```

Regeln (angewendet in dieser Reihenfolge):

1. `bridge` in tags und Wert nicht leer/`"no"` → `"bridge"`
2. `tunnel == "avalanche_protector"` → `"gallery"`
3. `tunnel` in tags und Wert nicht leer/`"no"` → `"tunnel"`
4. sonst → `"surface"`

Gilt nur für `highway=*`-Ways (die bereits über `extract_roads_from_osm()`
gefiltert werden). Wasser-Tunnel/Culverts (`waterway=*` mit `tunnel=*`) sind
bereits über `terrain/water.py::UNDERGROUND_TAGS` separat ausgeschlossen und
nicht Teil dieses Designs.

Grundannahme (durch Stichprobe der Gotthard-OSM-Daten bestätigt): eine
einzelne OSM-Way ist tag-homogen - Brücken/Tunnel/Galerien sind in OSM
bereits als eigene Way-Segmente mit exakt diesen Grenzen angelegt, ein
`highway`-Way wechselt nicht mitten im Verlauf den Strukturtyp. Es ist daher
keine Segmentierung innerhalb einer Way nötig.

## 2. Höhenprofil

In `geometry/polygon.py::get_road_polygons()`, direkt nach dem
Elevation-Lookup (aktuell Schritt 3) und vor dem optionalen Chaikin-Smoothing
(Schritt 4): für jede Straße mit `classify_structure(osm_tags) != "surface"`
wird die Z-Spalte der Centerline-Punkte ersetzt durch lineare Interpolation
zwischen `coords[0].z` und `coords[-1].z`, gewichtet nach kumulierter
Bogenlänge entlang der (bereits resampleten) XY-Punkte.

Die Endpunkte selbst bleiben unverändert die rohe DGM-Höhe an der Stelle, wo
die Way beginnt/endet - dort schließt (laut OSM-Konvention) die normale
Straße an. Dadurch passt der Anschluss an die umgebenden `"surface"`-Straßen
automatisch, ohne zusätzliche Junction-Logik: beide Seiten werten denselben
Endpunkt gleich aus.

## 3. Terrain-Interaktion

In `workflow/terrain_workflow.py::process_tile()` wird `road_slope_polygons_2d`
vor den Aufrufen von `build_road_embankment_profiles()`,
`apply_embankment_blend()` und `embed_roads_into_heightmap()` in zwei Gruppen
gesplittet:

- `surface_roads` (`structure_type == "surface"`): durchläuft alle drei
  Funktionen unverändert wie bisher.
- `structure_roads` (`structure_type in ("bridge", "tunnel", "gallery")`):
  überspringt alle drei Funktionen komplett.

Das Terrain bleibt unter Brücken (Talboden) und über/neben Tunneln/Galerien
(Bergflanke) vollständig unverändert/natürlich.

Jedes Dict in `road_slope_polygons_2d` bekommt ein neues Feld
`"structure_type"` (gesetzt beim Aufbau der Liste in `process_tile()`).

## 4. Brücken-Mesh (`world_to_beamng/bridges/bridge_mesh.py`)

Folgt dem Extrusions-Muster aus `walls/wall_mesh.py`
(`MeshBuilder`, `offset_points`, Profil entlang der Centerline), liefert
dasselbe Mesh-Dict-Format (`{"id", "vertices", "uvs", "normals",
"faces": {material: [...]}}`) für `dae.export_multi_mesh()`.

- **Deck:** flache Platte, Breite = `OSM_MAPPER.get_road_properties(osm_tags)["width"]`
  + kleine seitliche Fascia-Kante, Dicke `config.BRIDGE_DECK_THICKNESS`.
  Oberseite bekommt das Straßenmaterial (`internal_name` aus
  `get_road_properties()`, gleiche Textur wie die angrenzenden
  DecalRoad-Segmente, UV entlang der Länge) - visuell nahtloser Übergang
  Fahrbahn → Brücke → Fahrbahn.
- **Pfeiler:** alle `config.BRIDGE_PIER_SPACING` Meter ein rechteckiger
  Stützpfeiler (Querschnitt `config.BRIDGE_PIER_SIZE`) von der
  Deck-Unterkante bis zur natürlichen Geländehöhe darunter (abgetastet auf
  der Heightmap **vor** jeder Straßen-Änderung, gleicher `ground_at`-Callback
  wie bei den Mauern). Material: prozedurale Beton-Textur (Abschnitt 6).
- Mehrere sich kreuzende Brücken-Ways werden unabhängig gebaut, keine
  Stapel-/Kollisionslogik (siehe Nicht-Ziele).

## 5. Tunnel-Mesh (`world_to_beamng/tunnels/tunnel_mesh.py`)

Rechteckige Röhre entlang des linear interpolierten Profils (Abschnitt 2):

- Boden: Straßenmaterial (wie beim Brücken-Deck).
- Wände + Decke: prozedurale Beton-Textur.
- Segment-Abstand `config.TUNNEL_SEGMENT_STEP` (deutlich gröber als die 1 m
  bei Mauern, z.B. 10 m - die Röhre folgt keinem unebenen Gelände, das hält
  die Vertex-Zahl auch bei 16,9 km im Rahmen).
- Breite `road_width + config.TUNNEL_WIDTH_MARGIN`, Höhe `config.TUNNEL_HEIGHT`.

**Portal (an beiden Enden):** die natürliche Hangneigung wird an der
Endposition aus der unveränderten Heightmap abgetastet (Gradient über
`config.TUNNEL_PORTAL_SLOPE_SAMPLE_DIST` Meter entlang und quer zur
Tunnelachse - gleiche Technik wie das links/rechts-Sampling der Böschung).
Die Stirnfläche der Röhre (Boden, Wände, Decke) wird entlang dieser
Hangneigungs-Ebene abgeschnitten statt rechtwinklig zur Achse - die Röhre
wirkt dadurch, als würde sie schräg aus dem Hang herauswachsen. Ein Portal-
Rahmen-Mesh wird passend zur geneigten Schnittkante ausgerichtet und dort
platziert. Keine Änderung der Terrain-Heightmap.

## 6. Galerie-Mesh (`world_to_beamng/tunnels/gallery_mesh.py`)

Wie Tunnel (gleiche Portal-Behandlung, gleicher Boden), aber pro
Centerline-Punkt wird die Talseite bestimmt: natürliche Geländehöhe links vs.
rechts der Centerline vergleichen (identische Technik wie
`build_road_embankment_profiles()`) - die Seite mit der niedrigeren
natürlichen Höhe ist die offene/Talseite. Dort wird keine Wand gebaut,
sondern nur Stützen im Abstand `config.GALLERY_COLUMN_SPACING`; Dach
(`config.GALLERY_ROOF_THICKNESS`) und bergseitige Wand bleiben geschlossen.

## 7. Neue Textur: prozedurales Beton-Material

`textures/gravel.py::generate_gravel_texture()` dient als Vorbild für eine
neue `generate_concrete_texture()`-Funktion (gleicher Mechanismus:
deterministisch erzeugt, einmalig in `data/textures` abgelegt, kein Foto
nötig). Registrierung in `textures/registry.py::REGISTRY` als neuer
`TextureSpec`, `required=lambda: config.BRIDGES_ENABLED or config.TUNNELS_ENABLED`.

## 8. Export-Integration

- `export_decal_roads()` (`terrain_workflow.py`) überspringt Straßen mit
  `structure_type != "surface"` - keine DecalRoad-Items für Brücken-/
  Tunnel-/Galerie-Segmente, deren Fahrbahn ist Teil der jeweiligen
  Struktur-Mesh.
- Neue Methoden `export_bridges(mesh_data)` und `export_tunnels(mesh_data)`
  (Letzteres deckt auch Galerien ab), analog zu `export_walls()`: je ein
  gemeinsames DAE + ein TSStatic-Item, Material-Registrierung über
  `MaterialManager`. Ohne aktivierte/vorhandene Objekte dieses Typs werden
  vorhandene `.dae`/`.cdae`-Reste entfernt (wie bei `export_walls()`).
  Beide werden aus `export_tile()` aufgerufen.
- Ausschlusszonen für Vegetation/GroundCover: nur kleine Fußabdrücke um
  Pfeiler-Basen und Portal-Positionen (nicht die gesamte Brücken-/Tunnel-
  Spanne) werden der bestehenden `road_shapes`-Liste hinzugefügt.

## 9. Konfiguration (`config.py`)

Neue Schalter (Default: an) und Konstanten, im Stil der bestehenden
`WALL_*`-Gruppe:

```
BRIDGES_ENABLED = True
BRIDGE_DECK_THICKNESS = ...   # m
BRIDGE_PIER_SPACING = ...     # m
BRIDGE_PIER_SIZE = ...        # m (Querschnitt)

TUNNELS_ENABLED = True        # deckt auch Galerien ab
TUNNEL_WIDTH_MARGIN = ...     # m, zusätzlich zur Fahrbahnbreite
TUNNEL_HEIGHT = ...           # m
TUNNEL_SEGMENT_STEP = ...     # m
TUNNEL_PORTAL_SLOPE_SAMPLE_DIST = ...  # m

GALLERY_COLUMN_SPACING = ...  # m
GALLERY_ROOF_THICKNESS = ...  # m
```

Exakte Zahlenwerte werden während der Implementierung anhand der
Gotthard-Daten kalibriert (z.B. Fahrbahnbreiten aus `osm_to_beamng.json`,
reale Pfeilerabstände von Viadukten).

## 10. Bekannte v1-Einschränkungen

- Keine Terrain-Einschneidung an Portalen über die Schrägschnitt-Anpassung
  hinaus (siehe Nicht-Ziele).
- Keine Geländer-Geometrie an Brücken.
- Keine Stapel-/Kollisionsauflösung für sich kreuzende Brücken.
- Keine Sonderbehandlung für sehr kurze/degenerierte Brücken-/Tunnel-Ways
  (nutzt denselben Mindestlängen-Filter wie normale Straßen,
  `config.DECAL_ROAD_MIN_NODE_SPACING`-Äquivalent).
- Galerie-Talseite wird aus lokalem Geländevergleich bestimmt, nicht aus
  einem OSM-Tag (kein zuverlässiger Tag dafür verfügbar).

## 11. Tests

Neue `tests/bridges/`, `tests/tunnels/` (Struktur wie `tests/walls/`):

- `classify_structure()`: alle vier Fälle inkl. Edge-Cases (`bridge=no`,
  fehlender Tag, `tunnel=avalanche_protector`).
- Lineare Höhenprofil-Interpolation: Endpunkte bleiben exakt erhalten,
  Zwischenpunkte linear nach Bogenlänge.
- Mesh-Geometrie je Builder: Vertex-/Face-Zahlen plausibel, keine NaNs,
  Materialzuordnung korrekt.
- Galerie: Talseiten-Erkennung an einem synthetischen Hang-Profil.
- Tunnel-Portal: Schnittebene folgt der abgetasteten Hangneigung.
- Workflow-Integrationstest (`tests/workflow/`): eine als Brücke/Tunnel
  getaggte synthetische Straße erzeugt kein `DecalRoad`-Item und wird nicht
  ins Terrain-Heightmap eingebettet.
