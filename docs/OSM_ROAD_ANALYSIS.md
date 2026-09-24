# OSM-Straßenanalyse der aktuellen Map (Gotthard/Airolo)

Stand: 2026-09-24 · Datenquelle: `cache/osm_all_4113e78937c1.json` (16 679 OSM-Elemente)

Ziel: Überblick, welche Straßendaten OSM für die aktuelle Map liefert, was davon der DecalRoad-/Brücken-/Tunnel-Export
heute auswertet und welche Spezial-Elemente noch fehlen. Die Analyse-Skripte liegen nicht im Repo; alle Zahlen stammen
aus einer Direktauswertung des Caches plus einem Durchlauf jedes Ways durch `OSMMapper.get_road_properties()`.

---

## 1. Bestand

540 `highway=*`-Ways (davon 3 mit `area=yes`, die `extract_roads_from_osm()` herausfiltert).

| highway          | Anzahl | Länge (m) | in `highway_defaults`?                      |
|------------------|-------:|----------:|---------------------------------------------|
| track            |    133 |    31 848 | ja                                          |
| path             |    115 |    28 726 | ja                                          |
| primary          |    105 |    14 598 | **nein** → Fallback `unclassified`          |
| secondary        |     64 |    14 251 | ja                                          |
| service          |     75 |     8 837 | ja                                          |
| trunk            |      1 |     3 458 | **nein** (Gotthard-Straßentunnel, A2)       |
| construction     |      1 |     2 533 | **nein** (2. Gotthardröhre im Bau, Tunnel!) |
| primary_link     |     23 |     1 699 | **nein** → Fallback `unclassified`          |
| unclassified     |      6 |     1 029 | ja                                          |
| footway          |      6 |       748 | ja                                          |
| residential      |      3 |       643 | ja                                          |
| pedestrian       |      1 |       219 | **nein** (`area=yes`, wird gefiltert)       |
| steps            |      7 |       202 | ja                                          |

Relevante Tags auf den Ways (Auszug):

| Tag                                   | Anzahl | Heute ausgewertet?                           |
|---------------------------------------|-------:|----------------------------------------------|
| `surface`                             |    276 | teilweise (nur 9 Werte in `surface_overrides`) |
| `lanes`                               |    186 | ja (nur `lanes × 3.25 m`)                    |
| `lanes:forward` / `lanes:backward`    |    158 | nein                                         |
| `maxspeed`                            |    183 | nein                                         |
| `layer`                               |     95 | nein                                         |
| `bridge`                              |     69 | ja (Brückenmesh)                             |
| `oneway`                              |     68 | **nein** (17× `yes`, davon 13 Links)         |
| `priority_road`                       |     52 | nein                                         |
| `incline`                             |     43 | nein                                         |
| `motorroad`                           |     41 | **nein** (Schweizer Autostraße A2/A2P)       |
| `tunnel`                              |     30 | ja (28× `yes`, 2× `avalanche_protector`)     |
| `lane_markings`                       |     11 | nein (10× `no`, u.a. Tremola)                |
| `destination*`, `turn:lanes*`         |    ~25 | nein                                         |
| `width`                               |      9 | ja                                           |
| `embankment`                          |      4 | nein                                         |
| `covered`                             |      4 | nein (1× ohne `tunnel`-Tag, s. u.)           |

---

## 2. Auffahrten/Abfahrten (`*_link`) und `motorway_junction`

In der Map gibt es **23 `primary_link`-Ways** (Auf- und Abfahrten der autobahnähnlichen Hauptstraße 2 / A2P,
`motorroad=yes`). Dazu kommen **zwei `highway=motorway_junction`-Knoten**, beide mit dem Namen „Motto Bartola“:

| Knoten     | Lage          | beteiligte Ways                                                                 |
|------------|---------------|---------------------------------------------------------------------------------|
| 24869503   | am Boden      | 422732675 primary (2), 956406313 primary (4), 26245062 / 26245219 primary_link |
| 3688461068 | **auf Brücke** | 129718739 primary (2, Brücke, layer 1), 44220547 primary (4, Brücke, **layer 2**), 1036670761 / 1036670763 primary_link (Brücke, layer 1, oneway) |

Was der aktuelle Code mit Links macht:

- `OSMMapper.get_road_properties()` kürzt `primary_link` → `primary`. Da `primary` (wie `trunk` und `motorway`) aber gar
  nicht in `highway_defaults` steht, landen Links **und** Hauptstraßen beim `unclassified`-Default (5 m). Breiter werden
  sie nur, wenn `lanes` gesetzt ist. Bei 3 Links ohne `lanes` bleibt es bei 5 m, für eine einspurige Rampe zu breit.
- `oneway=yes` wird nirgends gelesen. Das DecalRoad bekommt kein `oneWay`, die KI fährt Rampen also in beide Richtungen.
- Links bekommen dasselbe Material wie die Hauptfahrbahn. Es gibt keine Unterscheidung „Rampe“ (kein Mittelstreifen,
  einseitige Randlinie).

### 2.1 Der Brückenfall „Motto Bartola“ (Knoten 3688461068)

Lokale Geometrie um den Knoten (x/y in m, Knoten = 0/0):

```
44220547  primary  4 Spuren  layer 2   (0,0) → (36,30)      → weiter als 44220548 (4 Spuren, keine Brücke)
129718739 primary  2 Spuren  layer 1   (-9,-11) → (0,0)
1036670761 primary_link 1 Spur, oneway (0,0) → (-5,-2) → … → (-27,-23)   Abfahrt „Motto Bartola“
1036670763 primary_link 1 Spur, oneway (-17,-31) → … → (-1,-4) → (0,0)   Auffahrt
man_made=bridge 1036670764 (layer 1, beam) – Umriss des tatsächlichen Brückenbauwerks, 16 Knoten
```

Das ist ein **Verflechtungs-/Aufweitungsknoten**: Die zweispurige Hauptfahrbahn und die beiden einspurigen Rampen laufen
auf den ersten ~10–20 m fast parallel (Winkel ≈ 15–25°) und vereinigen sich im Knoten zu einem vierspurigen Querschnitt.
Warum das Brückenbauwerk dort scheitert:

1. **Ein Mesh pro Way:** `_build_bridges()` / `build_bridge_mesh()` baut für jeden der 4 Ways ein eigenes Deck mit
   eigenen Bordsteinen und Geländern. Wo Rampe und Hauptfahrbahn parallel laufen, überlappen sich die Decks.
   Die Geländer und Bordsteine der Innenseiten stehen dann **mitten auf der Fahrbahn** des Nachbar-Ways.
2. **Breitensprung im Knoten:** 2 Spuren (6,5 m) enden, 4 Spuren (13 m) beginnen an genau demselben Punkt.
   Die flachen Enden (`cap_style=2`) passen nicht zusammen, dazwischen bleiben Stufen und Lücken.
3. **Layer-Widerspruch:** Der vierspurige Teil hat `layer=2`, die übrigen drei Ways `layer=1`, obwohl sie einen Knoten
   teilen. `layer` wird derzeit nicht ausgewertet. Wer das künftig für die Deckhöhe nutzt, darf es hier nicht wörtlich
   nehmen (OSM-`layer` ist nur relative Ordnung, keine Höhe).
4. **Das echte Bauwerk ist bekannt, wird aber ignoriert:** `man_made=bridge` 1036670764 liefert den Grundriss des Decks,
   das Rampen und Hauptfahrbahn gemeinsam trägt.

Ähnlich, aber harmloser: `101008815` (primary_link, Brücke, 17 m, 2 Spuren) verbindet zwei Rampen. Einzelnes Deck ohne
Überlappung, sollte also funktionieren. Das bleibt aber zu prüfen.

---

## 3. Weitere Spezial-Elemente, die wir noch nicht behandeln

### 3.1 Straßentypen / Filter

| Element                                    | Problem heute                                                                 |
|--------------------------------------------|-------------------------------------------------------------------------------|
| `highway=construction` (798912292, `construction=trunk`, `tunnel=yes`, 2,5 km) | wird als normaler Tunnel mit 6,5 m gebaut, obwohl die Röhre noch im Bau ist. Muss gefiltert werden (ebenso `proposed`, `abandoned`, `razed`). |
| `highway=trunk` (Gotthard-Straßentunnel, 3,5 km) | kein Eintrag in `highway_defaults`. Die Breite stimmt nur dank `lanes=2`.       |
| `highway=primary` (105 Ways)               | kein Eintrag in `highway_defaults` (siehe oben).                                |
| `motorway`, `motorway_link`, `trunk_link`, `secondary_link`, `tertiary_link` | kommen hier nicht vor, fehlen aber ebenfalls. Für Karten mit echter Autobahn nötig. |
| `covered=yes` ohne `tunnel` (746194686)    | **Umgesetzt 2026-09-24:** mit negativem `layer` als Galerie gebaut (offen zur per Mehrheit ermittelten Talseite); Übergänge Tunnel ↔ Galerie mit Portalwand, siehe `docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md`. |

### 3.2 Oberflächen, die in `surface_overrides` fehlen

| surface   | Ways | betrifft                                    | fällt heute auf         |
|-----------|-----:|---------------------------------------------|-------------------------|
| sett      |   28 | **Tremola** (historische Pflasterstraße), Via San Gottardo | Asphalt           |
| concrete  |   31 | Hauptstraße 2, Rampen, Service              | Asphalt (vertretbar)    |
| rock      |    6 | Wanderwege                                  | Erdweg                  |
| grass     |    4 | Wege                                        | Erdweg                  |
| unpaved   |    3 | track/path/service                          | uneinheitlich (Service → Asphalt!) |

Für die Tremola lohnt sich ein eigenes Pflaster-Material (`sett`/`cobblestone`), das ist *das* Wahrzeichen der Map.
Das `unpaved`-Service-Beispiel zeigt eine Lücke im Mapper: Nicht gemappte Oberflächen behalten den
Highway-Default, statt „unbefestigt“ zu erkennen.

### 3.3 Breiten-Inkonsistenzen

- `secondary` ohne `lanes` → 7,0 m, mit `lanes=2` → 6,5 m.
- `lanes × 3.25` gilt für alle Typen gleich, auch für die Tremola (schmale Pflasterstraße ohne Markierung) und für
  Service-Straßen mit `lanes=2`.
- Keine Randstreifen bei Schnellstraßen (`motorroad=yes`, `shoulder`).
- Kein Übergang (Taper) bei Spurzahlwechsel. 2 → 3 → 4 Spuren springen hart um. Betrifft 13 `lanes=3`- und
  5 `lanes=4`-Ways.

### 3.4 Knoten-Tags auf Straßenknoten (bisher komplett ignoriert)

| Knoten-Tag                    | Anzahl | Idee                                                  |
|-------------------------------|-------:|-------------------------------------------------------|
| `ford=yes`                    |     15 | Furt: Straße nicht über Bach-River heben, Wasser-Decal |
| `highway=give_way` / `stop`   |  10/1  | Verkehrszeichen-Props, KI-Vorfahrt                     |
| `highway=passing_place`       |      4 | lokale Verbreiterung (Ausweichstelle)                  |
| `highway=emergency_bay`       |      2 | Nothaltebucht an der Hauptstraße                       |
| `highway=motorway_junction`   |      2 | Ausfahrtsschild, Markierung für Verflechtungsbereich   |
| `barrier=gate/lift_gate/swing_gate/block/bollard` | 14 | Schranken/Poller als Props           |
| `highway=milestone`           |      3 | Deko                                                  |
| `highway=crossing`            |      1 | Zebrastreifen-Decal                                    |

### 3.5 Weitere OSM-Objekte mit Straßenbezug

- **`man_made=bridge`** (21 Flächen): echte Grundrisse der Brücken, siehe 2.1.
- **`area:highway=*`** (22 Flächen, 1× `motorway`): exakte Fahrbahnflächen, gut für Knoten und Aufweitungen.
- **Abbiegeverbote** (`type=restriction`, 22 Relationen): `no_left_turn` 9, `only_straight_on` 7, `no_u_turn` 5,
  `only_right_turn` 1. Relevant für die KI-Navigation.
- **`type=tunnel`-Relationen** (Gotthard-Straßen- und Bahntunnel): fassen mehrere Tunnel-Ways zu einem Bauwerk zusammen.
- **`man_made=avalanche_protection`** (96 Ways): Lawinenverbauungen, keine Straßen, aber optisch prägend.

---

## 4. Empfohlene Reihenfolge

1. **Mapper-Grundlagen** (klein, sofort wirksam): `motorway`/`trunk`/`primary` + alle `*_link` in `highway_defaults`
   (Links einspurig ~4 m), fehlende Oberflächen (`sett`, `concrete`, `rock`, `grass`, `unpaved`),
   `construction`/`proposed` filtern.
   **Umgesetzt 2026-09-24:**
   - `highway_defaults`: motorway 8,0 / trunk 7,5 / primary 7,5 m, `*_link` 4,0 m (tertiary_link 3,75 m). Der Mapper
     sucht jetzt zuerst den exakten Typ, erst dann den Basistyp.
   - Neue Oberfläche `cobblestone_road`: Groundmodel `COBBLESTONE`, Texturen aus BeamNGs `tileable/stone/italy_cobblestone`,
     Priorität 6. Sie gilt für `sett`, `cobblestone`, `unhewn_cobblestone` und `paving_stones`.
   - `unpaved` → Kies, `grass`/`rock` → Erdweg.
   - `concrete` bleibt bewusst beim Highway-Default. Das Beton-Material ist für Fußwege gedacht (`drivability` 0).
   - `extract_roads_from_osm()` verwirft `highway=construction/proposed/planned/abandoned/disused/razed/demolished`.
   - Im Spiel noch zu prüfen: Maßstab und Kachelung der Pflastertextur auf der Tremola (DecalRoad streckt die Textur
     über die ganze Breite). Ohne `opacityMap` hat der Rand außerdem eine harte Kante.
2. **Fahrtrichtung im DecalRoad:** `oneway` → `oneWay`/Spurzahlen (`lanesLeft`/`lanesRight` aus
   `lanes:forward`/`lanes:backward`), damit die KI Rampen richtig befährt. Die Feldnamen vorher gegen ein
   BeamNG-Referenzlevel (`items.level.json`) prüfen.
3. **Brücken an Verzweigungen** (Motto Bartola): Brücken-Ways, die sich einen Knoten teilen, zu einem Bauwerk gruppieren.
   Das Deck als Vereinigung der Fahrbahnpolygone (oder direkt aus `man_made=bridge`) bauen, Geländer nur an der
   Außenkontur, Pfeiler gemeinsam.
4. **Spurzahlwechsel mit Übergang** (Taper über ~30–50 m statt hartem Sprung), gilt für DecalRoad und Brückendeck.
5. **Knoten-Features** (Furten, Ausweichstellen, Schranken, Schilder) als eigene Stufe.
