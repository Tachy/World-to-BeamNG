# Übergang Tunnel ↔ Galerie – Design

Stand: 2026-09-24 · Ergänzt `2026-09-22-bridges-tunnels-design.md` (Abschnitte 1, 5, 6)

## Ziel

Geht eine Tunnelröhre direkt in eine Galerie über, soll ein durchgehendes Bauwerk entstehen: Die Röhre endet an einer
Betonstirnwand, an die Galeriedach, Bergwand und Sockel bündig anschließen. Die Durchfahrt bleibt frei, es gibt keine
Erdwand in der Röhre und keine flackernden Flächen. Vorbild ist die Nordseite von Tunnel Banchi (Foto des Nutzers):
Galerie mit Stützen talseits, Bergwand, am Ende die flache Portalwand mit der Tunnelöffnung.

## Befund (Karte Gotthard, OSM-Stand 2026-09-24)

Kette der Nuova strada del Passo del San Gottardo von Norden:

| Way | Tags (nach Korrektur durch den Nutzer) | Länge | Name |
|---|---|---|---|
| 430132545 | `covered=yes` (vorher zusätzlich `tunnel=yes`) | 49 m | Galleria artificiale Banchi |
| 430132546 | `tunnel=yes` | 44 m | Tunnel Banchi |
| 746194686 | `covered=yes` | 727 m | lange Galerie |
| 430132547 | `tunnel=yes` | 796 m | Tunnel Fieud |

Keiner der Ways trägt `avalanche_protector:left/right`. Der Nutzer ergänzt voraussichtlich
`avalanche_protector:right=open` an beiden Galerien. Nach Foto und Kartenlage liegt die Talseite in
Digitalisierungsrichtung vermutlich bei beiden rechts; der Nutzer bestätigt das beim Taggen.

Heutiges Verhalten und Fehler:

- `classify_structure()` kennt Galerien nur über `tunnel=avalanche_protector`. Way 746194686 wird deshalb als
  Oberflächenstraße (DecalRoad) gebaut.
- Das Tunnelende wird als offenes Portal ins Gelände behandelt. Am Nordportal von Tunnel Fieud liegt das Gelände
  laut Höhenmodell die ersten ~9 m hinter der Portalebene auf Fahrbahnhöhe; die Röhre steckt erst ab dort im Berg.
  `_raise_cover()` überdeckt nur Abschnitte, in denen die Röhre schon im Gelände steckt. Zwischen der
  Portal-Überdeckung (+7,9 m über dem Röhrenboden, 3–4,5 m hinter der Portalebene) und diesem flachen Stück fällt das
  Gelände auf 1 m um ~8 m ab. Die Geländefläche schneidet dabei quer durch die Röhre (sichtbare Erdwand). Bei 9–10 m
  entsteht eine zweite.

## 1. Klassifizierung (`geometry/road_structures.py::classify_structure`)

Neue Reihenfolge:

1. `bridge=*` (außer `no`) → `"bridge"`
2. `tunnel=avalanche_protector` → `"gallery"`
3. `covered=yes` und `tunnel` fehlt oder ist `no` → `"gallery"` **(neu)**
4. jedes andere `tunnel=*` (außer `no`) → `"tunnel"`
5. sonst `"surface"`

`tunnel=yes` + `covered=yes` bleibt ein Tunnel. Die offene Talseite bestimmt `gallery_mesh.resolve_open_side()` aus
`avalanche_protector:left/right=open`. Ohne Tag (der Nutzer setzt ihn nicht, 2026-09-24) gilt **eine** Seite für
die ganze Galerie: die Mehrheit der punktweisen `valley_side()` (Geländevergleich). Die Galerie ist damit komplett
nach dieser Seite offen.

Klassifizierung präzisiert bei der Umsetzung: `covered=yes` wird nur mit **negativem `layer`** zur Galerie. Beide
Galerien der Nuova strada haben `layer=-1`; ein Vordach über einer Service-Straße (ohne oder mit `layer ≥ 0`) bleibt
Oberfläche.

## 2. Übergänge erkennen (`tunnels/tunnel_portal.py::plan_tunnels`)

- Neuer Parameter `galleries`: Liste der Galerie-Centerlines (x, y, z), also dieselben Eingaben wie für
  `build_galleries()`.
- Ein Portal bekommt `kind="gallery"`, wenn sein Punkt höchstens `TUNNEL_TRANSITION_ENDPOINT_TOL` (0,5 m) von einem
  **Endpunkt** einer Galerie-Centerline entfernt liegt. Tunnel und Galerie teilen sich in OSM den Endknoten; das
  Smoothing hält Endpunkte exakt fest (`smooth_roads_xy_only`), die Koordinaten fallen also zusammen. Sonst gilt
  `kind="open"` wie bisher.
- Ein Übergangs-Portal übernimmt zusätzlich die Maße der anschließenden Galerie: `gallery_half_width` (halbe
  Fahrbahnbreite), `gallery_height` (`GALLERY_HEIGHT`), `gallery_outer_half_width` (halbe Fahrbahnbreite +
  `GALLERY_WALL_THICKNESS`), `gallery_floor_bottom` (Boden − `GALLERY_FLOOR_THICKNESS`), `gallery_roof_top`
  (Boden + `GALLERY_HEIGHT` + `GALLERY_ROOF_THICKNESS`).
- Der Portalblock wird so erweitert, dass er den ganzen Galerie-Querschnitt abdeckt:
  - `half_width = max(radius + wing, gallery_outer_half_width)`
  - `top_z = max(top_z, gallery_roof_top + 0.2)`
  - `bottom_z = min(bottom_z, gallery_floor_bottom)`

  Das Galerie-Profil passt bei allen Straßenbreiten in den Röhrenbogen. Bei 6,5 m Fahrbahn: Bogen in 5 m Höhe
  ±3,75 m, Galerie ±3,25 m.

## 3. Portal-Mesh (`tunnels/tunnel_portal.py::build_portal_block_mesh`)

Für `kind="open"` bleibt alles unverändert. Für `kind="gallery"`:

- **Stirnseite (Portalebene, Normale Richtung Galerie):** Rechteck von `bottom_z` bis `top_z` über die volle
  `half_width`, mit einer rechteckigen Öffnung von `−gallery_half_width … +gallery_half_width` × `0 … gallery_height`
  (relativ zum Röhrenboden), statt der Bogenöffnung.
- **Rückseite zum Tunnel (gleiche Ebene, Normale ins Tunnelinnere):** die Fläche zwischen Röhrenbogen und
  Rechteck-Öffnung, also Bogen-Innenfläche minus Rechteck. Sie ist der sichtbare Querschnittssprung von innen. Der
  erste Röhrenring trifft den Bogen wie bisher exakt.
- **Seitenwände, Oberseite, Rückwand oberhalb der Röhre:** wie beim offenen Portal, mit der erweiterten Breite und
  Höhe aus Abschnitt 2.

## 4. Galerie-Enden am Übergang (`tunnels/gallery_mesh.py`)

- `build_gallery_mesh()` bekommt `cap_start`/`cap_end` (Default `True`). An einem Ende, das an ein Übergangs-Portal
  stößt, entfällt die Stirnfläche (`_add_end_caps`). Sie läge in derselben Ebene wie die Portalwand (Z-Fighting),
  und die Portalwand deckt den ganzen Galerie-Querschnitt ab (Abschnitt 2).
- `build_galleries()` bestimmt das aus denselben Übergängen: Galerie-Endpunkt ≤ `TUNNEL_TRANSITION_ENDPOINT_TOL` von einem Portal mit
  `kind="gallery"`.

## 5. Gelände (`terrain/tunnel_terrain.py`)

- **Übergangs-Portale sind immer Portale.** `_portal_is_open()` gilt nur für `kind="open"`; `kind="gallery"` ist
  immer `open=True`, denn davor liegt ein Bauwerk.
- **Portal-Zone:** Wie beim offenen Portal: Gelände bis `TUNNEL_PORTAL_FLAT_DEPTH` hinter der Portalebene knapp unter
  Bodenhöhe, dahinter Überdeckung, Loch-Zellen am Übergang im Block. Das Vorfeld (Galerie-Seite) bestimmt die
  Galerie-Einbettung; Galerien sind schon heute in `protected` enthalten.
- **Überdeckung ohne Lücke (alle Portale, auch offene):** `_raise_cover()` überdeckt zusätzlich jede zusammenhängende
  Strecke nicht-eingegrabener Centerline-Punkte, die an einem offenen Portal beginnt und höchstens
  `TUNNEL_COVER_GAP_MAX` (25 m) lang ist, bevor die Röhre im Gelände steckt. Damit läuft die Überdeckung vom
  Portalblock bis in den Berg ohne Absatz. Längere offene Strecken (echte Hanglage, Tagbau im Höhenmodell) bleiben
  unüberdeckt, sonst entstünden Dämme.

## 6. Export

- `TerrainWorkflow`: `plan_tunnels()` bekommt die Galerie-Eingaben (`_gallery_road`/`_structure_items(…, "gallery")`),
  `_build_tunnels()` reicht die Übergänge an `build_galleries()` weiter.
- Way 746194686 und 430132545 werden damit keine DecalRoads und keine Markierungen mehr. Beides gibt es für
  Strukturen ohnehin nicht (nur `structure_type == "surface"`).

## Konfiguration (`config.py`)

```python
TUNNEL_TRANSITION_ENDPOINT_TOL = 0.5  # Tunnel-Portal und Galerie-Ende gelten als Übergang, in Metern
TUNNEL_COVER_GAP_MAX = 25.0  # so lange Lücke zwischen Portal und eingegrabener Röhre wird noch überdeckt, in Metern
```

## Tests

- **Klassifizierung:**
  - `covered=yes` → `gallery`
  - `covered=yes` + `tunnel=no` → `gallery`
  - `covered=yes` + `tunnel=yes` → `tunnel`
  - `tunnel=avalanche_protector` → `gallery`
  - `tunnel=yes` → `tunnel`
- **`plan_tunnels`:**
  - Tunnelende auf einem Galerie-Endpunkt → `kind="gallery"` mit Galerie-Maßen und erweitertem Block
  - Tunnelende an einer normalen Straße → `kind="open"`
  - Galerie, die nur seitlich den Tunnel berührt (kein Endpunkt) → kein Übergang
- **Portal-Mesh:**
  - Stirnseite hat eine rechteckige Öffnung in Galeriegröße: kein Dreieck der Stirnseite überdeckt das
    Öffnungsrechteck, und die Stirnseite deckt den restlichen Blockquerschnitt vollständig ab (Flächensumme).
  - Die Rückseite zeigt ins Tunnelinnere und hat die Fläche (Bogenfläche − Rechteck).
- **Galerie:** Ohne Stirnfläche am Übergangsende gibt es dort keine Flächen in der Portalebene; das andere Ende
  bleibt verschlossen.
- **Gelände:**
  - Synthetischer Hang mit 8 m flachem Stück hinter dem Portal: Überdeckung durchgehend, keine Stufe innerhalb des
    Röhrenquerschnitts.
  - 40 m flaches Stück: keine Überdeckung dort.
  - Übergangs-Portal ohne offenes Vorfeld bekommt trotzdem Portal-Zone und Löcher.
- **Echter Export:**
  - Gelände-Profil entlang der Achse am Nordportal von Tunnel Fieud: zwischen Portal-Überdeckung und Berg kein
    Abfall unter die Kronenhöhe.
  - Drei Übergangs-Portale mit `kind="gallery"`: Tunnel Banchi Nord und Süd, Tunnel Fieud Nord.
  - Keine DecalRoads für 746194686 und 430132545.
- **Im Spiel:** Durchfahrt Nord → Süd durch Galerie – Tunnel Banchi – Galerie – Tunnel Fieud.

## Nicht im Umfang

- Fließender Querschnittsübergang (Kreis → Rechteck über mehrere Meter).
- Galerie ↔ Galerie, Galerie ↔ Brücke.
- Tagbau-Tunnel mit rechteckigem, geschlossenem Querschnitt.
- Fensteröffnungen in Galerie-Bergwänden (die „lange Galerie mit Fenstern“ im Hintergrund des Fotos).

## Vor der Umsetzung

- Den OSM-Cache neu laden, sobald Overpass die Korrektur von Way 430132545 (Version 9, `tunnel=yes` entfernt) und
  ggf. die `avalanche_protector:right=open`-Tags liefert: `cache/osm_all_4113e78937c1.json` löschen und neu
  exportieren.
