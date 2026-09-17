# Terrain-Migration: Custom-Mesh → natives `.terrain` + `DecalRoad`/Mesh-Straßen

**Status:** Design genehmigt, bereit für Implementierungsplan
**Datum:** 2026-09-17

## 1. Ziel & Motivation

`world_to_beamng` baut aktuell das komplette Terrain (inklusive Straßen) als ein
einziges, custom-generiertes DAE-Mesh: Terrain-Grid wird trianguliert, Straßen
werden analytisch als eigenes Mesh erzeugt und anschließend topologisch ins
Terrain-Mesh **verschweißt** (`mesh/stitch_local.py`: KDTree-Suche,
Connected-Components via DFS, Union-Find-Merging, Ear-Clipping-Triangulation).

Diese Verschweißung ist die Quelle der größten bestehenden Instabilität im
Projekt — sichtbar u.a. daran, dass `GENERATE_SLOPES` in `config.py` aktuell
deaktiviert ist ("vorübergehend deaktiviert bis Remeshing stabil"). BeamNG
selbst arbeitet für reale/Geodaten-basierte Level nativ mit einem
Heightmap-Terrain (`.terrain`) statt mit einem Custom-Mesh — das bestätigen
auch existierende Community-Tools mit vergleichbarem Ziel (MapNG,
BeamNG.WorldForge).

Ziel dieser Migration: das Terrain nativ als `.terrain`-Heightmap exportieren
und die Straße-Terrain-Verschweißung ersatzlos eliminieren — bei **voller
Beibehaltung** der bestehenden Straßen-Geometrie-Qualität und des gesamten
Nutzer-Workflows.

## 2. Harte Anforderungen (vom Nutzer vorgegeben)

1. **Workflow unverändert:** gleicher CLI-Aufruf (`python world_to_beamng.py`),
   gleiche Config-Struktur, gleiche Eingabedaten (DGM1/DGM30/DOP20/OSM-Cache).
2. **Auflösung/Feinheit bleibt mindestens gleich** wie heute (`GRID_SPACING`).
3. **Straßen-Querschnitt (90° zum Straßenverlauf) bleibt immer horizontal/eben**
   — auch wenn das umgebende Gelände seitlich abfällt.
4. **Saubere Einbettung**, wenn die Straße tiefer liegt als das umgebende
   Terrain (kein Schweben, kein Durchstechen, kein sichtbarer Spalt).
5. Rechteckige Level-Flächen sind akzeptabel (keine Anforderung an frei
   geformte, nicht-rechteckige Level-Umrisse).

## 3. Betrachtete Ansätze

- **A — Vollständig nativ (`DecalRoad` + Heightmap-Sculpting):** Straßen werden
  reine Decals, Böschungen entstehen durch Höhen-Sculpting im Raster (Margin/
  Domain-of-Influence, wie der offizielle, aber nicht skriptbare "Terrain And
  Road Importer" es tut). Größte Code-Reduktion, aber `DecalRoad` kann selbst
  keine über das Terrain hinausragende Böschung darstellen — verwirft die
  bestehende, fein abgestimmte Böschungslogik zugunsten eines gröberen,
  rasterbasierten Verfahrens. **Verworfen** — zu großer Fidelity-Verlust
  gegenüber Anforderung 3/4.
- **B — Straßen bleiben Custom-Mesh, nur das Hintergrund-Terrain wird nativ
  (GEWÄHLT):** Straßen-Generierung (Querschnitt, Böschung, Junction-Remeshing)
  bleibt exakt wie heute. Terrain wird zum nativen Heightmap; keine
  topologische Verschweißung mehr nötig — die Straße liegt einfach *auf* dem
  Terrain, das im Straßenkorridor knapp genug abgesenkt wird.
- **C — Hybrid nach Straßentyp:** Hauptstraßen als Mesh (wie B), kleine Wege
  als reine `DecalRoad`. Spart Rechenzeit bei vielen kleinen Wegen, aber zwei
  parallele Codepfade für wenig Nutzen bei der aktuellen Straßenmenge.
  **Verworfen** — unnötige Komplexität für den jetzigen Bedarf.

## 4. Kern-Mechanismus: Straßen-Einbettung (beantwortet Anforderung 3 + 4)

Die Straßen-/Böschungsgeometrie wird **unverändert** wie heute berechnet
(exaktes, analytisches Mesh, horizontal im Querschnitt, mit vorhandener
Junction-Remeshing-Logik zwischen Straßensegmenten).

Neu ist ausschließlich, wie sie mit dem Terrain zusammenspielt:

Für jede Zelle des Terrain-Höhenrasters, die innerhalb eines Korridors um eine
Straße liegt, wird die Höhe der (bereits vorhandenen) Straßen-/Böschungsfläche
an exakt dieser XY-Position abgefragt, ein konfigurierbarer
Sicherheitsabstand (`ROAD_EMBED_MARGIN`) abgezogen, und das Minimum aus
diesem Wert und der natürlichen Terrainhöhe ins Raster geschrieben. Außerhalb
des Korridors bleibt das Terrain unverändert. Liegen mehrere Straßen nah
beieinander (Kreuzungen), wird einfach das Minimum aller beteiligten
Absenkungen genommen — keine Sonderbehandlung für Kreuzungen nötig.

Weil die Böschung selbst (die den Höhenunterschied zur Umgebung überbrückt)
bereits Teil der unveränderten Straßen-Mesh-Geometrie ist und dort endet, wo
sie natürlich ins Gelände übergeht, muss das Terrain-Raster die
Böschungsform nicht selbst nachbilden — es muss nur überall dort, wo diese
Geometrie existiert, knapp darunter bleiben. Straßenoberfläche und
Terrain-Kante werden dadurch an derselben Referenz ausgerichtet statt an zwei
unabhängig berechneten Näherungen, was Anforderung 3 (immer horizontal, weil
unveränderter Code) und Anforderung 4 (saubere Einbettung, weil beide Seiten
konsistent sind) direkt erfüllt.

## 5. Architektur: Komponenten & Datenfluss

### Neue Module
- `terrain/heightmap.py` — baut das 2D-Höhenraster aus der bestehenden
  Elevation-Grid-Logik (nur Werte ins Array schreiben statt zu triangulieren).
- `terrain/road_embedding.py` — Mechanismus aus Abschnitt 4.
- `terrain/ter_writer.py` — schreibt `.ter` (Heightmap + Layer-Map +
  Material-Liste) gemäß dokumentiertem Binärformat.
- `terrain/terrain_materials.py` — baut `TerrainMaterialTextureSet` +
  Layer-Map (siehe Abschnitt 6).

### Entfällt ersatzlos
- `mesh/stitch_local.py` komplett.
- Terrain-Grid-Triangulierung in `mesh/mesh.py` (Straßen-Triangulierung
  bleibt bestehen).
- Hole-Filling-/CCW-Normalisierungs-Schritte, die speziell für die
  Straße-Terrain-Verschweißung existieren.
- Config-Flags, die nur für den alten Mechanismus existierten, werden
  entfernt: `FILL_ALL_MESH_HOLES`, `FILL_HOLES_MAX_EDGE_LENGTH` (beide nur in
  `mesh/stitch_gaps.py` verwendet, das komplett entfällt).
  `HORIZON_BOUNDARY_STITCHING` bleibt als Flag bestehen (weiterhin `False`,
  siehe Abschnitt 7) — seine Zielfunktion `stitch_terrain_horizon_boundary`
  kann in der neuen Architektur ohnehin keine Mesh-Mesh-Naht mehr bilden, da
  das Haupt-Terrain kein Mesh mehr ist; das ist kein Regressionsrisiko, weil
  der Flag heute schon deaktiviert ist. **Korrektur:** `GENERATE_SLOPES`
  entfällt NICHT — es sitzt in `mesh/road_mesh.py` (nicht im
  Stitching-Code) und steuert dort, ob Böschungs-Geometrie überhaupt erzeugt
  wird. Es ist aktuell `False`, Böschungen werden also **heute gar nicht
  generiert**. Für Anforderung 4 muss dieser Flag im Zuge der Migration auf
  `True` gesetzt werden — vermutlich stand er wegen der (jetzt entfallenden)
  Stitching-Instabilität auf `False`, nicht wegen eines Fehlers in der
  Böschungs-Geometrie selbst. Das ist jetzt Teil des Implementierungsplans.
- Terrain-Anteil der DAE-Tile-Aufteilung (`TILE_SIZE`) — ein Heightmap pro
  Level statt N Terrain-Tiles. Ob Straßen-DAEs weiterhin in 500m-Stücken
  exportiert werden (rein Streaming-Optimierung, nicht mehr technisch
  erzwungen), bleibt vorerst wie heute und ist eine spätere, unabhängige
  Optimierungsentscheidung.

### Unverändert
OSM-Extraktion, Höhendaten-Laden, Straßen-Centerline/Querschnitt/
Junction/Böschungs-Code, Gebäude-Workflow, Forest-Workflow,
Horizont-Workflow, `MaterialManager`/`ItemManager` als Klassen (werden
erweitert, nicht ersetzt), `CacheManager`.

### Ablauf pro Export-Lauf
1. Elevation-Grid laden (unverändert).
2. OSM-Straßen/Junctions extrahieren (unverändert).
3. Straßen-/Böschungsmesh generieren (unverändert), bleibt zunächst im
   Speicher — **keine** Terrain-Verschmelzung mehr.
4. Heightmap-Array direkt aus dem Elevation-Grid befüllen.
5. Straßen-Einbettung (Abschnitt 4) auf das Heightmap-Array anwenden.
6. Layer-Map aus Luftbild-Material (pro Tile-Ausschnitt) + OSM-Landnutzung
   (Abschnitt 6) befüllen.
7. `.ter`-Datei schreiben.
8. Straßenmesh wie bisher als DAE exportieren (unverändert, ohne
   Terrain-Verschmelzung — tendenziell sogar weniger Flächen, da keine
   Lücken-Dreiecke mehr nötig sind).
9. `TerrainBlock` + Straßen-Objekte in `items.level.json` registrieren
   (gleicher `ItemManager`-Mechanismus wie beim bestehenden `Forest`-Objekt).

## 6. Materialien & Textur

### Basis: Luftbild-Drapierung (Parität zu heute)
Statt eines einzigen riesigen zusammengesetzten Fotos (Risiko: Textur würde
bei 3-4 km Levelgröße weit über praktikable GPU-Texturgrößen hinauswachsen)
werden die Materialgrenzen 1:1 von den heutigen 500m-DAE-Tiles übernommen:
pro Tile ein Material mit dem heutigen Luftbild-Ausschnitt als
`baseColorMap`. Nur der Verbrauchsort ändert sich — von Mesh-Face-UV zu
Layer-Map-Zelle. Der bestehende `MaterialManager` liefert die Materialdaten
im Kern unverändert.

### Neu: OSM-Landnutzung → Terrain-Material
Bisher nicht umgesetztes Ziel des Nutzers, das durch die Layer-Map-Struktur
jetzt natürlich erreichbar wird: OSM-Landnutzungs-Polygone werden direkt auf
passende BeamNG-Terrain-Materialien gemappt.

**Mechanismus:** Für jedes Landnutzungs-Polygon wird per
`rasterio.features.rasterize()` (bereits vorhandene Dependency) sein
Material-Index in die Layer-Map gebrannt. Eine Mapping-Tabelle
(Erweiterung von `data/osm_to_beamng.json`, gleiches Muster wie das
bestehende `forest_mappings`) übersetzt OSM-Tags in Material-Namen.
Überlappende Polygone werden nach Flächengröße aufgelöst (kleinere zuletzt
gebrannt, damit sie nicht von größeren überschrieben werden). Zellen ohne
Treffer behalten das Luftbild-Fallback-Material dieser Tile — es bleibt
garantiert nichts unbedeckt.

**Materialquelle:** BeamNGs eigener Content (`content/assets/materials/
terrain.zip`), nach demselben Vendoring-Prinzip wie bereits für Bäume und
Straßen-Decals angewendet (siehe `tools/vendor_shared_textures.py` /
`tools/generate_forest_assets.py`).

**Startsatz Mapping** (bewusst konservativ, später erweiterbar):

| OSM-Tag | Material |
|---|---|
| `landuse=forest`, `natural=wood` | `forest_floor` |
| `landuse=meadow`, `natural=grassland`, `landuse=grass`, `leisure=park` | `grass` |
| `landuse=farmland`, `landuse=farmyard` | `farmland` |
| `natural=sand`, `natural=beach` | `sand` |
| `natural=bare_rock`, `natural=scree` | `rock` |

**Bewusst außerhalb des Startsatzes** (bleibt Luftbild-Fallback):
- Wasser (`natural=water`, `landuse=reservoir`) — Wasser ist in BeamNG ein
  eigenes Objekt (Water-Plane), kein Terrain-Material; eigenes Thema für
  später.
- Siedlung/Gewerbe (`landuse=residential`, `industrial`, `commercial`) — zu
  heterogen für ein pauschales Material, Luftbild ist hier vermutlich die
  bessere Darstellung.

**Zusammenspiel mit dem Forest-System:** Für `landuse=forest` bekommt der
Boden das `forest_floor`-Material *und* das bestehende Forest-System
bestückt dieselbe Fläche weiterhin mit Bäumen (`forest_type_templates`
referenziert `underground_material: forest_floor` bereits) — zwei
unabhängige Systeme auf denselben Quellpolygonen, kein Konflikt.

## 7. Scope-Grenzen

- **Gebäude (LoD2):** unberührt — werden bereits heute mit absoluten
  Z-Koordinaten aus den LoD2-Rohdaten platziert, unabhängig vom
  Terrain-Rendering-System.
- **Horizont-Layer:** bleibt bewusst als eigenständiges Fern-Mesh bestehen.
  Ein einzelnes `.terrain`-Quadrat, das sowohl das detaillierte Hauptgebiet
  als auch den kilometerweiten Horizont abdeckt, würde entweder das
  8192px-Formatlimit sprengen oder die Hauptfläche unnötig grob machen; der
  Horizont braucht zudem keine Kollision/Befahrbarkeit.
- **Brücken/Tunnel:** werden aktuell nicht gehandhabt, kein
  Regressionsrisiko. `.terrain` hätte über die Hole-Map später eine saubere
  Erweiterungsoption, falls relevant — nicht Teil dieser Migration.

## 8. Fehlerbehandlung & Edge-Cases

- **Überlappende Straßen-Einbettungskorridore** (Kreuzungen): Minimum aller
  beteiligten Absenkungen, keine Sonderbehandlung.
- **`.ter` erfordert Zweierpotenz-Quadrat**, die reale Fläche ist es nicht:
  Heightmap wird auf die nächstgrößere Zweierpotenz aufgerundet; der
  Überschussrand wird mit den Randwerten der echten Daten fortgeführt
  (Extrapolation), nicht als Hole markiert — vermeidet eine sichtbare Kante
  am Datenrand.
- **`maxHeight`:** wird aus dem tatsächlichen Min/Max der geladenen
  Höhendaten + Puffer berechnet, nicht hartkodiert; Warnung im Log, falls
  künftig Daten außerhalb dieses Bereichs auftauchen.
- **`ROAD_EMBED_MARGIN`:** neuer, konfigurierbarer Wert (Konvention wie
  bestehende `config.py`-Parameter); zu klein riskiert Z-Fighting, zu groß
  einen sichtbaren Spalt.
- **Überlappende OSM-Landnutzungs-Polygone:** deterministisch nach
  Flächengröße aufgelöst (kleinere zuletzt gebrannt).
- **Obsolete Config-Flags** werden im Code explizit als entfernt markiert
  (Commit-Historie dokumentiert das), nicht stillschweigend verwaist.
- **Migration:** kein Live-Umzug bestehender exportierter Level — der
  nächste volle Lauf erzeugt automatisch das neue Format. Alter
  Mesh-Terrain-Code wird **vollständig entfernt**, kein Parallelbetrieb,
  kein Fallback-Flag (explizite Nutzerentscheidung).

## 9. Testing

- **Unit-Tests pro neuem Modul (TDD):**
  - `ter_writer`: Write→Read-Rundlauf, Byte-Struktur gegen dokumentierte
    Spec geprüft.
  - `road_embedding`: synthetisches Straßenmesh + synthetisches Grid — nahe
    Zellen korrekt abgesenkt, ferne Zellen unverändert.
  - `terrain_materials`: Rasterize-Ergebnis gegen bekannte Test-Polygone.
- **Struktur-Validierung ohne BeamNG:** Script, das die erzeugte `.ter`
  zurückparst und gegen die Format-Spec prüft (Zweierpotenz, u16-Länge,
  Layer-Anzahl) — findet Formatfehler in Sekunden statt nach jedem Test
  einen BeamNG-Ladevorgang abzuwarten.
- **Regression:** neues Ausgabeformat, also keine alten Golden-Files. Nach
  dem ersten erfolgreichen neuen Lauf wird ein neuer Baseline-Snapshot
  (Level-Größe, Straßen-Anzahl, Rasterwerte-Stichproben) angelegt.
- **Visuelle Verifikation:** Pflicht-Schritt vor Fertigstellung — Level in
  BeamNG laden, Straßen-Einbettung an mindestens einer bekannten
  Hangsituation aus dem Testgebiet visuell prüfen.

## 10. Explizit außerhalb dieses Scopes (für später)

- Wasser-Flächen als eigenes BeamNG-Objekt statt Terrain-Material.
- Siedlungs-/Gewerbeflächen-Materialien.
- Brücken/Tunnel über die `.terrain`-Hole-Map.
- Vereinfachung zu einer einzigen großen Luftbild-Komposit-Textur (statt
  Tile-Materialien) — nur falls die praktischen Texturgrößen-Limits von
  BeamNG das später zulassen.
- `squareSize` feiner als `GRID_SPACING` wählen (Nutzer möchte zunächst
  Parität, spätere Feinabstimmung möglich).
- Straßen-DAE-Tiling-Strategie überdenken, jetzt da sie nicht mehr an die
  Terrain-Kachelung gebunden ist.
