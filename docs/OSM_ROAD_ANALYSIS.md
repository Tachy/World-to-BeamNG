# OSM Road Analysis of the Current Map (Gotthard/Airolo)

As of: 2026-09-24 · Data source: `cache/osm_all_4113e78937c1.json` (16,679 OSM elements)

Goal: overview of which road data OSM provides for the current map, which of it the DecalRoad/bridge/tunnel export
evaluates today, and which special elements are still missing. The analysis scripts are not in the repo; all figures come
from a direct evaluation of the cache plus a pass of every way through `OSMMapper.get_road_properties()`.

---

## 1. Inventory

540 `highway=*` ways (3 of them with `area=yes`, which `extract_roads_from_osm()` filters out).

| highway          | Count | Length (m) | in `highway_defaults`?                      |
|------------------|-------:|----------:|---------------------------------------------|
| track            |    133 |    31,848 | yes                                         |
| path             |    115 |    28,726 | yes                                         |
| primary          |    105 |    14,598 | **no** → fallback `unclassified`            |
| secondary        |     64 |    14,251 | yes                                         |
| service          |     75 |     8,837 | yes                                         |
| trunk            |      1 |     3,458 | **no** (Gotthard road tunnel, A2)           |
| construction     |      1 |     2,533 | **no** (2nd Gotthard tube under construction, tunnel!) |
| primary_link     |     23 |     1,699 | **no** → fallback `unclassified`            |
| unclassified     |      6 |     1,029 | yes                                         |
| footway          |      6 |       748 | yes                                         |
| residential      |      3 |       643 | yes                                         |
| pedestrian       |      1 |       219 | **no** (`area=yes`, is filtered)            |
| steps            |      7 |       202 | yes                                         |

Relevant tags on the ways (excerpt):

| Tag                                   | Count | Evaluated today?                             |
|---------------------------------------|-------:|----------------------------------------------|
| `surface`                             |    276 | partially (only 9 values in `surface_overrides`) |
| `lanes`                               |    186 | yes (only `lanes × 3.25 m`)                  |
| `lanes:forward` / `lanes:backward`    |    158 | no                                           |
| `maxspeed`                            |    183 | no                                           |
| `layer`                               |     95 | no                                           |
| `bridge`                              |     69 | yes (bridge mesh)                            |
| `oneway`                              |     68 | **no** (17× `yes`, 13 of them links)         |
| `priority_road`                       |     52 | no                                           |
| `incline`                             |     43 | no                                           |
| `motorroad`                           |     41 | **no** (Swiss expressway A2/A2P)             |
| `tunnel`                              |     30 | yes (28× `yes`, 2× `avalanche_protector`)    |
| `lane_markings`                       |     11 | no (10× `no`, e.g. Tremola)                  |
| `destination*`, `turn:lanes*`         |    ~25 | no                                           |
| `width`                               |      9 | yes                                          |
| `embankment`                          |      4 | no                                           |
| `covered`                             |      4 | no (1× without `tunnel` tag, see below)      |

---

## 2. Ramps (`*_link`) and `motorway_junction`

The map contains **23 `primary_link` ways** (on- and off-ramps of the expressway-like main road 2 / A2P,
`motorroad=yes`). In addition there are **two `highway=motorway_junction` nodes**, both named "Motto Bartola":

| Node       | Location      | Ways involved                                                                   |
|------------|---------------|---------------------------------------------------------------------------------|
| 24869503   | at ground level | 422732675 primary (2), 956406313 primary (4), 26245062 / 26245219 primary_link |
| 3688461068 | **on bridge** | 129718739 primary (2, bridge, layer 1), 44220547 primary (4, bridge, **layer 2**), 1036670761 / 1036670763 primary_link (bridge, layer 1, oneway) |

What the current code does with links:

- `OSMMapper.get_road_properties()` shortens `primary_link` → `primary`. Since `primary` (like `trunk` and `motorway`) is not
  in `highway_defaults` at all, links **and** main roads end up with the `unclassified` default (5 m). They only get wider
  if `lanes` is set. For 3 links without `lanes` it stays at 5 m, too wide for a single-lane ramp.
- `oneway=yes` is not read anywhere. The DecalRoad gets no `oneWay`, so the AI drives ramps in both directions.
- Links get the same material as the main carriageway. There is no "ramp" distinction (no median,
  one-sided edge line).

### 2.1 The "Motto Bartola" Bridge Case (Node 3688461068)

Local geometry around the node (x/y in m, node = 0/0):

```
44220547  primary  4 lanes  layer 2   (0,0) → (36,30)      → continues as 44220548 (4 lanes, no bridge)
129718739 primary  2 lanes  layer 1   (-9,-11) → (0,0)
1036670761 primary_link 1 lane, oneway (0,0) → (-5,-2) → … → (-27,-23)   Exit "Motto Bartola"
1036670763 primary_link 1 lane, oneway (-17,-31) → … → (-1,-4) → (0,0)   Entrance
man_made=bridge 1036670764 (layer 1, beam) – outline of the actual bridge structure, 16 nodes
```

This is a **weaving/widening junction**: the two-lane main carriageway and the two single-lane ramps run almost parallel
for the first ~10–20 m (angle ≈ 15–25°) and merge at the node into a four-lane cross-section.
Why the bridge structure fails there:

1. **One mesh per way:** `_build_bridges()` / `build_bridge_mesh()` builds a separate deck for each of the 4 ways, with
   its own curbs and railings. Where the ramp and main carriageway run in parallel, the decks overlap.
   The railings and curbs on the inner sides then stand **in the middle of the carriageway** of the neighboring way.
2. **Width jump at the node:** 2 lanes (6.5 m) end, 4 lanes (13 m) begin at exactly the same point.
   The flat ends (`cap_style=2`) do not fit together; steps and gaps remain in between.
3. **Layer contradiction:** The four-lane part has `layer=2`, the other three ways `layer=1`, although they share a node.
   `layer` is currently not evaluated. Anyone who uses it for the deck height in the future must not take it literally here
   (OSM `layer` is only a relative ordering, not a height).
4. **The real structure is known but ignored:** `man_made=bridge` 1036670764 provides the footprint of the deck
   that carries the ramps and main carriageway together.

Similar but less problematic: `101008815` (primary_link, bridge, 17 m, 2 lanes) connects two ramps. A single deck without
overlap, so it should work. That still needs to be verified, though.

---

## 3. Other Special Elements We Do Not Handle Yet

### 3.1 Road Types / Filters

| Element                                    | Problem today                                                                 |
|--------------------------------------------|-------------------------------------------------------------------------------|
| `highway=construction` (798912292, `construction=trunk`, `tunnel=yes`, 2.5 km) | is built as a normal tunnel of 6.5 m although the tube is still under construction. Must be filtered (likewise `proposed`, `abandoned`, `razed`). |
| `highway=trunk` (Gotthard road tunnel, 3.5 km) | no entry in `highway_defaults`. The width is only correct thanks to `lanes=2`. |
| `highway=primary` (105 ways)               | no entry in `highway_defaults` (see above).                                     |
| `motorway`, `motorway_link`, `trunk_link`, `secondary_link`, `tertiary_link` | do not occur here, but are missing as well. Needed for maps with a real motorway. |
| `covered=yes` without `tunnel` (746194686) | **Implemented 2026-09-24:** built as a gallery with negative `layer` (open toward the valley side determined by majority); tunnel ↔ gallery transitions with portal wall, see `docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md`. |

### 3.2 Surfaces Missing from `surface_overrides`

| surface   | Ways | affects                                     | falls back to today     |
|-----------|-----:|---------------------------------------------|-------------------------|
| sett      |   28 | **Tremola** (historic cobblestone road), Via San Gottardo | asphalt           |
| concrete  |   31 | main road 2, ramps, service                 | asphalt (acceptable)    |
| rock      |    6 | hiking trails                               | dirt track              |
| grass     |    4 | paths                                       | dirt track              |
| unpaved   |    3 | track/path/service                          | inconsistent (service → asphalt!) |

For the Tremola a dedicated cobblestone material (`sett`/`cobblestone`) is worthwhile; it is *the* landmark of the map.
The `unpaved` service example shows a gap in the mapper: unmapped surfaces keep the
highway default instead of being recognized as "unpaved".

### 3.3 Width Inconsistencies

- `secondary` without `lanes` → 7.0 m, with `lanes=2` → 6.5 m.
- `lanes × 3.25` applies equally to all types, including the Tremola (narrow cobblestone road without markings) and
  service roads with `lanes=2`.
- No shoulders on expressways (`motorroad=yes`, `shoulder`).
- No transition (taper) when the lane count changes. 2 → 3 → 4 lanes jump abruptly. Affects 13 `lanes=3` and
  5 `lanes=4` ways.

### 3.4 Node Tags on Road Nodes (completely ignored so far)

| Node tag                      | Count | Idea                                                  |
|-------------------------------|-------:|-------------------------------------------------------|
| `ford=yes`                    |     15 | Ford: do not raise the road over the stream River, water decal |
| `highway=give_way` / `stop`   |  10/1  | Traffic-sign props, AI right of way                    |
| `highway=passing_place`       |      4 | local widening (passing place)                         |
| `highway=emergency_bay`       |      2 | Emergency stopping bay on the main road                |
| `highway=motorway_junction`   |      2 | Exit sign, marking for the weaving area                |
| `barrier=gate/lift_gate/swing_gate/block/bollard` | 14 | Barriers/bollards as props           |
| `highway=milestone`           |      3 | Decoration                                             |
| `highway=crossing`            |      1 | Zebra-crossing decal                                   |

### 3.5 Other OSM Objects Related to Roads

- **`man_made=bridge`** (21 areas): real footprints of the bridges, see 2.1.
- **`area:highway=*`** (22 areas, 1× `motorway`): exact carriageway areas, good for junctions and widenings.
- **Turn restrictions** (`type=restriction`, 22 relations): `no_left_turn` 9, `only_straight_on` 7, `no_u_turn` 5,
  `only_right_turn` 1. Relevant for AI navigation.
- **`type=tunnel` relations** (Gotthard road and rail tunnels): combine several tunnel ways into one structure.
- **`man_made=avalanche_protection`** (96 ways): avalanche barriers, not roads, but visually prominent.

---

## 4. Recommended Order

1. **Mapper basics** (small, immediately effective): `motorway`/`trunk`/`primary` + all `*_link` in `highway_defaults`
   (links single-lane ~4 m), missing surfaces (`sett`, `concrete`, `rock`, `grass`, `unpaved`),
   filter `construction`/`proposed`.
   **Implemented 2026-09-24:**
   - `highway_defaults`: motorway 8.0 / trunk 7.5 / primary 7.5 m, `*_link` 4.0 m (tertiary_link 3.75 m). The mapper
     now looks up the exact type first, and only then the base type.
   - New surface `cobblestone_road`: ground model `COBBLESTONE`, textures from BeamNG's `tileable/stone/italy_cobblestone`,
     priority 6. It applies to `sett`, `cobblestone`, `unhewn_cobblestone` and `paving_stones`.
   - `unpaved` → gravel, `grass`/`rock` → dirt track.
   - `concrete` deliberately stays at the highway default. The concrete material is intended for footways (`drivability` 0).
   - `extract_roads_from_osm()` discards `highway=construction/proposed/planned/abandoned/disused/razed/demolished`.
   - Still to be checked in game: scale and tiling of the cobblestone texture on the Tremola (DecalRoad stretches the texture
     across the full width). Without an `opacityMap` the edge also has a hard border.
2. **Travel direction in the DecalRoad:** `oneway` → `oneWay`/lane counts (`lanesLeft`/`lanesRight` from
   `lanes:forward`/`lanes:backward`), so the AI drives ramps correctly. Check the field names beforehand against a
   BeamNG reference level (`items.level.json`).
3. **Bridges at branches** (Motto Bartola): group bridge ways that share a node into one structure.
   Build the deck as the union of the carriageway polygons (or directly from `man_made=bridge`), railings only on the
   outer contour, shared piers.
4. **Lane-count changes with a transition** (taper over ~30–50 m instead of an abrupt jump), applies to DecalRoad and bridge deck.
5. **Node features** (fords, passing places, barriers, signs) as a separate stage.
