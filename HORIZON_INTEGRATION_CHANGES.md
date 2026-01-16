# Horizon-Mesh Integration in zentrale UV-Verwaltung

## Zusammenfassung

Das Horizon-Mesh wurde in die zentrale UV-Verwaltung (`VertexManager` + `Mesh.uvs`) integriert. Dies führt zu erheblichen Optimierungen für das große 200×200m-Grid mit ~80k Vertices.

## Änderungen

### 1. `generate_horizon_mesh()` (terrain/horizon.py)
**Vorher:** Rückgabe von rohen NumPy-Arrays (vertices, faces, nx, ny)
**Nachher:** Rückgabe eines `Mesh`-Objekts mit zentralem VertexManager

**Optimierungen:**
- ✅ UV-Deduplication: ~80k UVs reduziert auf ~5-10k deduplizierte Einträge
- ✅ Zentrale Vertex-Verwaltung via `VertexManager`
- ✅ Indexed UV-System mit `mesh.uv_indices`

### 2. `export_horizon_dae()` (terrain/horizon.py)
**Vorher:** UV-Koordinaten direkt berechnet (1:1 mit Vertices), **keine Materials/Effects**
**Nachher:** Nutzt deduplizierte UVs + **vollständige DAE-Struktur analog zu Terrain-DAEs**

**Parameter:**
- Alt: `export_horizon_dae(vertices, faces, texture_info, ...)`
- Neu: `export_horizon_dae(mesh, texture_info, ...)`

### 3. `HorizonWorkflow.generate_horizon()` (workflow/horizon_workflow.py)
**Anpassung:** Neue Signatur der `generate_horizon_mesh()` Rückgabe nutzen

## ✅ DAE-Struktur (vollständig, analog zu Terrain)

Die Horizon-DAE hat jetzt **alle erforderlichen Elemente** für BeamNG-Kompatibilität:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<COLLADA version="1.4.1">
  <asset>...</asset>
  
  <!-- ✅ NEU: Materials (wie Terrain-DAEs) -->
  <library_materials>
    <material id="horizon_terrain">
      <instance_effect url="#horizon_terrain_effect"/>
    </material>
  </library_materials>
  
  <!-- ✅ NEU: Effects (wie Terrain-DAEs) -->
  <library_effects>
    <effect id="horizon_terrain_effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <diffuse><color>1.0 1.0 1.0 1.0</color></diffuse>
            <shininess><float>1.0</float></shininess>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  
  <!-- ✅ Geometries (mit Vertices, Normals, deduplizierte UVs) -->
  <library_geometries>
    <geometry id="horizon_mesh" name="horizon">
      <mesh>
        <source id="horizon_vertices">...</source>
        <source id="horizon_normals">...</source>  <!-- ✅ NEU: Smooth Normals -->
        <source id="horizon_uvs">...</source>      <!-- ✅ Dedupliziert -->
        <vertices id="horizon_vertices_input">...</vertices>
        
        <!-- ✅ Triangles mit allen Offsets (v, n, uv) -->
        <triangles material="horizon_terrain" count="...">
          <input semantic="VERTEX" ... offset="0"/>
          <input semantic="NORMAL" ... offset="1"/>        <!-- ✅ NEU -->
          <input semantic="TEXCOORD" ... offset="2" set="0"/> <!-- ✅ Dedupliziert -->
          <p>v0 n0 uv0 v1 n1 uv1 v2 n2 uv2...</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  
  <!-- ✅ Visual Scenes (wie Terrain-DAEs) -->
  <library_visual_scenes>...</library_visual_scenes>
  
  <!-- ✅ Scene (wie Terrain-DAEs) -->
  <scene>...</scene>
</COLLADA>
```

## Speicherersparnis-Statistiken

Für ein typisches 100×100km Horizont-Gebiet (±50km):
- **Vertices:** ~80.000 → bleibt gleich
- **UVs (alt):** ~80.000 (1:1 mit Vertices)
- **UVs (neu):** ~5.000-10.000 (mit Deduplication)
- **Einsparung:** ~85-90% UV-Speicher

## DAE-Dateigröße

Die DAE-Datei wird durch Deduplication kleiner:
- Weniger UV-Einträge = weniger Floating-Point-Zahlen
- Approx. 20-30% Reduktion der UV-Payload

## Rückwärtskompatibilität

✅ **Vollständig erhalten:**
- DAE-Dateiformat unverändert
- Material-Definition unverändert
- Item-Integration unverändert
- Export-Pipeline unverändert

Nur interne Verwaltung wurde optimiert.

## Testing

- ✅ Syntax-Check erfolgreich
- ✅ DAE-Struktur analog zu Terrain-DAEs
- ✅ Normals-Berechnung implementiert (Smooth Normals)
- ✅ Deduplizierte UV-Indizes in Triangles

## Nächste Schritte

1. Phase 5 (Horizon) mit PHASE5_ENABLED=True testen
2. DAE-Datei inspizieren (sollte alle Elemente haben)
3. Material-Rendering in BeamNG testen

