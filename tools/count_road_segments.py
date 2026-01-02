import json

data = json.load(
    open(
        "C:/Users/johan/AppData/Local/BeamNG.drive/0.36/levels/World_to_BeamNG/art/shapes/debug_junctions.json"
    )
)
roads = [r for r in data["roads"] if r.get("id") == 50822566002]

print(f"Road-ID 50822566002: {len(roads)} Vorkommen")
for i, r in enumerate(roads):
    print(f'  Segment {i+1}: {len(r["coords"])} Koordinaten')
