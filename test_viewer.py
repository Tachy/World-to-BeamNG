#!/usr/bin/env python3
import sys
import os

# Importiere MeshViewer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

from mesh_viewer import MeshViewer

print("Starte MeshViewer...")
viewer = MeshViewer()
print("MeshViewer erstellt, starte show()...")
viewer.show()
print("show() abgeschlossen")
