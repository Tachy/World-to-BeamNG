#!/usr/bin/env python3
"""Test PyVista line rendering"""

import numpy as np
import pyvista as pv

# Erstelle einfache Test-Linien
vertices = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [2, 1, 0],
        [2, 2, 0],
    ]
)

edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
]

print(f"Vertices: {vertices.shape}")
print(f"Edges: {len(edges)}")

# Methode 1: Mit [[2, idx1, idx2], ...]
cells_method1 = np.array([[2, edge[0], edge[1]] for edge in edges]).flatten()
print(f"Method 1 cells: {cells_method1}")

mesh1 = pv.PolyData(vertices, cells_method1)
print(f"Method 1 mesh: {mesh1}")

# Methode 2: Mit pyvista cells format
cells_method2 = np.hstack([[2, edge[0], edge[1]] for edge in edges])
print(f"Method 2 cells: {cells_method2}")

mesh2 = pv.PolyData(vertices, cells_method2)
print(f"Method 2 mesh: {mesh2}")

# Methode 3: Mit Lines
mesh3 = pv.PolyData()
mesh3.points = vertices
for edge in edges:
    mesh3.lines = np.hstack(mesh3.lines, [2, edge[0], edge[1]])
print(f"Method 3 mesh: {mesh3}")

# Teste Rendering
plotter = pv.Plotter()
plotter.add_mesh(mesh1, color="red", line_width=5)
plotter.add_title("Test PyVista Lines")
plotter.show()
