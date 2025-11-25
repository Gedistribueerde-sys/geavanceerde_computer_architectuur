import numpy as np
import open3d as o3d

# --- 1. Definities ---
GRID_SIZE = 80.0  # De bounding box grootte (0 tot 80)
VOXEL_SIZE = 2.0  # Elke voxel is 2x2x2 eenheden

# 2. Genereer Dichte Raster Punten (De "Chaos Cube" is vol)
# Dit creëert 80x80x80 = 512,000 dichte punten.
xs, ys, zs = np.meshgrid(
    np.arange(0, GRID_SIZE, VOXEL_SIZE, dtype=float),  # Start bij 0, stap VOXEL_SIZE
    np.arange(0, GRID_SIZE, VOXEL_SIZE, dtype=float),
    np.arange(0, GRID_SIZE, VOXEL_SIZE, dtype=float),
    indexing="ij"
)
# We gebruiken hier de grid punten zelf als de centra, dit is eenvoudiger 
# dan de 512.000 punten genereren en dan te voxelizeren
unique_voxel_centers = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T

# Converteer de centers naar hun Voxel Indices [i, j, k]
# Voxel Index = Center / Voxel Grootte
unique_voxel_indices = (unique_voxel_centers / VOXEL_SIZE).astype(int)

# --- 3. Randdetectie voor Oppervlakte-Extractie 🛡️ ---

# Maak een set voor extreem snelle lookup van bezette indices (O(1))
occupied_voxels_set = set(map(tuple, unique_voxel_indices))
surface_voxels_indices = []

# Definieer de 6 directe buren-offsets
neighbors = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
])

# Itereren over elke unieke bezette voxel
for ijk in unique_voxel_indices:
    is_surface = False
    
    # Controleer of ten minste één van de 6 buren NIET bezet is
    for offset in neighbors:
        neighbor_idx = (ijk + offset)
        neighbor_tuple = tuple(neighbor_idx)

        # De buur is de 'lege' ruimte als hij niet in de set van bezette voxels zit
        if neighbor_tuple not in occupied_voxels_set:
            is_surface = True
            break
            
    if is_surface:
        surface_voxels_indices.append(ijk)

# Converteer de lijst van oppervlakte-indices terug naar een NumPy array
surface_voxels_array = np.array(surface_voxels_indices)

# Bereken de CENTRA van de oppervlakte-voxels
# Center = Voxel Index * Voxel Grootte + (Voxel Grootte / 2)
surface_centers = (surface_voxels_array * VOXEL_SIZE) + (VOXEL_SIZE / 2)

# --- 4. Visualisatie van Oppervlakte Voxels 🎨 ---

print(f"Totaal unieke voxels: {len(unique_voxel_indices)}")
print(f"Gereduceerd naar oppervlakte voxels: {len(surface_voxels_array)}")

mesh_list = []
color = [0.0, 0.7, 0.0] # Groen voor de rand

for center in surface_centers:
    # Maak een kubus gecentreerd op het voxel center
    cube = o3d.geometry.TriangleMesh.create_box(width=VOXEL_SIZE, height=VOXEL_SIZE, depth=VOXEL_SIZE)
    
    # Vertaal de kubus zodat zijn middelpunt op 'center' ligt
    cube.translate(center - VOXEL_SIZE / 2)
    
    # Wijs kleur toe en maak de lijnen duidelijker (door schaduw te berekenen)
    cube.paint_uniform_color(color)
    cube.compute_vertex_normals()
    mesh_list.append(cube)

# Combineer alle meshes in één object
surface_voxel_mesh = o3d.geometry.TriangleMesh()
for mesh in mesh_list:
    surface_voxel_mesh += mesh

# Visualiseer alleen de oppervlaktevoxels
o3d.visualization.draw_geometries([surface_voxel_mesh], 
                                 window_name="Oppervlakte Voxel Blocks (Binnenkant Verwijderd)")