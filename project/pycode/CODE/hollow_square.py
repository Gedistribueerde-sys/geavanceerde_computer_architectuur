import numpy as np
import open3d as o3d

GRID_SIZE = 80.0 
VOXEL_SIZE = 2.0  

xs, ys, zs = np.meshgrid(
    np.arange(0, GRID_SIZE, VOXEL_SIZE, dtype=float),  # Start bij 0, stap VOXEL_SIZE
    np.arange(0, GRID_SIZE, VOXEL_SIZE, dtype=float),
    np.arange(0, GRID_SIZE, VOXEL_SIZE, dtype=float),
    indexing="ij"
)

unique_voxel_centers = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T
unique_voxel_indices = (unique_voxel_centers / VOXEL_SIZE).astype(int)
occupied_voxels_set = set(map(tuple, unique_voxel_indices))
surface_voxels_indices = []


neighbors = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
])

for ijk in unique_voxel_indices:
    is_surface = False
    
    
    for offset in neighbors:
        neighbor_idx = (ijk + offset)
        neighbor_tuple = tuple(neighbor_idx)

        
        if neighbor_tuple not in occupied_voxels_set:
            is_surface = True
            break
            
    if is_surface:
        surface_voxels_indices.append(ijk)


surface_voxels_array = np.array(surface_voxels_indices)


surface_centers = (surface_voxels_array * VOXEL_SIZE) + (VOXEL_SIZE / 2)

# 

print(f"Totaal unieke voxels: {len(unique_voxel_indices)}")
print(f"Gereduceerd naar oppervlakte voxels: {len(surface_voxels_array)}")

mesh_list = []
color = [0.0, 0.7, 0.0] # Groen voor de rand

for center in surface_centers:
   
    cube = o3d.geometry.TriangleMesh.create_box(width=VOXEL_SIZE, height=VOXEL_SIZE, depth=VOXEL_SIZE)
    
   
    cube.translate(center - VOXEL_SIZE / 2)
    

    cube.paint_uniform_color(color)
    cube.compute_vertex_normals()
    mesh_list.append(cube)


surface_voxel_mesh = o3d.geometry.TriangleMesh()
for mesh in mesh_list:
    surface_voxel_mesh += mesh

o3d.visualization.draw_geometries([surface_voxel_mesh], 
                                 window_name="Oppervlakte Voxel Blocks (Binnenkant Verwijderd)")