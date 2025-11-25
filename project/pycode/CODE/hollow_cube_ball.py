import numpy as np
import open3d as o3d


    
def extract_surface_voxels(occupied_indices: np.ndarray) -> np.ndarray:
    """
    Neemt een array van unieke bezette voxel-indices en retourneert alleen
    de indices die grenzen aan een lege ruimte (het oppervlak).
    """
    # 1. Maak een set voor snelle lookup (O(1))
    occupied_voxels_set = set(map(tuple, occupied_indices))
    surface_voxels_indices = []

    # 2. Definieer de 6 directe buren-offsets
    neighbors = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])

    # 3. Itereer over alle bezette voxels en controleer op rand
    for ijk in occupied_indices:
        is_surface = False
        
        for offset in neighbors:
            neighbor_idx = (ijk + offset)
            neighbor_tuple = tuple(neighbor_idx)

            # Als de buur niet in de bezette set zit, is het een randvoxel
            if neighbor_tuple not in occupied_voxels_set:
                is_surface = True
                break
                
        if is_surface:
            surface_voxels_indices.append(ijk)

    return np.array(surface_voxels_indices)

# --- Configuratie ---
GRID_SIZE = 200.0
VOXEL_SIZE = 2.0
GRID_INDICES = int(GRID_SIZE / VOXEL_SIZE) 

# Genereer alle mogelijke Voxel Indices en Centers
indices_range = np.arange(GRID_INDICES)
i, j, k = np.meshgrid(indices_range, indices_range, indices_range, indexing="ij")
all_indices = np.vstack([i.ravel(), j.ravel(), k.ravel()]).T
all_centers = (all_indices * VOXEL_SIZE) + (VOXEL_SIZE / 2)


# een paar kubussen aanmaken
CUBE1_MIN = 10.0
CUBE1_MAX = 50.0
is_in_cube1 = (
    (all_centers[:, 0] >= CUBE1_MIN) & (all_centers[:, 0] <= CUBE1_MAX) &
    (all_centers[:, 1] >= CUBE1_MIN) & (all_centers[:, 1] <= CUBE1_MAX) &
    (all_centers[:, 2] >= CUBE1_MIN) & (all_centers[:, 2] <= CUBE1_MAX)
)
cube1_indices = all_indices[is_in_cube1]

CUBE2_MIN = 30.0
CUBE2_MAX = 70.0
is_in_cube2 = (
    (all_centers[:, 0] >= CUBE2_MIN) & (all_centers[:, 0] <= CUBE2_MAX) &
    (all_centers[:, 1] >= CUBE2_MIN) & (all_centers[:, 1] <= CUBE2_MAX) &
    (all_centers[:, 2] >= CUBE2_MIN) & (all_centers[:, 2] <= CUBE2_MAX)
)
cube2_indices = all_indices[is_in_cube2]


# een paar bollen aanmaken
BALL_CENTER = np.array([30.0, 50.0, 30.0]) 
BALL_RADIUS = 25.0
shifted_centers = all_centers - BALL_CENTER
distances = np.linalg.norm(shifted_centers, axis=1)
is_in_ball = (distances <= BALL_RADIUS)
ball_indices = all_indices[is_in_ball]

BALL2_CENTER = np.array([10.0, 40.0, 60.0]) 
BALL2_RADIUS = 25.0
shifted2_centers = all_centers - BALL2_CENTER
distances2 = np.linalg.norm(shifted2_centers, axis=1)
is_in_ball2 = (distances2 <= BALL2_RADIUS)
ball_indices2 = all_indices[is_in_ball2]

BALL3_CENTER = np.array([110.0, 40.0, 60.0]) 
BALL3_RADIUS = 40.0
shifted3_centers = all_centers - BALL3_CENTER
distances3 = np.linalg.norm(shifted3_centers, axis=1)
is_in_ball3 = (distances3 <= BALL3_RADIUS)
ball_indices3 = all_indices[is_in_ball3]

combined_indices=np.concatenate((cube1_indices, cube2_indices,ball_indices2 ,ball_indices3, ball_indices), axis=0)


occupied_indices = np.unique(combined_indices, axis=0) # Unieke indices = Union


occupied_centers = (occupied_indices * VOXEL_SIZE) + (VOXEL_SIZE / 2)

#maak een pointcloud
pcd_occupied = o3d.geometry.PointCloud()
pcd_occupied.points = o3d.utility.Vector3dVector(occupied_centers)

# kleur
color_occupied = [1.0, 0.0, 0.0] 
colors = np.tile(color_occupied, (len(occupied_centers), 1))
pcd_occupied.colors = o3d.utility.Vector3dVector(colors)

 # visualisatie 
o3d.visualization.draw_geometries([pcd_occupied], 
                                 window_name="Input Puntenwolk (Bezette Voxel Centers)",
                                 point_show_normal=False)

# Dit is de enige stap die nodig is op de input data:
surface_voxels_array = extract_surface_voxels(occupied_indices)


# visualisatie van de voxels

surface_centers = (surface_voxels_array * VOXEL_SIZE) + (VOXEL_SIZE / 2)

print(f"Totaal bezette voxels (Kubus1 + Kubus2 + Bol Union): {len(occupied_indices)}")
print(f"Gereduceerd naar oppervlakte voxels: {len(surface_voxels_array)}")

mesh_list = []
color = [0.0, 0.5, 0.5] # Cyaan

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
                                 window_name="Samengestelde Oppervlakte (2 Kubussen + 1 Bol)")