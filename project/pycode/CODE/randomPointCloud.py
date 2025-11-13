import numpy as np
import open3d as o3d

NUM_POINTS = 10000
GRID_SIZE = 80.0  # The bounding box size (0 to 80)
VOXEL_SIZE = 2.0  # Each voxel is 2x2x2 units

# 2. Generate Random Points (The "Chaos Cube")
# Create random floats between 0 and 80
points = np.random.rand(NUM_POINTS, 3) * GRID_SIZE
colors = np.random.rand(NUM_POINTS, 3) # Random RGB

# Visualize Raw Points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd], window_name="Raw Points")


# Level 0: Voxelizing Point Cloud
voxel_indices = (points // VOXEL_SIZE).astype(int)

# Calculate center of each voxel
snapped_points = (voxel_indices * VOXEL_SIZE) + (VOXEL_SIZE / 2)

# Visualize Snapped Points as Voxel Blocks
# Create a mesh for each voxel
mesh_list = []
for i, (center, color) in enumerate(zip(snapped_points, colors)):
    # Create a cube centered at the voxel center
    cube = o3d.geometry.TriangleMesh.create_box(width=VOXEL_SIZE, height=VOXEL_SIZE, depth=VOXEL_SIZE)
    # Translate cube to voxel center
    cube.translate(center - VOXEL_SIZE / 2)
    # Assign color to the cube
    cube.paint_uniform_color(color)
    mesh_list.append(cube)

# Combine all meshes
voxel_mesh = o3d.geometry.TriangleMesh()
for mesh in mesh_list:
    voxel_mesh += mesh

o3d.visualization.draw_geometries([voxel_mesh], window_name="Voxelized Blocks")