import numpy as np
import open3d as o3d
import laspy as lp

pcd_path = "../DATA/neighborhood.las"

VOXEL_SIZE=0.5

# Import dataset with laspy
point_cloud = lp.read(pcd_path)
xyz = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
rgb = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose() / 65535

# Transform to open3d point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# getting the minimas
mins = np.min(xyz, axis=0)
maxs = np.max(xyz, axis=0)

#normalise this
print("some basic printouts ")
print (mins)
print("-------")
print (maxs)
xyz_shifted = xyz - mins


voxel_indices = (xyz_shifted // VOXEL_SIZE).astype(int)

snapped_points = (voxel_indices * VOXEL_SIZE) + (VOXEL_SIZE / 2)

# Visualize Snapped Points as Voxel Blocks
# Create a mesh for each voxel
mesh_list = []
for i, (center, color) in enumerate(zip(snapped_points, rgb)):
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