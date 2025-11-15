import numpy as np
import open3d as o3d
import laspy as lp

pcd_path = "project/pycode/DATA/neighborhood.las"

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

# normalize the coordinates
xyz_shifted = xyz - mins


voxel_indices = (xyz_shifted // VOXEL_SIZE).astype(int)

unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
voxel_colors = np.zeros((len(unique_voxels), 3))

# Sum colors per voxel
np.add.at(voxel_colors, inverse, rgb)

# Count points per voxel
counts = np.bincount(inverse)

# Average colors
voxel_colors /= counts[:, None]

mesh_list = []
for center_idx, voxel in enumerate(unique_voxels):
    center = voxel * VOXEL_SIZE + VOXEL_SIZE/2
    color = voxel_colors[center_idx]

    cube = o3d.geometry.TriangleMesh.create_box(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)
    cube.translate(center - VOXEL_SIZE/2)
    cube.paint_uniform_color(color)
    mesh_list.append(cube)


# Combine all meshes
voxel_mesh = o3d.geometry.TriangleMesh()
for mesh in mesh_list:
    voxel_mesh += mesh

o3d.visualization.draw_geometries([voxel_mesh], window_name="Voxelized Blocks")