import numpy as np
import open3d as o3d
import laspy as lp

pcd_path = "project/pycode/DATA/neighborhood.las"

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