# Point-Cloud to Voxel
## Goals
### Level 0: Create Random Point-Cloud
- Create a random 3d point-cloud for example 100x100x100 points.
- Each point should have an RGB color.
- Now create a voxel grid that has a voxel size of 80x80x80.
- Snap each point to the nearest voxel center.

### Level 1: Voxelization with CUDA
- Read a .las file containing point-cloud data.
- Use CUDA to snap points to a voxel grid.

### Level 2: Voxel filtering 
- Read a .las file containing point-cloud data.
- Now instead of snapping points in to the middle of the voxel, filter points inside each voxel and compute the centroid.

## Links Websites
- https://www.open3d.org/docs/release/tutorial/geometry/voxelization.html
- https://medium.com/data-science/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d
- https://pointclouds.org/documentation/tutorials/voxel_grid.html

## Links GitHub
- https://github.com/Forceflow/cuda_voxelizer/tree/main
