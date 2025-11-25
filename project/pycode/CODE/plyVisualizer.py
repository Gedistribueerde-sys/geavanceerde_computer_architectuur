import numpy as np
import open3d as o3d
import sys

# File paths - adjust these or pass as arguments
ORIGINAL_PLY = "../../cudaproject/build/original.ply"
VOXELIZED_PLY = "../../cudaproject/build/voxelized.ply"
VOXEL_SIZE = 0.5  # Must match the voxel size used in voxelization

def load_ply(filepath):
    """Load a PLY file and return an Open3D point cloud."""
    print(f"Loading: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    print(f"  Loaded {len(pcd.points)} points")
    return pcd

def visualize_side_by_side(original_pcd, voxelized_pcd):
    """Visualize original and voxelized point clouds side by side."""
    
    # Get bounds of original to calculate offset
    bounds = original_pcd.get_axis_aligned_bounding_box()
    offset = bounds.get_max_bound()[0] - bounds.get_min_bound()[0] + 10  # X-axis offset with gap
    
    # Create a copy of voxelized and translate it
    voxelized_shifted = o3d.geometry.PointCloud(voxelized_pcd)
    voxelized_shifted.translate([offset, 0, 0])
    
    print("\nVisualization:")
    print(f"  Left:  Original ({len(original_pcd.points)} points)")
    print(f"  Right: Voxelized ({len(voxelized_pcd.points)} points)")
    print(f"  Compression: {len(original_pcd.points) / len(voxelized_pcd.points):.2f}x")
    
    o3d.visualization.draw_geometries(
        [original_pcd, voxelized_shifted],
        window_name="Original (left) vs Voxelized (right)",
        width=1600,
        height=900
    )

def visualize_single(pcd, title="Point Cloud"):
    """Visualize a single point cloud."""
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
        width=1200,
        height=800
    )

def main():
    # Handle command line arguments
    if len(sys.argv) >= 3:
        original_path = sys.argv[1]
        voxelized_path = sys.argv[2]
    else:
        original_path = ORIGINAL_PLY
        voxelized_path = VOXELIZED_PLY
    
    # Load point clouds
    original_pcd = load_ply(original_path)
    voxelized_pcd = load_ply(voxelized_path)
    
    # Print statistics
    print("\n=== Point Cloud Statistics ===")
    print(f"Original points:   {len(original_pcd.points)}")
    print(f"Voxelized points:  {len(voxelized_pcd.points)}")
    print(f"Compression ratio: {len(original_pcd.points) / len(voxelized_pcd.points):.2f}x")
    
    # Menu for visualization options
    while True:
        print("\n=== Visualization Options ===")
        print("1. Side-by-side comparison (points)")
        print("2. Original only")
        print("3. Voxelized only (points)")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            visualize_side_by_side(original_pcd, voxelized_pcd)
        elif choice == '2':
            visualize_single(original_pcd, "Original Point Cloud")
        elif choice == '3':
            visualize_single(voxelized_pcd, "Voxelized Point Cloud")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option, try again.")

if __name__ == "__main__":
    main()
