#include "voxelizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>

__global__ void voxelize(const Point* points, size_t numPoints, float voxelSize) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= numPoints) return;
}


std::vector<Point> voxelizeOnGPU(const PointCloudVecs &hostPoints, size_t totalPoints, float voxelSize) {
    
    if (totalPoints == 0){
        std::cerr << "No points to voxelize\n";
        return std::vector<Point>();
    }

    Point* d_points;
    cudaMalloc((void**)&d_points, totalPoints * sizeof(Point));
    cudaMemcpy(d_points, &hostPoints, totalPoints * sizeof(Point), cudaMemcpyHostToDevice);


    std::vector<Point> voxelizedPoints;

    cudaFree(d_points);
    return voxelizedPoints;

}

std::vector<Point> voxelizeOnCPU(
    const PointCloudVecs &hostPoints,
    size_t totalPoints,
    float voxelSize)
{
    if (totalPoints == 0){
        std::cerr << "No points to voxelize\n";
        return std::vector<Point>();
    }

    std::vector<Point> voxelizedPoints;

    return voxelizedPoints;
}