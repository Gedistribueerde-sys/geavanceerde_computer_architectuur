#include "voxelizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>

// __global__ void voxelize(const Point* points, size_t numPoints, float voxelSize) {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;

//     if (tid >= numPoints) return;
// }


// std::vector<Point> voxelizeOnGPU(const PointCloudVecs &hostPoints, size_t totalPoints, float voxelSize) {
    
//     if (totalPoints == 0){
//         std::cerr << "No points to voxelize\n";
//         return std::vector<Point>();
//     }

//     Point* d_points;
//     cudaMalloc((void**)&d_points, totalPoints * sizeof(Point));
//     cudaMemcpy(d_points, &hostPoints, totalPoints * sizeof(Point), cudaMemcpyHostToDevice);


//     std::vector<Point> voxelizedPoints;

//     cudaFree(d_points);
//     return voxelizedPoints;

// }

// std::vector<Point> voxelizeUniformOnGPU(const PointCloudVecs &hostPoints, size_t totalPoints, float voxelSize){

// }



std::vector<Point> voxelizeUniformOnCPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize)
{
    if (totalPoints == 0){
        std::cerr << "No points to voxelize\n";
        return std::vector<Point>();
    }

    // generate a map with keys and an accumalator
    // this is for easy accumalating points that belong to same voxel
    std::map<VoxelKey, VoxelAccumulator> voxelMap;

    for (size_t i = 0; i < totalPoints; i++) {
        int ix = static_cast<int>(floor(hostPoints.x[i] / voxelSize));
        int iy = static_cast<int>(floor(hostPoints.y[i] / voxelSize));
        int iz = static_cast<int>(floor(hostPoints.z[i] / voxelSize));
        VoxelKey key = VoxelKey(ix, iy, iz);

        VoxelAccumulator& acc = voxelMap[key];
        acc.sumX += hostPoints.x[i];
        acc.sumY += hostPoints.y[i];
        acc.sumZ += hostPoints.z[i];
        acc.sumR += static_cast<uint32_t>(hostPoints.r[i]);
        acc.sumG += static_cast<uint32_t>(hostPoints.g[i]);
        acc.sumB += static_cast<uint32_t>(hostPoints.b[i]);
        acc.count++;
        
    }

    // transform the accumulated points to Point
    std::vector<Point> voxelizedPoints;
    voxelizedPoints.reserve(voxelMap.size());

    for (const auto& [key, acc] : voxelMap) {
        Point p;
        p.x = static_cast<float>(std::max(0.0, acc.sumX / acc.count));
        p.y = static_cast<float>(std::max(0.0, acc.sumY / acc.count));
        p.z = static_cast<float>(std::max(0.0, acc.sumZ / acc.count));
        p.r = static_cast<uint8_t>(acc.sumR / acc.count);
        p.g = static_cast<uint8_t>(acc.sumG / acc.count);
        p.b = static_cast<uint8_t>(acc.sumB / acc.count);
        voxelizedPoints.push_back(p);
    }

    return voxelizedPoints;
}