#pragma once
#include "las_reader.h"
#include <vector>

struct Voxel {
    float x, y, z;           // Center position
    uint8_t r, g, b;         // Average RGB
    uint32_t pointCount;     // Number of points in this voxel
};

// GPU voxelization function
void voxelizeOnGPU(
    const std::vector<Point>& hostPoints,
    float voxelSize,
    std::vector<Voxel>& outputVoxels
);