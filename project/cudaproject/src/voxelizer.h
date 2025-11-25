#pragma once
#include <vector>
#include "point.h"

struct VoxelKey {
    int x, y, z;

    VoxelKey(int ix, int iy, int iz): x(ix), y(iy), z(iz) {}

    bool operator<(const VoxelKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct VoxelAccumulator {
    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
    uint32_t sumR = 0, sumG = 0, sumB = 0;
    size_t count = 0;
};
// GPU voxelization function
// std::vector<Point> voxelizeOnGPU(
//     const std::vector<Point>& hostPoints,
//     size_t totalPoints,
//     float voxelSize
// );

// std::vector<Point> voxelizeUniformOnGPU(
//     const std::vector<Point>& hostPoints,
//     size_t totalPoints,
//     const std::vector<float>& voxelSizes
// );

// std::vector<Point> voxelizeHashingOnGPU(
//     const std::vector<Point>& hostPoints,
//     size_t totalPoints,
//     float voxelSize
// );

std::vector<Point> voxelizeUniformOnCPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize
);