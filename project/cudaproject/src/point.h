#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

struct Point {
    float x, y, z;
    uint8_t r, g, b;  // RGB colors
};

struct PointCloudVecs {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<uint8_t> r;
    std::vector<uint8_t> g;
    std::vector<uint8_t> b;
};

struct PointCloudSoA {
    float* x;
    float* y;
    float* z;
    uint8_t* r;
    uint8_t* g;
    uint8_t* b;
    size_t numPoints;
};

struct VoxelGridSoA {
    float* sumX;
    float* sumY;
    float *sumZ;
    uint32_t* sumR;
    uint32_t* sumG;
    uint32_t* sumB;
    uint32_t* count;
    size_t gridSize;
};

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