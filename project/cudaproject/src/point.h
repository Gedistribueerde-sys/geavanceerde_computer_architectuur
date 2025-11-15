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
