#pragma once
#include <pdal/StageFactory.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/Dimension.hpp>
#include <limits>
#include <vector>
#include <string>
#include <iostream>

struct Point {
    float x, y, z;
    uint8_t r, g, b;  // RGB colors
};

std::vector<Point> readLASFileNormalized(const std::string& filename);