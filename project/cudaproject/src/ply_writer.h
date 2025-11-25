#pragma once
#include <string>
#include <vector>
#include "point.h"

// Write a vector of Points to a PLY file (ASCII format)
bool writePLY(const std::string& filename, const std::vector<Point>& points);

// Write PointCloudVecs to a PLY file (ASCII format)
bool writePLY(const std::string& filename, const PointCloudVecs& points);
