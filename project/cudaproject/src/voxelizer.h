#pragma once
#include <vector>
#include "point.h"


// GPU voxelization function
std::vector<Point> voxelizeOnGPU(
    const std::vector<Point>& hostPoints,
    size_t totalPoints,
    float voxelSize
);

std::vector<Point> voxelizeOnCPU(
    const std::vector<Point>& hostPoints,
    size_t totalPoints,
    float voxelSize
);