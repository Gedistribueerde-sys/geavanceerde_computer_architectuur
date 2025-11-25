#pragma once
#include <vector>
#include "point.h"



std::vector<Point> voxelizeMortonOnGPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize
);

std::vector<Point> voxelizeUniformOnCPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize
);