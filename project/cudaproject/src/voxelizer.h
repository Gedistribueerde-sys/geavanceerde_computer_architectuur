#pragma once
#include <vector>
#include "point.h"



std::vector<Point> voxelizeMortonOnGPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize,
    int blockSize
);
std::vector<Point> voxelizeMortonOnGPU_timed(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize,
    int blockSize,
    int runs);

std::vector<Point> voxelizeUniformOnCPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize
);

std::vector<Point> voxerlizerMortonOnCPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize
);