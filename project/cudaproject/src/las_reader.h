#pragma once
#include <pdal/StageFactory.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/Dimension.hpp>
#include <limits>
#include <vector>
#include <string>
#include <iostream>
#include "point.h"


PointCloudVecs readLASFileNormalized(const std::string& filename);
bool hasColorData(const pdal::PointTable& table);