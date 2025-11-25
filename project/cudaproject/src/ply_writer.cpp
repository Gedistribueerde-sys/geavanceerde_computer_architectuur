#include "ply_writer.h"
#include <fstream>
#include <iostream>

bool writePLY(const std::string& filename, const std::vector<Point>& points) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return false;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    // Write point data
    for (const auto& p : points) {
        file << p.x << " " << p.y << " " << p.z << " "
             << static_cast<int>(p.r) << " "
             << static_cast<int>(p.g) << " "
             << static_cast<int>(p.b) << "\n";
    }

    file.close();
    std::cout << "Successfully wrote " << points.size() << " points to " << filename << "\n";
    return true;
}

bool writePLY(const std::string& filename, const PointCloudVecs& points) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return false;
    }

    size_t numPoints = points.x.size();

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << numPoints << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    // Write point data
    for (size_t i = 0; i < numPoints; ++i) {
        file << points.x[i] << " " << points.y[i] << " " << points.z[i] << " "
             << static_cast<int>(points.r[i]) << " "
             << static_cast<int>(points.g[i]) << " "
             << static_cast<int>(points.b[i]) << "\n";
    }

    file.close();
    std::cout << "Successfully wrote " << numPoints << " points to " << filename << "\n";
    return true;
}
