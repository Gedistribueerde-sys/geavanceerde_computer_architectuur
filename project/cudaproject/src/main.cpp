#include "las_reader.h"
#include "voxelizer.h"
#include "ply_writer.h"
#include <algorithm>
#include <limits>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.las>\n";
        return 1;
    }
    
    // 1. Read the las file
    std::string inputFile = argv[1];
    PointCloudVecs pcVecs = readLASFileNormalized(inputFile);
    
    size_t totalPoints = pcVecs.x.size();   
    std::cout << "Total points read: " << totalPoints << "\n";

    std::vector<Point> voxelizedPoints = voxelizeUniformOnCPU(pcVecs, totalPoints, 0.5f);

    // print some results

    std::cout << "Number of points: " << totalPoints << std::endl;
    std::cout << "Number of voxelizedPoints: " << voxelizedPoints.size() << std::endl;
    std::cout << "Compression ratio: " << static_cast<float>(totalPoints) / voxelizedPoints.size() << "x" << std::endl;
    std::cout << "Some points: " << std::endl;
    for (size_t i = 0; i < std::min(totalPoints, size_t(10)); i++) {
        std::cout << "x: " << pcVecs.x[i] << "; y: " << pcVecs.y[i] << "; z: " << pcVecs.z[i] <<
        "; r: " << static_cast<int>(pcVecs.r[i]) << "; g: " << static_cast<int>(pcVecs.g[i]) << "; b: " << static_cast<int>(pcVecs.b[i]) << std::endl;
    }
    std::cout << "Some voxelizedPoints: " << std::endl;
    for (size_t i = 0; i < std::min(voxelizedPoints.size(), size_t(10)); i++){
        std::cout << "x: " << voxelizedPoints[i].x << "; y: " << voxelizedPoints[i].y << "; z: " << voxelizedPoints[i].z <<
        "; r: " << static_cast<int>(voxelizedPoints[i].r) << "; g: " << static_cast<int>(voxelizedPoints[i].g) << "; b: " << static_cast<int>(voxelizedPoints[i].b) << std::endl;
    }
    std::cout << "Bounds original point cloud: " << std::endl;
    float maxX = *std::max_element(pcVecs.x.begin(), pcVecs.x.end());
    float maxY = *std::max_element(pcVecs.y.begin(), pcVecs.y.end());
    float maxZ = *std::max_element(pcVecs.z.begin(), pcVecs.z.end());
    std::cout << "Data bounds: X[0, " << maxX << "], Y[0, " << maxY << "], Z[0, " << maxZ << "]\n";

    std::cout << "Bounds voxelized points: " << std::endl;
    float voxMaxX = 0, voxMaxY = 0, voxMaxZ = 0;
    float voxMinX = std::numeric_limits<float>::max();
    float voxMinY = std::numeric_limits<float>::max();
    float voxMinZ = std::numeric_limits<float>::max();
    for (const auto& p : voxelizedPoints) {
        voxMaxX = std::max(voxMaxX, p.x);
        voxMaxY = std::max(voxMaxY, p.y);
        voxMaxZ = std::max(voxMaxZ, p.z);
        voxMinX = std::min(voxMinX, p.x);
        voxMinY = std::min(voxMinY, p.y);
        voxMinZ = std::min(voxMinZ, p.z);
    }
    std::cout << "Data bounds: X[" << voxMinX << ", " << voxMaxX << "], Y[" << voxMinY << ", " << voxMaxY << "], Z[" << voxMinZ << ", " << voxMaxZ << "]\n";
    // Write output files
    writePLY("original.ply", pcVecs);
    writePLY("voxelized.ply", voxelizedPoints);
    return 0;
}