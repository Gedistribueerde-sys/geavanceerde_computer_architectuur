#include "las_reader.h"
#include "voxelizer.h"
#include "ply_writer.h"
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.las> [voxel_size]\n";
        return 1;
    }
    
    std::string inputFile = argv[1];
    float voxelSize = (argc >= 3) ? std::stof(argv[2]) : 0.5f;
    
    // Read the LAS file
    PointCloudVecs pcVecs = readLASFileNormalized(inputFile);
    size_t totalPoints = pcVecs.x.size();   
    std::cout << "Total points read: " << totalPoints << "\n";
    std::cout << "Voxel size: " << voxelSize << "\n\n";

    // GPU Warmup (initializes CUDA context, don't time this)
    std::cout << "Warming up GPU..." << std::endl;
    {
        // Small dummy run to initialize CUDA
        PointCloudVecs dummy;
        dummy.x = {0, 1, 2};
        dummy.y = {0, 1, 2};
        dummy.z = {0, 1, 2};
        dummy.r = {0, 0, 0};
        dummy.g = {0, 0, 0};
        dummy.b = {0, 0, 0};
        voxelizeMortonOnGPU(dummy, 3, 1.0f ,256);
    }
    std::cout << "GPU warmed up.\n\n";

    // CPU Voxelization
    std::cout << "=== CPU Voxelization ===" << std::endl;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<Point> voxelizedCPU = voxerlizerMortonOnCPU(pcVecs, totalPoints, voxelSize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    std::cout << "CPU voxels: " << voxelizedCPU.size() << std::endl;
    std::cout << "CPU time: " << std::fixed << std::setprecision(2) << cpuTimeMs << " ms\n\n";



    // GPU Voxelization (Morton code based)
    std::cout << "=== GPU Voxelization (Morton) ===" << std::endl;
    auto gpuStart = std::chrono::high_resolution_clock::now();
    std::vector<Point> voxelizedGPU = voxelizeMortonOnGPU(pcVecs, totalPoints, voxelSize, 256); // 256 = block size
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    double gpuTimeMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
    std::cout << "GPU voxels: " << voxelizedGPU.size() << std::endl;
    std::cout << "GPU time: " << std::fixed << std::setprecision(2) << gpuTimeMs << " ms\n\n";

    // Summary
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Input points:      " << totalPoints << std::endl;
    std::cout << "Voxel size:        " << std::fixed << std::setprecision(2) << voxelSize << std::endl;
    std::cout << "CPU output voxels: " << voxelizedCPU.size() << std::endl;
    std::cout << "GPU output voxels: " << voxelizedGPU.size() << std::endl;
    std::cout << "Compression ratio: " << std::setprecision(2) 
              << (float)totalPoints / voxelizedCPU.size() << "x" << std::endl;
    std::cout << "\nTiming:" << std::endl;
    std::cout << "  CPU: " << cpuTimeMs << " ms" << std::endl;
    std::cout << "  GPU: " << gpuTimeMs << " ms" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << cpuTimeMs / gpuTimeMs << "x\n\n";

    for (int i = 1; i <= 1024; i *= 2){
        std::cout << "=== GPU Voxelization (Morton) === Block size: " << i << std::endl;
        auto gpuStart = std::chrono::high_resolution_clock::now();
        std::vector<Point> voxelizedGPU = voxelizeMortonOnGPU_timed(pcVecs, totalPoints, voxelSize, i, 100); // 256 = block size
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        double gpuTimeMs = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
        
    }

    // Sample output comparison
    std::cout << "=== Sample Points (first 5) ===" << std::endl;
    std::cout << "Original:" << std::endl;
    for (size_t i = 0; i < std::min(totalPoints, size_t(5)); i++) {
        std::cout << "  [" << i << "] x=" << std::setprecision(2) << pcVecs.x[i] 
                  << " y=" << pcVecs.y[i] << " z=" << pcVecs.z[i]
                  << " rgb=(" << (int)pcVecs.r[i] << "," << (int)pcVecs.g[i] 
                  << "," << (int)pcVecs.b[i] << ")" << std::endl;
    }
    std::cout << "CPU voxelized:" << std::endl;
    for (size_t i = 0; i < std::min(voxelizedCPU.size(), size_t(5)); i++) {
        std::cout << "  [" << i << "] x=" << voxelizedCPU[i].x 
                  << " y=" << voxelizedCPU[i].y << " z=" << voxelizedCPU[i].z
                  << " rgb=(" << (int)voxelizedCPU[i].r << "," << (int)voxelizedCPU[i].g 
                  << "," << (int)voxelizedCPU[i].b << ")" << std::endl;
    }
    std::cout << "GPU voxelized:" << std::endl;
    for (size_t i = 0; i < std::min(voxelizedGPU.size(), size_t(5)); i++) {
        std::cout << "  [" << i << "] x=" << voxelizedGPU[i].x 
                  << " y=" << voxelizedGPU[i].y << " z=" << voxelizedGPU[i].z
                  << " rgb=(" << (int)voxelizedGPU[i].r << "," << (int)voxelizedGPU[i].g 
                  << "," << (int)voxelizedGPU[i].b << ")" << std::endl;
    }

    // Bounds check
    std::cout << "\n=== Bounds ===" << std::endl;
    float maxX = *std::max_element(pcVecs.x.begin(), pcVecs.x.end());
    float maxY = *std::max_element(pcVecs.y.begin(), pcVecs.y.end());
    float maxZ = *std::max_element(pcVecs.z.begin(), pcVecs.z.end());
    std::cout << "Original: X[0, " << maxX << "], Y[0, " << maxY << "], Z[0, " << maxZ << "]\n";

    // Write output files
    writePLY("original.ply", pcVecs);
    writePLY("voxelized_cpu.ply", voxelizedCPU);
    writePLY("voxelized_gpu.ply", voxelizedGPU);
    
    std::cout << "\nOutput files written: original.ply, voxelized_cpu.ply, voxelized_gpu.ply\n";

    return 0;
}