#include "las_reader.h"
#include "voxelizer.h"
#include "ply_writer.h"

#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
using namespace std;
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input.las> [voxel_size]\n";
        return 1;
    }
    
    string inputFile = argv[1];
    float voxelSize = (argc >= 3) ? stof(argv[2]) : 0.5f;
    
    // Read the LAS file
    PointCloudVecs pcVecs = readLASFileNormalized(inputFile);
    size_t totalPoints = pcVecs.x.size();   
    cout << "Total points read: " << totalPoints << "\n";
    cout << "Voxel size: " << voxelSize << "\n\n";

    // GPU Warmup (initializes CUDA context, don't time this)
    cout << "Warming up GPU..." << endl;
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
    cout << "GPU warmed up.\n\n";

    // CPU Voxelization
    cout << "=== CPU Voxelization ===" << endl;
    auto cpuStart = chrono::high_resolution_clock::now();
    vector<Point> voxelizedCPU = voxerlizerMortonOnCPU(pcVecs, totalPoints, voxelSize);
    auto cpuEnd = chrono::high_resolution_clock::now();
    double cpuTimeMs = chrono::duration<double, milli>(cpuEnd - cpuStart).count();
    cout << "CPU voxels: " << voxelizedCPU.size() << endl;
    cout << "CPU time: " << fixed << setprecision(2) << cpuTimeMs << " ms\n\n";



    // GPU Voxelization (Morton code based)
    cout << "=== GPU Voxelization (Morton) ===" << endl;
    auto gpuStart = chrono::high_resolution_clock::now();
    vector<Point> voxelizedGPU = voxelizeMortonOnGPU(pcVecs, totalPoints, voxelSize, 256); // 256 = block size
    auto gpuEnd = chrono::high_resolution_clock::now();
    double gpuTimeMs = chrono::duration<double, milli>(gpuEnd - gpuStart).count();
    cout << "GPU voxels: " << voxelizedGPU.size() << endl;
    cout << "GPU time: " << fixed << setprecision(2) << gpuTimeMs << " ms\n\n";




    // Summary
    cout << "=== Summary (Morton)===" << endl;
    cout << "Input points:      " << totalPoints << endl;
    cout << "Voxel size:        " << fixed << setprecision(2) << voxelSize << endl;
    cout << "CPU output voxels: " << voxelizedCPU.size() << endl;
    cout << "GPU output voxels: " << voxelizedGPU.size() << endl;
    cout << "Compression ratio: " << setprecision(2) 
              << (float)totalPoints / voxelizedCPU.size() << "x" << endl;
    cout << "\nTiming:" << endl;
    cout << "  CPU: " << cpuTimeMs << " ms" << endl;
    cout << "  GPU: " << gpuTimeMs << " ms" << endl;
    cout << "  Speedup: " << setprecision(2) << cpuTimeMs / gpuTimeMs << "x\n\n";




    // Sample output comparison
    cout << "=== Sample Points (first 5) ===" << endl;
    cout << "Original:" << endl;
    for (size_t i = 0; i < min(totalPoints, size_t(5)); i++) {
        cout << "  [" << i << "] x=" << setprecision(2) << pcVecs.x[i] 
                  << " y=" << pcVecs.y[i] << " z=" << pcVecs.z[i]
                  << " rgb=(" << (int)pcVecs.r[i] << "," << (int)pcVecs.g[i] 
                  << "," << (int)pcVecs.b[i] << ")" << endl;
    }
    cout << "CPU voxelized:" << endl;
    for (size_t i = 0; i < min(voxelizedCPU.size(), size_t(5)); i++) {
        cout << "  [" << i << "] x=" << voxelizedCPU[i].x 
                  << " y=" << voxelizedCPU[i].y << " z=" << voxelizedCPU[i].z
                  << " rgb=(" << (int)voxelizedCPU[i].r << "," << (int)voxelizedCPU[i].g 
                  << "," << (int)voxelizedCPU[i].b << ")" << endl;
    }
    cout << "GPU voxelized:" << endl;
    for (size_t i = 0; i < min(voxelizedGPU.size(), size_t(5)); i++) {
        cout << "  [" << i << "] x=" << voxelizedGPU[i].x 
                  << " y=" << voxelizedGPU[i].y << " z=" << voxelizedGPU[i].z
                  << " rgb=(" << (int)voxelizedGPU[i].r << "," << (int)voxelizedGPU[i].g 
                  << "," << (int)voxelizedGPU[i].b << ")" << endl;
    }

 




    // Bounds check
    cout << "\n=== Bounds ===" << endl;
    float maxX = *max_element(pcVecs.x.begin(), pcVecs.x.end());
    float maxY = *max_element(pcVecs.y.begin(), pcVecs.y.end());
    float maxZ = *max_element(pcVecs.z.begin(), pcVecs.z.end());
    cout << "Original: X[0, " << maxX << "], Y[0, " << maxY << "], Z[0, " << maxZ << "]\n";

    // Write output files
    writePLY("original.ply", pcVecs);
    writePLY("voxelized_cpu.ply", voxelizedCPU);
    writePLY("voxelized_gpu.ply", voxelizedGPU);
    
    cout << "\nOutput files written: original.ply, voxelized_cpu.ply, voxelized_gpu.ply\n";

//==========================================================================================

//==========================================================================================

//==========================================================================================

      cout <<"hashtable"<<endl<< "Warming up GPU..." << endl;
    {
        // Small dummy run to initialize CUDA
        PointCloudVecs dummy;
        dummy.x = {0, 1, 2};
        dummy.y = {0, 1, 2};
        dummy.z = {0, 1, 2};
        dummy.r = {0, 0, 0};
        dummy.g = {0, 0, 0};
        dummy.b = {0, 0, 0};
        voxelizeDynamicHashMap(dummy, 3, 1.0f, 2 , 256);
    }
    cout << "GPU warmed up.\n\n";
    uint32_t hashfactor =2; 
    cout << "=== GPU Voxelization (Hash)==factor = " << hashfactor << endl;
    gpuStart = chrono::high_resolution_clock::now();
    vector<Point> voxelizedGPU2 = voxelizeDynamicHashMap(pcVecs, totalPoints, voxelSize,hashfactor,256);
    gpuEnd = chrono::high_resolution_clock::now();
    gpuTimeMs = chrono::duration<double, milli>(gpuEnd - gpuStart).count();
    cout << "GPU voxels: " << voxelizedGPU2.size() << endl;
    cout << "GPU time: " << fixed << setprecision(2) << gpuTimeMs << " ms\n\n";

   
    cout << "=== Summary (Hash)===" << endl;
    cout << "Input points:      " << totalPoints << endl;
    cout << "Voxel size:        " << fixed << setprecision(2) << voxelSize << endl;
    cout << "CPU output voxels: " << voxelizedCPU.size() << endl;
    cout << "GPU output voxels: " << voxelizedGPU2.size() << endl;
    cout << "Compression ratio: " << setprecision(2) 
              << (float)totalPoints / voxelizedCPU.size() << "x" << endl;
    cout << "\nTiming:" << endl;
    cout << "  CPU: " << cpuTimeMs << " ms" << endl;
    cout << "  GPU: " << gpuTimeMs << " ms" << endl;
    cout << "  Speedup: " << setprecision(2) << cpuTimeMs / gpuTimeMs << "x\n\n";

//==========================================================================================

//==========================================================================================

//==========================================================================================
    int nruns =100;
    for (int i = 1; i <= 1024; i *= 2){
        cout << "=== GPU Voxelization (Morton) === Block size: " << i << endl;
        vector<Point> voxelizedGPU = voxelizeMortonOnGPU_timed(pcVecs, totalPoints, voxelSize, i, nruns); 
    }


    for (uint32_t factor = 2; factor <= 4; ++factor) {
        for (int i = 1; i <= 1024; i *= 2){
        cout << "=== GPU Voxelization (Hash) === Block size: " << i << endl;
        vector<Point> voxelizedHash = voxelizeDynamicHashMap_timed(pcVecs,totalPoints,voxelSize,factor,nruns,i);
        }
    }



    return 0;
}