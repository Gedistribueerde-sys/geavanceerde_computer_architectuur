#include "voxelizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>

// Simple hash function for voxel coordinates
struct VoxelKey {
    int ix, iy, iz;
    
    bool operator<(const VoxelKey& other) const {
        if (ix != other.ix) return ix < other.ix;
        if (iy != other.iy) return iy < other.iy;
        return iz < other.iz;
    }
};

// Kernel to assign points to voxels and accumulate properties
__global__ void voxelizeKernel(
    const Point* points,
    int numPoints,
    float voxelSize,
    int* voxelIndices,      // Maps point to voxel ID
    float* voxelX,          // Accumulated X position
    float* voxelY,          // Accumulated Y position
    float* voxelZ,          // Accumulated Z position
    float* voxelR,          // Accumulated Red
    float* voxelG,          // Accumulated Green
    float* voxelB,          // Accumulated Blue
    int* voxelCount         // Point count per voxel
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid >= numPoints) return;
    
    Point p = points[tid];
    
    // Calculate voxel grid indices (integer indices)
    int ix = static_cast<int>(floorf(p.x / voxelSize));
    int iy = static_cast<int>(floorf(p.y / voxelSize));
    int iz = static_cast<int>(floorf(p.z / voxelSize));
    
    // Create unique voxel ID using hash
    // Simple hash: (ix + offset) * prime1 ^ (iy + offset) * prime2 ^ (iz + offset) * prime3
    int offset = 1000;
    int voxelID = ((ix + offset) * 73856093) ^ ((iy + offset) * 19349663) ^ ((iz + offset) * 83492791);
    voxelID = abs(voxelID) % 1000000;  // Keep ID in reasonable range
    
    // Store voxel assignment for this point
    voxelIndices[tid] = voxelID;
    
    // Atomically accumulate point properties into voxel
    atomicAdd(&voxelX[tid], p.x);
    atomicAdd(&voxelY[tid], p.y);
    atomicAdd(&voxelZ[tid], p.z);
    atomicAdd(&voxelR[tid], static_cast<float>(p.r));
    atomicAdd(&voxelG[tid], static_cast<float>(p.g));
    atomicAdd(&voxelB[tid], static_cast<float>(p.b));
    atomicAdd(&voxelCount[tid], 1);
}

// Host function to launch kernel and process results
void voxelizeOnGPU(
    const std::vector<Point>& hostPoints,
    float voxelSize,
    std::vector<Voxel>& outputVoxels
) {
    int numPoints = hostPoints.size();
    
    if (numPoints == 0) {
        std::cerr << "No points to voxelize\n";
        return;
    }
    
    std::cout << "Starting GPU voxelization of " << numPoints << " points\n";
    std::cout << "Voxel size: " << voxelSize << "\n";
    
    // Allocate GPU memory for input points
    Point* d_points;
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMemcpy(d_points, hostPoints.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    
    // Allocate GPU memory for accumulation
    int* d_voxelIndices;
    float* d_voxelX;
    float* d_voxelY;
    float* d_voxelZ;
    float* d_voxelR;
    float* d_voxelG;
    float* d_voxelB;
    int* d_voxelCount;
    
    cudaMalloc(&d_voxelIndices, numPoints * sizeof(int));
    cudaMalloc(&d_voxelX, numPoints * sizeof(float));
    cudaMalloc(&d_voxelY, numPoints * sizeof(float));
    cudaMalloc(&d_voxelZ, numPoints * sizeof(float));
    cudaMalloc(&d_voxelR, numPoints * sizeof(float));
    cudaMalloc(&d_voxelG, numPoints * sizeof(float));
    cudaMalloc(&d_voxelB, numPoints * sizeof(float));
    cudaMalloc(&d_voxelCount, numPoints * sizeof(int));
    
    // Initialize arrays to 0
    cudaMemset(d_voxelX, 0, numPoints * sizeof(float));
    cudaMemset(d_voxelY, 0, numPoints * sizeof(float));
    cudaMemset(d_voxelZ, 0, numPoints * sizeof(float));
    cudaMemset(d_voxelR, 0, numPoints * sizeof(float));
    cudaMemset(d_voxelG, 0, numPoints * sizeof(float));
    cudaMemset(d_voxelB, 0, numPoints * sizeof(float));
    cudaMemset(d_voxelCount, 0, numPoints * sizeof(int));
    
    // Configure kernel
    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    
    std::cout << "Launching kernel with " << gridSize << " blocks of " 
              << blockSize << " threads\n";
    
    // Launch kernel
    voxelizeKernel<<<gridSize, blockSize>>>(
        d_points,
        numPoints,
        voxelSize,
        d_voxelIndices,
        d_voxelX,
        d_voxelY,
        d_voxelZ,
        d_voxelR,
        d_voxelG,
        d_voxelB,
        d_voxelCount
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << "\n";
        return;
    }
    
    // Synchronize
    cudaDeviceSynchronize();
    
    // Copy results back to CPU
    std::vector<int> h_voxelIndices(numPoints);
    std::vector<float> h_voxelX(numPoints);
    std::vector<float> h_voxelY(numPoints);
    std::vector<float> h_voxelZ(numPoints);
    std::vector<float> h_voxelR(numPoints);
    std::vector<float> h_voxelG(numPoints);
    std::vector<float> h_voxelB(numPoints);
    std::vector<int> h_voxelCount(numPoints);
    
    cudaMemcpy(h_voxelIndices.data(), d_voxelIndices, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxelX.data(), d_voxelX, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxelY.data(), d_voxelY, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxelZ.data(), d_voxelZ, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxelR.data(), d_voxelR, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxelG.data(), d_voxelG, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxelB.data(), d_voxelB, numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxelCount.data(), d_voxelCount, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Build unique voxels from accumulated data
    std::map<int, Voxel> voxelMap;
    
    for (int i = 0; i < numPoints; ++i) {
        if (h_voxelCount[i] > 0) {
            int voxelID = h_voxelIndices[i];
            
            if (voxelMap.find(voxelID) == voxelMap.end()) {
                Voxel v;
                v.x = h_voxelX[i] / h_voxelCount[i];
                v.y = h_voxelY[i] / h_voxelCount[i];
                v.z = h_voxelZ[i] / h_voxelCount[i];
                v.r = static_cast<uint8_t>(h_voxelR[i] / h_voxelCount[i]);
                v.g = static_cast<uint8_t>(h_voxelG[i] / h_voxelCount[i]);
                v.b = static_cast<uint8_t>(h_voxelB[i] / h_voxelCount[i]);
                v.pointCount = h_voxelCount[i];
                
                voxelMap[voxelID] = v;
            }
        }
    }
    
    // Convert map to vector
    outputVoxels.clear();
    for (auto& pair : voxelMap) {
        outputVoxels.push_back(pair.second);
    }
    
    std::cout << "Created " << outputVoxels.size() << " unique voxels\n";
    
    // Free GPU memory
    cudaFree(d_points);
    cudaFree(d_voxelIndices);
    cudaFree(d_voxelX);
    cudaFree(d_voxelY);
    cudaFree(d_voxelZ);
    cudaFree(d_voxelR);
    cudaFree(d_voxelG);
    cudaFree(d_voxelB);
    cudaFree(d_voxelCount);
}