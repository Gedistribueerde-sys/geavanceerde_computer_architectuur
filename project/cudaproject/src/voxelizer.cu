#include "voxelizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <algorithm>

// Thrust includes for Morton code sorting approach
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

/*
The morton approach for the encoding can be found here:
https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
*/
__device__ __host__ inline uint64_t splitBy3(uint64_t v) {
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v <<  8)) & 0x100f00f00f00f00fULL;
    v = (v | (v <<  4)) & 0x10c30c30c30c30c3ULL;
    v = (v | (v <<  2)) & 0x1249249249249249ULL;
    return v;
}

__device__ __host__ inline uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) {
    return (splitBy3((uint64_t)x) << 2) | 
           (splitBy3((uint64_t)y) << 1) | 
            splitBy3((uint64_t)z);
}

// Struct to hold point data for reduction
struct PointAccum {
    float sumX, sumY, sumZ;
    uint32_t sumR, sumG, sumB;
    uint32_t count;
    
    __device__ __host__ PointAccum() 
        : sumX(0), sumY(0), sumZ(0), sumR(0), sumG(0), sumB(0), count(0) {}
    
    __device__ __host__ PointAccum(float x, float y, float z, uint8_t r, uint8_t g, uint8_t b)
        : sumX(x), sumY(y), sumZ(z), sumR(r), sumG(g), sumB(b), count(1) {}
};

// Binary operator for reducing PointAccum
struct PointAccumOp {
    __device__ __host__ PointAccum operator()(const PointAccum& a, const PointAccum& b) const {
        PointAccum result;
        result.sumX = a.sumX + b.sumX;
        result.sumY = a.sumY + b.sumY;
        result.sumZ = a.sumZ + b.sumZ;
        result.sumR = a.sumR + b.sumR;
        result.sumG = a.sumG + b.sumG;
        result.sumB = a.sumB + b.sumB;
        result.count = a.count + b.count;
        return result;
    }
};

// Kernel: compute Morton code for each point
__global__ void computeMortonCodesKernel(
    const float* x,
    const float* y, 
    const float* z,
    uint64_t* mortonCodes,
    float minX, float minY, float minZ,
    float invVoxelSize,
    size_t numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    // Compute voxel indices (offset by min to ensure positive)
    uint32_t ix = (uint32_t)floorf((x[idx] - minX) * invVoxelSize);
    uint32_t iy = (uint32_t)floorf((y[idx] - minY) * invVoxelSize);
    uint32_t iz = (uint32_t)floorf((z[idx] - minZ) * invVoxelSize);
    
    mortonCodes[idx] = mortonEncode(ix, iy, iz);
}

// Kernel: create PointAccum from point data (after sorting)
__global__ void createPointAccumKernel(
    const float* x,
    const float* y,
    const float* z,
    const uint8_t* r,
    const uint8_t* g,
    const uint8_t* b,
    const uint32_t* indices,
    PointAccum* accums,
    size_t numPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    
    uint32_t origIdx = indices[idx];
    accums[idx] = PointAccum(x[origIdx], y[origIdx], z[origIdx],
                              r[origIdx], g[origIdx], b[origIdx]);
}

std::vector<Point> voxelizeMortonOnGPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize)
{
    if (totalPoints == 0) {
        return std::vector<Point>();
    }

    // Find bounds on CPU (could also do on GPU with Thrust)
    float minX = *std::min_element(hostPoints.x.begin(), hostPoints.x.end());
    float minY = *std::min_element(hostPoints.y.begin(), hostPoints.y.end());
    float minZ = *std::min_element(hostPoints.z.begin(), hostPoints.z.end());
    float invVoxelSize = 1.0f / voxelSize;

    // Allocate device memory for input points
    thrust::device_vector<float> d_x(hostPoints.x.begin(), hostPoints.x.end());
    thrust::device_vector<float> d_y(hostPoints.y.begin(), hostPoints.y.end());
    thrust::device_vector<float> d_z(hostPoints.z.begin(), hostPoints.z.end());
    thrust::device_vector<uint8_t> d_r(hostPoints.r.begin(), hostPoints.r.end());
    thrust::device_vector<uint8_t> d_g(hostPoints.g.begin(), hostPoints.g.end());
    thrust::device_vector<uint8_t> d_b(hostPoints.b.begin(), hostPoints.b.end());

    // Step 1: Compute Morton codes
    thrust::device_vector<uint64_t> d_mortonCodes(totalPoints);
    thrust::device_vector<uint32_t> d_indices(totalPoints);
    
    // Initialize indices to 0, 1, 2, ...
    thrust::sequence(d_indices.begin(), d_indices.end());
    
    int blockSize = 256;
    int numBlocks = (totalPoints + blockSize - 1) / blockSize;
    
    computeMortonCodesKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_z.data()),
        thrust::raw_pointer_cast(d_mortonCodes.data()),
        minX, minY, minZ,
        invVoxelSize,
        totalPoints
    );
    cudaDeviceSynchronize();

    // Step 2: Sort indices by Morton code
    thrust::sort_by_key(d_mortonCodes.begin(), d_mortonCodes.end(), d_indices.begin());

    // Step 3: Create PointAccum for each point (reordered by Morton code)
    thrust::device_vector<PointAccum> d_accums(totalPoints);
    
    createPointAccumKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_z.data()),
        thrust::raw_pointer_cast(d_r.data()),
        thrust::raw_pointer_cast(d_g.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        thrust::raw_pointer_cast(d_accums.data()),
        totalPoints
    );
    cudaDeviceSynchronize();

    // Step 4: Reduce by key (Morton code) to accumulate points per voxel
    thrust::device_vector<uint64_t> d_uniqueMorton(totalPoints);
    thrust::device_vector<PointAccum> d_reducedAccums(totalPoints);
    
    auto reduceEnd = thrust::reduce_by_key(
        d_mortonCodes.begin(), d_mortonCodes.end(),
        d_accums.begin(),
        d_uniqueMorton.begin(),
        d_reducedAccums.begin(),
        thrust::equal_to<uint64_t>(),
        PointAccumOp()
    );
    
    size_t numVoxels = reduceEnd.first - d_uniqueMorton.begin();

    // Step 5: Copy reduced accumulators back to host and compute averages
    std::vector<PointAccum> h_accums(numVoxels);
    thrust::copy(d_reducedAccums.begin(), d_reducedAccums.begin() + numVoxels, h_accums.begin());

    // Convert accumulators to final points
    std::vector<Point> result(numVoxels);
    for (size_t i = 0; i < numVoxels; i++) {
        const PointAccum& acc = h_accums[i];
        float c = static_cast<float>(acc.count);
        result[i].x = acc.sumX / c;
        result[i].y = acc.sumY / c;
        result[i].z = acc.sumZ / c;
        result[i].r = static_cast<uint8_t>(acc.sumR / acc.count);
        result[i].g = static_cast<uint8_t>(acc.sumG / acc.count);
        result[i].b = static_cast<uint8_t>(acc.sumB / acc.count);
    }

    return result;
}

std::vector<Point> voxelizeUniformOnCPU(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize)
{
    if (totalPoints == 0){
        std::cerr << "No points to voxelize\n";
        return std::vector<Point>();
    }

    // generate a map with keys and an accumalator
    // this is for easy accumalating points that belong to same voxel
    std::map<VoxelKey, VoxelAccumulator> voxelMap;

    for (size_t i = 0; i < totalPoints; i++) {
        int ix = static_cast<int>(floor(hostPoints.x[i] / voxelSize));
        int iy = static_cast<int>(floor(hostPoints.y[i] / voxelSize));
        int iz = static_cast<int>(floor(hostPoints.z[i] / voxelSize));
        VoxelKey key = VoxelKey(ix, iy, iz);

        VoxelAccumulator& acc = voxelMap[key];
        acc.sumX += hostPoints.x[i];
        acc.sumY += hostPoints.y[i];
        acc.sumZ += hostPoints.z[i];
        acc.sumR += static_cast<uint32_t>(hostPoints.r[i]);
        acc.sumG += static_cast<uint32_t>(hostPoints.g[i]);
        acc.sumB += static_cast<uint32_t>(hostPoints.b[i]);
        acc.count++;
        
    }

    // transform the accumulated points to Point
    std::vector<Point> voxelizedPoints;
    voxelizedPoints.reserve(voxelMap.size());

    for (const auto& [key, acc] : voxelMap) {
        Point p;
        p.x = static_cast<float>(std::max(0.0, acc.sumX / acc.count));
        p.y = static_cast<float>(std::max(0.0, acc.sumY / acc.count));
        p.z = static_cast<float>(std::max(0.0, acc.sumZ / acc.count));
        p.r = static_cast<uint8_t>(acc.sumR / acc.count);
        p.g = static_cast<uint8_t>(acc.sumG / acc.count);
        p.b = static_cast<uint8_t>(acc.sumB / acc.count);
        voxelizedPoints.push_back(p);
    }

    return voxelizedPoints;
}