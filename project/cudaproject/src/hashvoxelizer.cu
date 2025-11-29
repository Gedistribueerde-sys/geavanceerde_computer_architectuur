#include "voxelizer.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <algorithm>
#include <chrono> // added for timing


#include <algorithm>

/*
The morton approach for the encoding can be found here:
https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
*/
// functions are used in both the voxelizers
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

//https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf





// the hashmap entry structure
struct __align__(16) HashBucket { // 16 forces that the memory starts on 16 byte alligned addresses
    unsigned long long key; // morton code is the unique key
    float sumX, sumY, sumZ;
    uint32_t sumR, sumG, sumB;
    uint32_t count;
};

// to prove that a slot is empty
// 0xFFFFFFFFFFFFFFFF almost no morton code's have 111111...
#define EMPTY_KEY 0xFFFFFFFFFFFFFFFFULL

// Kernel 1: initialise the map as empty
__global__ void initHashMapKernel(HashBucket* table, size_t capacity) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < capacity) {
        table[idx].key = EMPTY_KEY;
        table[idx].count = 0; // empty buckets are not seen as real voxels
        //sums do not need to be zero , but its cleaner
        table[idx].sumX = 0.0f; table[idx].sumY = 0.0f; table[idx].sumZ = 0.0f;
        table[idx].sumR = 0;    table[idx].sumG = 0;    table[idx].sumB = 0;
    }
}

// Kernel 2: Insert Points into Hash Map
__global__ void populateHashMapKernel(
    const float* x, const float* y, const float* z,
    const uint8_t* r, const uint8_t* g, const uint8_t* b,
    HashBucket* table,
    size_t capacity,
    size_t numPoints,
    float minX, float minY, float minZ,
    float invVoxelSize)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // 1. calculate the grid an morton code
    uint32_t ix = (uint32_t)floorf((x[idx] - minX) * invVoxelSize);
    uint32_t iy = (uint32_t)floorf((y[idx] - minY) * invVoxelSize);
    uint32_t iz = (uint32_t)floorf((z[idx] - minZ) * invVoxelSize);

    uint64_t mortonCode = mortonEncode(ix, iy, iz);

    // 2. calc the start hash index
    size_t hashIdx = mortonCode % capacity;
    
    // 3. Linear Probing Loop
    // try to claim the slot , max capacity tryout
    for (size_t i = 0; i < capacity; i++) {
        size_t currentSlot = (hashIdx + i) % capacity;

        // look what is in the table
        unsigned long long oldKey = table[currentSlot].key;
        
        // if the slot is empty, claim it
        if (oldKey == EMPTY_KEY) {
            unsigned long long assumed = atomicCAS((unsigned long long*)&table[currentSlot].key, EMPTY_KEY, (unsigned long long)mortonCode);
            if (assumed == EMPTY_KEY) {
                // claimed succesfully , the old key is now the own key
                oldKey = mortonCode; 
            } else {
                //the slot is not empty , update the key
                oldKey = assumed;
            }
        }

        // if the key is valid
        if (oldKey == mortonCode) {
            //add data to the slot
            atomicAdd(&table[currentSlot].sumX, x[idx]);
            atomicAdd(&table[currentSlot].sumY, y[idx]);
            atomicAdd(&table[currentSlot].sumZ, z[idx]);
            
            // add colors to the slot
            atomicAdd(&table[currentSlot].sumR, (uint32_t)r[idx]);
            atomicAdd(&table[currentSlot].sumG, (uint32_t)g[idx]);
            atomicAdd(&table[currentSlot].sumB, (uint32_t)b[idx]);
            
            atomicAdd(&table[currentSlot].count, 1);
            return; // finished this point
        }
        
        // if oldKey != mortonCode and != EMPTY_KEY, there is a collison.
        // Go to the next iteration (linear probing)
    }
}

// Kernel 3: count the amount of valid voxels , for the output calculation
__global__ void countValidBucketsKernel(HashBucket* table, size_t capacity, uint32_t* counter) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < capacity) {
        if (table[idx].key != EMPTY_KEY) {
            // simple atomic counter.
            atomicAdd(counter, 1);
        }
    }
}

// Kernel 4: Convert HashBuckets into output Points
// We need an index to write them consecutively in the output array.
// This can be done using an atomic counter inside the loop.
__global__ void collectResultsKernel(HashBucket* table, size_t capacity, Point* output, uint32_t* globalCounter) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= capacity) return;

    HashBucket bucket = table[idx];
    
    if (bucket.key != EMPTY_KEY && bucket.count > 0) {
        //valid voxel -> atomic add to have a unique output index
    
        uint32_t outIdx = atomicAdd(globalCounter, 1);
        
        float c = (float)bucket.count;
        
        Point p;
        p.x = bucket.sumX / c;
        p.y = bucket.sumY / c;
        p.z = bucket.sumZ / c;
        p.r = (uint8_t)(bucket.sumR / bucket.count);
        p.g = (uint8_t)(bucket.sumG / bucket.count);
        p.b = (uint8_t)(bucket.sumB / bucket.count);
        
        output[outIdx] = p;
    }
}


std::vector<Point> voxelizeDynamicHashMap(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize,
    uint32_t mapCapacityFactor ,int blockSize)
{
    if (totalPoints == 0) 
        return std::vector<Point>();

    // 1. Set up hash table capacity
    // A factor of 2.0 is usually the minimum to reduce collisions for open addressing.
    // A factor of 3–4 is faster but uses more GPU memory.
    if (mapCapacityFactor < 2) mapCapacityFactor = 2;
    size_t hashCapacity = totalPoints * mapCapacityFactor;

    // 2. Compute bounds (needed to convert world coords to voxel grid coords)
    float minX = *std::min_element(hostPoints.x.begin(), hostPoints.x.end());
    float minY = *std::min_element(hostPoints.y.begin(), hostPoints.y.end());
    float minZ = *std::min_element(hostPoints.z.begin(), hostPoints.z.end());
    float invVoxelSize = 1.0f / voxelSize;

    // 3. Allocate device memory for input arrays
    float *d_x, *d_y, *d_z;
    uint8_t *d_r, *d_g, *d_b;

    cudaMalloc(&d_x, totalPoints * sizeof(float));
    cudaMalloc(&d_y, totalPoints * sizeof(float));
    cudaMalloc(&d_z, totalPoints * sizeof(float));
    cudaMalloc(&d_r, totalPoints * sizeof(uint8_t));
    cudaMalloc(&d_g, totalPoints * sizeof(uint8_t));
    cudaMalloc(&d_b, totalPoints * sizeof(uint8_t));

    cudaMemcpy(d_x, hostPoints.x.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, hostPoints.y.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, hostPoints.z.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, hostPoints.r.data(), totalPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, hostPoints.g.data(), totalPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hostPoints.b.data(), totalPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 4. Allocate the hash table on the GPU
    HashBucket* d_hashTable;
    cudaMalloc(&d_hashTable, hashCapacity * sizeof(HashBucket));

    // Compute CUDA launch sizes

    int numBlocksTable  = (hashCapacity  + blockSize - 1) / blockSize;
    int numBlocksPoints = (totalPoints   + blockSize - 1) / blockSize;

    // 5. Initialize hash table (set all keys to EMPTY)
    initHashMapKernel<<<numBlocksTable, blockSize>>>(d_hashTable, hashCapacity);
    cudaDeviceSynchronize();

    // 6. Insert all points into the GPU hash table
    populateHashMapKernel<<<numBlocksPoints, blockSize>>>(
        d_x, d_y, d_z, d_r, d_g, d_b,
        d_hashTable,
        hashCapacity,
        totalPoints,
        minX, minY, minZ,
        invVoxelSize
    );
    cudaDeviceSynchronize();

    // 7. Count number of unique occupied voxels
    uint32_t* d_voxelCount;
    cudaMalloc(&d_voxelCount, sizeof(uint32_t));
    cudaMemset(d_voxelCount, 0, sizeof(uint32_t));

    countValidBucketsKernel<<<numBlocksTable, blockSize>>>(d_hashTable, hashCapacity, d_voxelCount);
    cudaDeviceSynchronize();

    uint32_t numVoxels = 0;
    cudaMemcpy(&numVoxels, d_voxelCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 8. Allocate output point buffer on GPU
    Point* d_outputPoints;
    cudaMalloc(&d_outputPoints, numVoxels * sizeof(Point));

    // Reset counter for writing compact output
    cudaMemset(d_voxelCount, 0, sizeof(uint32_t));

    // Collect results into dense array
    collectResultsKernel<<<numBlocksTable, blockSize>>>(d_hashTable, hashCapacity, d_outputPoints, d_voxelCount);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    std::vector<Point> result(numVoxels);
    if (numVoxels > 0) {
        cudaMemcpy(result.data(), d_outputPoints, numVoxels * sizeof(Point), cudaMemcpyDeviceToHost);
    }

    // Cleanup
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_hashTable);
    cudaFree(d_voxelCount);
    cudaFree(d_outputPoints);

    return result;
}

std::vector<Point> voxelizeDynamicHashMap_timed(
    const PointCloudVecs& hostPoints,
    size_t totalPoints,
    float voxelSize,
    uint32_t mapCapacityFactor,
    int runs , int blockSize) // added runs parameter
{
    if (totalPoints == 0)
        return std::vector<Point>();
    if (runs < 1) runs = 1;

    std::vector<Point> result; // will hold the result from the last run
    double totalMs = 0.0;

    // Events for timing individual GPU segments
    cudaEvent_t sAlloc, eAlloc;
    cudaEvent_t sInit, eInit;
    cudaEvent_t sPopulate, ePopulate;
    cudaEvent_t sCount, eCount;
    cudaEvent_t sCollect, eCollect;
    cudaEvent_t sCopy, eCopy;

    cudaEventCreate(&sAlloc);    cudaEventCreate(&eAlloc);
    cudaEventCreate(&sInit);     cudaEventCreate(&eInit);
    cudaEventCreate(&sPopulate); cudaEventCreate(&ePopulate);
    cudaEventCreate(&sCount);    cudaEventCreate(&eCount);
    cudaEventCreate(&sCollect);  cudaEventCreate(&eCollect);
    cudaEventCreate(&sCopy);     cudaEventCreate(&eCopy);

    float totalAllocMs    = 0.0f;
    float totalInitMs     = 0.0f;
    float totalPopulateMs = 0.0f;
    float totalCountMs    = 0.0f;
    float totalCollectMs  = 0.0f;
    float totalCopyMs     = 0.0f;

    // 1. Set up hash table capacity and compute CUDA launch sizes (outside the loop as they are constant)
    if (mapCapacityFactor < 2) mapCapacityFactor = 2;
    size_t hashCapacity = totalPoints * mapCapacityFactor;

    
    int numBlocksTable  = (hashCapacity  + blockSize - 1) / blockSize;
    int numBlocksPoints = (totalPoints   + blockSize - 1) / blockSize;


    for (int run = 0; run < runs; ++run) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 2. Compute bounds (needed to convert world coords to voxel grid coords)
        // These can be done outside the timing or included in the overall time.
        // We'll leave them here for completeness of a single run.
        float minX = *std::min_element(hostPoints.x.begin(), hostPoints.x.end());
        float minY = *std::min_element(hostPoints.y.begin(), hostPoints.y.end());
        float minZ = *std::min_element(hostPoints.z.begin(), hostPoints.z.end());
        float invVoxelSize = 1.0f / voxelSize;

        // --- Start Timing for Alloc + HtoD Copy ---
        float *d_x, *d_y, *d_z;
        uint8_t *d_r, *d_g, *d_b;
        HashBucket* d_hashTable;
        uint32_t* d_voxelCount;
        Point* d_outputPoints;
        float ms = 0.0f;

        cudaEventRecord(sAlloc);
        // 3. Allocate device memory for input arrays
        cudaMalloc(&d_x, totalPoints * sizeof(float));
        cudaMalloc(&d_y, totalPoints * sizeof(float));
        cudaMalloc(&d_z, totalPoints * sizeof(float));
        cudaMalloc(&d_r, totalPoints * sizeof(uint8_t));
        cudaMalloc(&d_g, totalPoints * sizeof(uint8_t));
        cudaMalloc(&d_b, totalPoints * sizeof(uint8_t));

        cudaMemcpy(d_x, hostPoints.x.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, hostPoints.y.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, hostPoints.z.data(), totalPoints * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, hostPoints.r.data(), totalPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, hostPoints.g.data(), totalPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, hostPoints.b.data(), totalPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);

        // 4. Allocate the hash table and counter on the GPU
        cudaMalloc(&d_hashTable, hashCapacity * sizeof(HashBucket));
        cudaMalloc(&d_voxelCount, sizeof(uint32_t));

        cudaEventRecord(eAlloc);
        cudaEventSynchronize(eAlloc);
        cudaEventElapsedTime(&ms, sAlloc, eAlloc);
        totalAllocMs += ms;
        // --- End Timing for Alloc + HtoD Copy ---

        // --- Start Timing for Initialization ---
        cudaEventRecord(sInit);
        // 5. Initialize hash table (set all keys to EMPTY)
        initHashMapKernel<<<numBlocksTable, blockSize>>>(d_hashTable, hashCapacity);
        cudaDeviceSynchronize();
        cudaEventRecord(eInit);
        cudaEventSynchronize(eInit);
        cudaEventElapsedTime(&ms, sInit, eInit);
        totalInitMs += ms;
        // --- End Timing for Initialization ---

        // --- Start Timing for Population/Insertion ---
        cudaEventRecord(sPopulate);
        // 6. Insert all points into the GPU hash table
        populateHashMapKernel<<<numBlocksPoints, blockSize>>>(
            d_x, d_y, d_z, d_r, d_g, d_b,
            d_hashTable,
            hashCapacity,
            totalPoints,
            minX, minY, minZ,
            invVoxelSize
        );
        cudaDeviceSynchronize();
        cudaEventRecord(ePopulate);
        cudaEventSynchronize(ePopulate);
        cudaEventElapsedTime(&ms, sPopulate, ePopulate);
        totalPopulateMs += ms;
        // --- End Timing for Population/Insertion ---

        // --- Start Timing for Counting ---
        cudaEventRecord(sCount);
        // 7. Count number of unique occupied voxels
        cudaMemset(d_voxelCount, 0, sizeof(uint32_t)); // Reset count to 0
        countValidBucketsKernel<<<numBlocksTable, blockSize>>>(d_hashTable, hashCapacity, d_voxelCount);
        cudaDeviceSynchronize();
        cudaEventRecord(eCount);
        cudaEventSynchronize(eCount);
        cudaEventElapsedTime(&ms, sCount, eCount);
        totalCountMs += ms;
        // --- End Timing for Counting ---

        // Get the result size (synchronizes implicitly via Memcpy)
        uint32_t numVoxels = 0;
        cudaMemcpy(&numVoxels, d_voxelCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // 8. Allocate output point buffer on GPU
        cudaMalloc(&d_outputPoints, numVoxels * sizeof(Point));
        // Reset counter for writing compact output
        cudaMemset(d_voxelCount, 0, sizeof(uint32_t)); // Reset count to 0 for collection index

        // --- Start Timing for Collection ---
        cudaEventRecord(sCollect);
        // Collect results into dense array
        collectResultsKernel<<<numBlocksTable, blockSize>>>(d_hashTable, hashCapacity, d_outputPoints, d_voxelCount);
        cudaDeviceSynchronize();
        cudaEventRecord(eCollect);
        cudaEventSynchronize(eCollect);
        cudaEventElapsedTime(&ms, sCollect, eCollect);
        totalCollectMs += ms;
        // --- End Timing for Collection ---

        // --- Start Timing for DtoH Copy and Cleanup ---
        cudaEventRecord(sCopy);
        // Copy results back to CPU
        result.assign(numVoxels, Point());
        if (numVoxels > 0) {
            cudaMemcpy(result.data(), d_outputPoints, numVoxels * sizeof(Point), cudaMemcpyDeviceToHost);
        }

        // Cleanup
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);
        cudaFree(d_hashTable);
        cudaFree(d_voxelCount);
        cudaFree(d_outputPoints);
        cudaEventRecord(eCopy);
        cudaEventSynchronize(eCopy);
        cudaEventElapsedTime(&ms, sCopy, eCopy);
        totalCopyMs += ms;
        // --- End Timing for DtoH Copy and Cleanup ---

        auto t1 = std::chrono::high_resolution_clock::now();
        double runMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        totalMs += runMs;
    }

    // Destroy events
    cudaEventDestroy(sAlloc);    cudaEventDestroy(eAlloc);
    cudaEventDestroy(sInit);     cudaEventDestroy(eInit);
    cudaEventDestroy(sPopulate); cudaEventDestroy(ePopulate);
    cudaEventDestroy(sCount);    cudaEventDestroy(eCount);
    cudaEventDestroy(sCollect);  cudaEventDestroy(eCollect);
    cudaEventDestroy(sCopy);     cudaEventDestroy(eCopy);

    // Print averages (per-segment)
    std::cout << "voxelizeDynamicHashMap_timed overall average over " << runs << " runs: "
              << (totalMs / runs) << " ms\n";
    std::cout << "  Allocation + HtoD Copy avg: " << (totalAllocMs    / runs) << " ms\n";
    std::cout << "  initHashMapKernel avg:      " << (totalInitMs     / runs) << " ms\n";
    std::cout << "  populateHashMapKernel avg:  " << (totalPopulateMs / runs) << " ms\n";
    std::cout << "  countValidBucketsKernel avg: " << (totalCountMs    / runs) << " ms\n";
    std::cout << "  collectResultsKernel avg:   " << (totalCollectMs  / runs) << " ms\n";
    std::cout << "  DtoH Copy + Cleanup avg:    " << (totalCopyMs     / runs) << " ms\n";

    return result;
}