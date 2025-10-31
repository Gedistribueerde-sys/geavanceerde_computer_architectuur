/*
Design a kernel that multiplies two matrices.
Assess the processing time for a kernel that utilizes global-only, global and
shared, and global and constant memory.
Evaluations should consider changing the grid and/or input size.

Pointers
define an array that represents a matrix - in 1D or 2D form (start off with small
square matrices and simple numbers for easier debugging).
In this way define/allocate memory for three matrices, which will serve for C= AB.
Make a kernel that uses only global memory to compute C→ verify the results.
Make a second and third matrix multiplication kernel, which use shared and constant
memory, respectively, in addition to global memory.
Time the kernels for different input and/or grid sizes

Questions

How much time does memory migration take (CPU to GPU global; GPU global to
GPU shared)?
How do the execution times of matrix multiplication compare when employing the
three different memory types?
How do the times scale with input and/or grid size?

*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memset
#include <time.h>   // For CPU timing
#include <cuda_runtime.h> // For CUDA functions
// #include <assert.h> // For assert - not strictly needed without explicit checks

// --- Configuration ---
#define BLOCK_SIZE_GLOBAL 16 // For global memory kernel, typical block size
#define BLOCK_SIZE_SHARED 16 // For shared memory kernel, block size for tiling
#define TILE_WIDTH BLOCK_SIZE_SHARED // Tile width for shared memory

// List of matrix sizes to test
// Ensure these are multiples of BLOCK_SIZE_SHARED for optimal shared memory performance,
// though the code will handle non-multiples with bounds checks.
const int MATRIX_SIZES[] = {
    6,     // Smallest for easy verification, will skip printing details
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048
};
const int NUM_MATRIX_SIZES = sizeof(MATRIX_SIZES) / sizeof(MATRIX_SIZES[0]);

// cpu functions

// Function to multiply two square matrices on CPU (C = A * B) - for verification
void multiplySquareMatricesCPU(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}
// Function to verify GPU results against CPU results
int verifyResult(int* h_C_cpu, int* h_C_gpu, int size) {
    int errors = 0;
    for (int i = 0; i < size * size; i++) {
        if (h_C_cpu[i] != h_C_gpu[i]) {
            // No error printing for verification, just count
            errors++;
        }
    }
    return errors;
}

// --- CUDA Kernels ---

// 1. CUDA Kernel: Matrix multiplication using only Global Memory
__global__ void matrixMultGlobalKernel(int* A, int* B, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

// 2. CUDA Kernel: Matrix multiplication using Shared Memory (tiling)
__global__ void matrixMultSharedKernel(int* A, int* B, int* C, int size) {
    __shared__ int s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int s_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int sum = 0;

    int numTiles = (size + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        int tileRowA = row;
        int tileColA = t * TILE_WIDTH + threadIdx.x;

        int tileRowB = t * TILE_WIDTH + threadIdx.y;
        int tileColB = col;

        if (tileRowA < size && tileColA < size) {
             s_A[threadIdx.y][threadIdx.x] = A[tileRowA * size + tileColA];
        } else {
             s_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (tileRowB < size && tileColB < size) {
            s_B[threadIdx.y][threadIdx.x] = B[tileRowB * size + tileColB];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < size && col < size) {
        C[row * size + col] = sum;
    }
}


// 3. CUDA Kernel: Matrix multiplication using Constant Memory
__constant__ int c_B[128 * 128]; // Max size for constant memory B

__global__ void matrixMultConstantKernel(int* A, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * c_B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}


// --- Main Host Program ---
int main() {
    printf("CUDA Matrix Multiplication Performance Comparison\n");
    printf("--------------------------------------------------\n\n");

    // Print header for the results table
    printf("%-8s | %-15s | %-12s | %-12s | %-12s | %-12s | %-12s\n",
           "Size", "Kernel Type", "HtoD (ms)", "Kernel (ms)", "DtoH (ms)", "Total (ms)", "Verified");
    printf("-------------------------------------------------------------------------------------------------------------------\n");

    // --- CUDA Events for Timing ---
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    float h_to_d_time_ms, kernel_time_ms, d_to_h_time_ms, total_gpu_time_ms;
    int verification_errors = 0;

    for (int s_idx = 0; s_idx < NUM_MATRIX_SIZES; ++s_idx) {
        int size = MATRIX_SIZES[s_idx];
        long long matrix_elements = (long long)size * size;
        long long matrix_bytes = matrix_elements * sizeof(int);

        // Host matrices
        int* h_A = (int*)malloc(matrix_bytes);
        int* h_B = (int*)malloc(matrix_bytes);
        int* h_C_cpu = (int*)malloc(matrix_bytes);
        int* h_C_gpu = (int*)malloc(matrix_bytes); // Reused for all kernel results

        if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
            fprintf(stderr, "Host memory allocation failed for size %d! Exiting.\n", size);
            exit(1); // Exit on critical allocation failure
        }
        memset(h_C_gpu, 0, matrix_bytes); // Clear for each test

        // --- Initialize Host Matrices ---
        for (long long i = 0; i < matrix_elements; i++) {
            h_A[i] = i % 10 + 1; // Values from 1 to 10
            h_B[i] = (i % 5) + 1; // Values from 1 to 5
        }

    
        // --- Perform CPU Multiplication for Verification ---
        multiplySquareMatricesCPU(h_A, h_B, h_C_cpu, size);


        // --- Common Kernel Launch Configuration ---
        dim3 dimBlock_global(BLOCK_SIZE_GLOBAL, BLOCK_SIZE_GLOBAL);
        dim3 dimGrid_global((size + dimBlock_global.x - 1) / dimBlock_global.x,
                            (size + dimBlock_global.y - 1) / dimBlock_global.y);

        dim3 dimBlock_shared(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid_shared((size + dimBlock_shared.x - 1) / dimBlock_shared.x,
                            (size + dimBlock_shared.y - 1) / dimBlock_shared.y);


        // =========================================================================
        // 1. GLOBAL MEMORY KERNEL
        // =========================================================================
        int* d_A_global, *d_B_global, *d_C_global;
        cudaMalloc((void**)&d_A_global, matrix_bytes);
        cudaMalloc((void**)&d_B_global, matrix_bytes);
        cudaMalloc((void**)&d_C_global, matrix_bytes);

        cudaEventRecord(start_event);
        cudaMemcpy(d_A_global, h_A, matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_global, h_B, matrix_bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&h_to_d_time_ms, start_event, stop_event);

        cudaEventRecord(start_event);
        matrixMultGlobalKernel<<<dimGrid_global, dimBlock_global>>>(d_A_global, d_B_global, d_C_global, size);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
        //check for errors silently
        cudaGetLastError();

        cudaEventRecord(start_event);
        cudaMemcpy(h_C_gpu, d_C_global, matrix_bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&d_to_h_time_ms, start_event, stop_event);

        total_gpu_time_ms = h_to_d_time_ms + kernel_time_ms + d_to_h_time_ms;
        verification_errors = verifyResult(h_C_cpu, h_C_gpu, size);
        printf("%-8d | %-15s | %-12.3f | %-12.3f | %-12.3f | %-12.3f | %-12s\n",
               size, "Global", h_to_d_time_ms, kernel_time_ms, d_to_h_time_ms, total_gpu_time_ms,
               (verification_errors == 0 ? "PASS" : "FAIL"));

        cudaFree(d_A_global);
        cudaFree(d_B_global);
        cudaFree(d_C_global);
        memset(h_C_gpu, 0, matrix_bytes);


        // =========================================================================
        // 2. SHARED MEMORY KERNEL
        // =========================================================================
        int* d_A_shared, *d_B_shared, *d_C_shared;
        cudaMalloc((void**)&d_A_shared, matrix_bytes);
        cudaMalloc((void**)&d_B_shared, matrix_bytes);
        cudaMalloc((void**)&d_C_shared, matrix_bytes);

        cudaEventRecord(start_event);
        cudaMemcpy(d_A_shared, h_A, matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_shared, h_B, matrix_bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&h_to_d_time_ms, start_event, stop_event);

        cudaEventRecord(start_event);
        matrixMultSharedKernel<<<dimGrid_shared, dimBlock_shared>>>(d_A_shared, d_B_shared, d_C_shared, size);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
        cudaGetLastError();

        cudaEventRecord(start_event);
        cudaMemcpy(h_C_gpu, d_C_shared, matrix_bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&d_to_h_time_ms, start_event, stop_event);

        total_gpu_time_ms = h_to_d_time_ms + kernel_time_ms + d_to_h_time_ms;
        verification_errors = verifyResult(h_C_cpu, h_C_gpu, size);
        printf("%-8d | %-15s | %-12.3f | %-12.3f | %-12.3f | %-12.3f | %-12s\n",
               size, "Shared", h_to_d_time_ms, kernel_time_ms, d_to_h_time_ms, total_gpu_time_ms,
               (verification_errors == 0 ? "PASS" : "FAIL"));

        cudaFree(d_A_shared);
        cudaFree(d_B_shared);
        cudaFree(d_C_shared);
        memset(h_C_gpu, 0, matrix_bytes);


        // =========================================================================
        // 3. CONSTANT MEMORY KERNEL
        // =========================================================================
        if (size * size * sizeof(int) <= (64 * 1024)) { // 64KB constant memory limit
            // if it gets bigger ( like 1024x124 the program wont compile)
            int* d_A_constant, *d_C_constant;
            cudaMalloc((void**)&d_A_constant, matrix_bytes);
            cudaMalloc((void**)&d_C_constant, matrix_bytes);

            cudaEventRecord(start_event);
            cudaMemcpy(d_A_constant, h_A, matrix_bytes, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(c_B, h_B, matrix_bytes, 0, cudaMemcpyHostToDevice); // Copy to constant memory symbol
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&h_to_d_time_ms, start_event, stop_event);

            cudaEventRecord(start_event);
            matrixMultConstantKernel<<<dimGrid_global, dimBlock_global>>>(d_A_constant, d_C_constant, size);
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
            cudaGetLastError();

            cudaEventRecord(start_event);
            cudaMemcpy(h_C_gpu, d_C_constant, matrix_bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&d_to_h_time_ms, start_event, stop_event);

            total_gpu_time_ms = h_to_d_time_ms + kernel_time_ms + d_to_h_time_ms;
            verification_errors = verifyResult(h_C_cpu, h_C_gpu, size);
            printf("%-8d | %-15s | %-12.3f | %-12.3f | %-12.3f | %-12.3f | %-12s\n",
                   size, "Constant", h_to_d_time_ms, kernel_time_ms, d_to_h_time_ms, total_gpu_time_ms,
                   (verification_errors == 0 ? "PASS" : "FAIL"));

            cudaFree(d_A_constant);
            cudaFree(d_C_constant);
            memset(h_C_gpu, 0, matrix_bytes);
        } else {
            printf("%-8d | %-15s | %-12s | %-12s | %-12s | %-12s | %-12s\n",
                   size, "Constant", "N/A", "N/A", "N/A", "N/A", "N/A (Too Large)");
        }
        printf("-------------------------------------------------------------------------------------------------------------------\n");


        // --- Free Host Memory for current size ---
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
    }

    // --- Destroy CUDA Events ---
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
/**
output



Size     | Kernel Type     | HtoD (ms)    | Kernel (ms)  | DtoH (ms)    | Total (ms)   | Verified    
-------------------------------------------------------------------------------------------------------------------
6        | Global          | 0.307        | 3.241        | 0.057        | 3.605        | PASS        
6        | Shared          | 0.048        | 0.061        | 0.056        | 0.165        | PASS        
6        | Constant        | 0.047        | 0.061        | 0.055        | 0.164        | PASS        
-------------------------------------------------------------------------------------------------------------------
16       | Global          | 0.054        | 0.004        | 0.058        | 0.117        | PASS        
16       | Shared          | 0.050        | 0.005        | 0.058        | 0.113        | PASS        
16       | Constant        | 0.050        | 0.007        | 0.059        | 0.116        | PASS        
-------------------------------------------------------------------------------------------------------------------
32       | Global          | 0.050        | 0.005        | 0.059        | 0.115        | PASS        
32       | Shared          | 0.051        | 0.005        | 0.056        | 0.112        | PASS        
32       | Constant        | 0.048        | 0.009        | 0.057        | 0.115        | PASS        
-------------------------------------------------------------------------------------------------------------------
64       | Global          | 0.054        | 0.005        | 0.060        | 0.119        | PASS        
64       | Shared          | 0.053        | 0.006        | 0.059        | 0.118        | PASS        
64       | Constant        | 0.050        | 0.018        | 0.057        | 0.126        | PASS        
-------------------------------------------------------------------------------------------------------------------
128      | Global          | 0.052        | 0.012        | 0.056        | 0.120        | PASS        
128      | Shared          | 0.068        | 0.023        | 0.067        | 0.158        | PASS        
128      | Constant        | 0.051        | 0.271        | 0.058        | 0.381        | PASS        
-------------------------------------------------------------------------------------------------------------------
256      | Global          | 0.310        | 0.092        | 0.127        | 0.528        | PASS        
256      | Shared          | 0.109        | 0.071        | 0.084        | 0.264        | PASS        
256      | Constant        | N/A          | N/A          | N/A          | N/A          | N/A (Too Large)
-------------------------------------------------------------------------------------------------------------------
512      | Global          | 0.313        | 0.450        | 0.232        | 0.996        | PASS        
512      | Shared          | 0.258        | 0.298        | 0.179        | 0.735        | PASS        
512      | Constant        | N/A          | N/A          | N/A          | N/A          | N/A (Too Large)
-------------------------------------------------------------------------------------------------------------------
1024     | Global          | 0.950        | 2.837        | 0.522        | 4.309        | PASS        
1024     | Shared          | 0.789        | 2.099        | 0.497        | 3.386        | PASS        
1024     | Constant        | N/A          | N/A          | N/A          | N/A          | N/A (Too Large)
-------------------------------------------------------------------------------------------------------------------
^C^C^C^C^C^C^C^C^C^C^C^C^C
➜  lab4 git:(main) ✗ nvcc main2.cu -o main2
➜  lab4 git:(main) ✗ ./main2               
CUDA Matrix Multiplication Performance Comparison
--------------------------------------------------

Size     | Kernel Type     | HtoD (ms)    | Kernel (ms)  | DtoH (ms)    | Total (ms)   | Verified    
-------------------------------------------------------------------------------------------------------------------
6        | Global          | 0.271        | 1.687        | 0.074        | 2.033        | PASS        
6        | Shared          | 0.081        | 0.069        | 0.069        | 0.218        | PASS        
6        | Constant        | 0.076        | 0.065        | 0.070        | 0.211        | PASS        
-------------------------------------------------------------------------------------------------------------------
16       | Global          | 0.107        | 0.005        | 0.087        | 0.199        | PASS        
16       | Shared          | 0.104        | 0.005        | 0.086        | 0.194        | PASS        
16       | Constant        | 0.193        | 0.012        | 0.086        | 0.290        | PASS        
-------------------------------------------------------------------------------------------------------------------
32       | Global          | 0.106        | 0.005        | 0.086        | 0.197        | PASS        
32       | Shared          | 0.104        | 0.005        | 0.086        | 0.195        | PASS        
32       | Constant        | 0.103        | 0.014        | 0.085        | 0.202        | PASS        
-------------------------------------------------------------------------------------------------------------------
64       | Global          | 0.112        | 0.005        | 0.091        | 0.208        | PASS        
64       | Shared          | 0.080        | 0.006        | 0.069        | 0.155        | PASS        
64       | Constant        | 0.078        | 0.015        | 0.068        | 0.161        | PASS        
-------------------------------------------------------------------------------------------------------------------
128      | Global          | 0.084        | 0.012        | 0.070        | 0.166        | PASS        
128      | Shared          | 0.118        | 0.010        | 0.099        | 0.228        | PASS        
128      | Constant        | 0.082        | 0.273        | 0.068        | 0.423        | PASS        
-------------------------------------------------------------------------------------------------------------------
256      | Global          | 0.111        | 0.151        | 0.159        | 0.421        | PASS        
256      | Shared          | 0.108        | 0.071        | 0.087        | 0.266        | PASS        
256      | Constant        | N/A          | N/A          | N/A          | N/A          | N/A (Too Large)
-------------------------------------------------------------------------------------------------------------------
512      | Global          | 0.282        | 0.460        | 0.175        | 0.918        | PASS        
512      | Shared          | 0.263        | 0.304        | 0.178        | 0.745        | PASS        
512      | Constant        | N/A          | N/A          | N/A          | N/A          | N/A (Too Large)
-------------------------------------------------------------------------------------------------------------------
1024     | Global          | 1.308        | 2.858        | 0.615        | 4.781        | PASS        
1024     | Shared          | 0.939        | 2.180        | 0.515        | 3.635        | PASS        
1024     | Constant        | N/A          | N/A          | N/A          | N/A          | N/A (Too Large)
-------------------------------------------------------------------------------------------------------------------
2048     | Global          | 10.601       | 274.443      | 6.154        | 291.198      | PASS        
2048     | Shared          | 4.724        | 17.159       | 2.226        | 24.109       | PASS        
2048     | Constant        | N/A          | N/A          | N/A          | N/A          | N/A (Too Large)
-------------------------------------------------------------------------------------------------------------------

*/