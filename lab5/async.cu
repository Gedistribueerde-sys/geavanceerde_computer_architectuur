#include<iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
__global__ void reduction_max(int *list, int* maxNum, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int offset = 1; offset < N; offset *= 2){
        int idx = tid * 2 * offset;
        if (idx + offset < N){
            atomicMax(&list[idx], list[idx + offset]);
        }
        __syncthreads();
    }

    if (tid == 0)
    *maxNum = list[tid];
}

__global__ void reduction_min(int *list, int* maxNum, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int offset = 1; offset < N; offset *= 2){
        int idx = tid * 2 * offset;
        if (idx + offset < N){
            atomicMin(&list[idx], list[idx + offset]);
        }
        __syncthreads();
    }

    if (tid == 0)
    *maxNum = list[tid];
}

__global__ void reduction_sum(int *list, int* maxNum, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int offset = 1; offset < N; offset *= 2){
        int idx = tid * 2 * offset;
        if (idx + offset < N){
            atomicAdd(&list[idx], list[idx + offset]);
        }
        __syncthreads();
    }

    if (tid == 0)
    *maxNum = list[tid];
}
__global__ void reduction_Sub(int *list, int* maxNum, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int offset = 1; offset < N; offset *= 2){
        int idx = tid * 2 * offset;
        if (idx + offset < N){
            atomicSub(&list[idx], list[idx + offset]);
        }
        __syncthreads();
    }

    if (tid == 0)
    *maxNum = list[tid];
}

