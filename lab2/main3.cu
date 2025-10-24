#include<iostream>
#include <cuda_runtime.h>

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


int main() {
    const int N = 16; // Should be 2^x
    int arr[N] = {2, 15, 85, 21, 0, 7, 99, 1, 200, 4, 90, 11, 602, 43, 33, 11};
    int maxNum = 0;
    int *p_arr, *p_maxNum;

    cudaMalloc((void**)&p_arr, sizeof(int)*N);
    cudaMalloc((void**)&p_maxNum, sizeof(int));
    cudaMemcpy(p_arr, arr, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(p_maxNum, &maxNum, sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = 4;
    int blocksPerThread = N / (numBlocks*2);

    std::cout << "Numblocks: " << numBlocks << " and BlocksPerThread: " << blocksPerThread << std::endl;
    reduction_max<<<numBlocks, blocksPerThread>>>(p_arr, p_maxNum, N);

    cudaMemcpy(&maxNum, p_maxNum, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "The max num is: " << maxNum << std::endl;
    cudaFree(p_arr);
    cudaFree(p_maxNum);
    return 0;
}