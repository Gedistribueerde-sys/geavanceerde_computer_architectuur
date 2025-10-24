#include <iostream>
#include <cuda_runtime.h>

__global__ void atomic_max(int* list, int* p_max, int N){
    int idx = threadIdx.x + blockDim.x + blockIdx.x;
    if (idx < N)
    atomicMax(p_max, list[idx]);
}


int main() {
    const int N = 10;
    int arr[N] = {23, 15, 85, 21, 0, 7, 99, 1, 2, 4};
    int *p_arr, *p_max, maxNum = 0;

    cudaMalloc((void**)&p_arr, sizeof(int)*N);
    cudaMalloc((void**)&p_max, sizeof(int));
    cudaMemcpy(p_arr, arr, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(p_max, &maxNum, sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = 2;
    int threadsBlock = N / numBlocks;
    atomic_max<<<numBlocks, threadsBlock>>>(p_arr, p_max, N);


    cudaMemcpy(&maxNum, p_max, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "The max number in the array is: " << maxNum << std::endl;

    cudaFree(p_arr);
    cudaFree(p_max);
 
}
