#include <iostream>
#include <cuda_runtime.h>


__global__ void reverse_list(int *list, int *reverse, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n){
        reverse[idx] = list[n - idx -1];
    }
}

int main() {
    const int arraySize = 10;
    int a[arraySize], r[arraySize];

    int *p_a, *p_r;

    for (int i = 0; i < arraySize; i++){
        a[i] = i;
    }


    cudaMalloc((void**)&p_a, sizeof(int)*arraySize);
    cudaMalloc((void**)&p_r, sizeof(int)*arraySize);
    cudaMemcpy(p_a, &a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_r, &r, arraySize * sizeof(int), cudaMemcpyHostToDevice);


    int numBlocks = 3;
    int threadsPerBlock = (arraySize + numBlocks - 1) / numBlocks;
    reverse_list<<<numBlocks, threadsPerBlock>>>(p_a, p_r, arraySize);
    reverse_list<<<numBlocks, threadsPerBlock>>>(p_a, p_r, arraySize);

    cudaMemcpy(&r, p_r, sizeof(int)*arraySize, cudaMemcpyDeviceToHost);

    std::cout << "Initial array: ";
    for (int i = 0; i < arraySize; i++){
        std::cout << a[i];
    }
    std::cout << std::endl;

    
    std::cout << "Reverse array: ";
    for (int i = 0; i < arraySize; i++){
        std::cout << r[i];
    }
    std::cout << std::endl;


    cudaFree(p_a);
    cudaFree(p_r);

    return 0;
}