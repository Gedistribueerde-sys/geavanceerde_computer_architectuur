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

int main() {
    clock_t start,end;
    srand(time(NULL));  // seed the random number generator
    for (int N= 32;N < 300000; N*=2){
        int arr[N];
        int numBlocks = 4;
        int blocksPerThread = N / (numBlocks*2);
        start = clock();
        for(int i = 0; i< 1000; i++){

             for (int j = 0; j<4 ; j++){

           
            for (int d = 0; d < N; d++) {
                arr[d] = rand() % 1000;
            }
            int Num = 0;
            int *p_arr, *p_Num;
            cudaMalloc((void**)&p_arr, sizeof(int)*N);
            cudaMalloc((void**)&p_Num, sizeof(int));
            cudaMemcpy(p_arr, arr, sizeof(int)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(p_Num, &Num, sizeof(int), cudaMemcpyHostToDevice);
            if (j ==0) reduction_max<<<numBlocks, blocksPerThread>>>(p_arr, p_Num, N);
            else if (j ==1) reduction_min<<<numBlocks, blocksPerThread>>>(p_arr, p_Num, N);
            else if (j ==2) reduction_sum<<<numBlocks, blocksPerThread>>>(p_arr, p_Num, N);
            else if (j ==3) reduction_Sub<<<numBlocks, blocksPerThread>>>(p_arr, p_Num, N);

            
            cudaDeviceSynchronize () ;
            cudaMemcpy(&Num, p_Num, sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(p_arr);
            cudaFree(p_Num);

        }
        }

        cudaDeviceSynchronize();

        end = clock();     

        double elapsed_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
        std::cout << "Totale programmatijd: " << elapsed_ms << " ms" <<" N: " << N<<   std::endl;
        
    }
    return 0;
}

