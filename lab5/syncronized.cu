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
            /*----------max---------- */
            // generate random numbers between 0 and 999
            for (int i = 0; i < N; i++) {
                arr[i] = rand() % 1000;
            }
            int maxNum = 0;
            int *p_arr, *p_maxNum;
            cudaMalloc((void**)&p_arr, sizeof(int)*N);
            cudaMalloc((void**)&p_maxNum, sizeof(int));
            cudaMemcpy(p_arr, arr, sizeof(int)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(p_maxNum, &maxNum, sizeof(int), cudaMemcpyHostToDevice);

            reduction_max<<<numBlocks, blocksPerThread>>>(p_arr, p_maxNum, N);
cudaDeviceSynchronize () ;
            cudaMemcpy(&maxNum, p_maxNum, sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(p_arr);
            cudaFree(p_maxNum);
            /*----------min---------- */
            // generate random numbers between 0 and 999
            for (int i = 0; i < N; i++) {
                arr[i] = rand() % 1000;
            }
            int minNum = 0;
            int *p_minNum;
            cudaMalloc((void**)&p_arr, sizeof(int)*N);
            cudaMalloc((void**)&p_minNum, sizeof(int));
            cudaMemcpy(p_arr, arr, sizeof(int)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(p_minNum, &minNum, sizeof(int), cudaMemcpyHostToDevice);

            reduction_min<<<numBlocks, blocksPerThread>>>(p_arr, p_minNum, N);
cudaDeviceSynchronize () ;
            cudaMemcpy(&minNum, p_minNum, sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(p_arr);
            cudaFree(p_minNum);

            /*---------sum----------*/
            // generate random numbers between 0 and 999
            for (int i = 0; i < N; i++) {
                arr[i] = rand() % 1000;
            }
            int sum = 0;
            int *p_sum;
            cudaMalloc((void**)&p_arr, sizeof(int)*N);
            cudaMalloc((void**)&p_sum, sizeof(int));
            cudaMemcpy(p_arr, arr, sizeof(int)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(p_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);

            reduction_sum<<<numBlocks, blocksPerThread>>>(p_arr, p_sum, N);
cudaDeviceSynchronize () ;
            cudaMemcpy(&sum, p_sum, sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(p_arr);
            cudaFree(p_sum);
                /*---------Sub----------*/

            // generate random numbers between 0 and 999
            for (int i = 0; i < N; i++) {
                arr[i] = rand() % 1000;
            }
            int sub = 0;
            int *p_sub;
            cudaMalloc((void**)&p_arr, sizeof(int)*N);
            cudaMalloc((void**)&p_sub, sizeof(int));
            cudaMemcpy(p_arr, arr, sizeof(int)*N, cudaMemcpyHostToDevice);
            cudaMemcpy(p_sub, &sub, sizeof(int), cudaMemcpyHostToDevice);

            reduction_Sub<<<numBlocks, blocksPerThread>>>(p_arr, p_sub, N);
cudaDeviceSynchronize () ;
            cudaMemcpy(&sub, p_sub, sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(p_arr);
            cudaFree(p_sub);


        }

        cudaDeviceSynchronize();

        end = clock();     

        double elapsed_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
        std::cout << "Totale programmatijd: " << elapsed_ms << " ms" <<" N: " << N<<   std::endl;
        
    }
    return 0;
}

/*
output
Totale programmatijd: 926.072 ms N: 1024
Totale programmatijd: 862.901 ms N: 2048
Totale programmatijd: 1008.52 ms N: 4096
Totale programmatijd: 1495.55 ms N: 8192
Totale programmatijd: 2288.3 ms N: 16384
Totale programmatijd: 3461.04 ms N: 32768
Totale programmatijd: 5301.68 ms N: 65536

*/