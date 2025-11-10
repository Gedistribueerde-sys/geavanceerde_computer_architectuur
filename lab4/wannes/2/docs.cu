#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define BLOCK_SIZE 16
#define MAX_CONST_SIZE 16384  // 64KB / 4 bytes per int

__constant__ int constB[MAX_CONST_SIZE];

typedef struct {
    int width;
    int height;
    int stride;
    int* elements;
} Matrix;

__device__ int GetElement(const Matrix A, int row, int col){
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, int value){
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col){
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulGlobal(Matrix A, Matrix B, Matrix C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < C.height && col < C.width) {
        int Cvalue = 0;
        for (int e = 0; e < A.width; ++e){
            Cvalue += GetElement(A, row, e) * GetElement(B, e, col);
        }
        SetElement(C, row, col, Cvalue);
    }
}

__global__ void MatMulShared(Matrix A, Matrix B, Matrix C){
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    int Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m){
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e){
            Cvalue += As[row][e] * Bs[e][col];
        }

        __syncthreads();
    }

    SetElement(Csub, row, col, Cvalue);
}

__global__ void MatMulConstant(Matrix A, Matrix B, Matrix C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < C.height && col < C.width) {
        int Cvalue = 0;
        for (int e = 0; e < A.width; ++e){
            Cvalue += GetElement(A, row, e) * constB[e * B.width + col];
        }
        SetElement(C, row, col, Cvalue);
    }
}

void fill_matrix(Matrix& M, int seed){
    srand(seed);
    for (int i = 0; i < M.height * M.width; ++i){
        M.elements[i] = rand() % 1000;
    }
}

void benchmark_kernel(const char* name, void (*kernel)(Matrix, Matrix, Matrix), 
                      Matrix A, Matrix B, Matrix C, dim3 grid, dim3 block, int runs){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0;
    
    for (int i = 0; i < runs; ++i){
        cudaEventRecord(start);
        
        if (kernel == (void(*)(Matrix, Matrix, Matrix))MatMulGlobal){
            MatMulGlobal<<<grid, block>>>(A, B, C);
        } else if (kernel == (void(*)(Matrix, Matrix, Matrix))MatMulShared){
            MatMulShared<<<grid, block>>>(A, B, C);
        } else if (kernel == (void(*)(Matrix, Matrix, Matrix))MatMulConstant){
            MatMulConstant<<<grid, block>>>(A, B, C);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    
    printf("%s: Avg time = %.4f ms (over %d runs)\n", name, total_time / runs, runs);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    int sizes[] = {64, 96, 192, 384, 768};
    int num_sizes = 5;
    int runs = 100;

    
    for (int s = 0; s < num_sizes; ++s){
        int N = sizes[s];
        printf("\n=== Matrix size: %dx%d ===\n", N, N);
        
        // Allocate host matrices
        Matrix A, B, C;
        A.width = A.height = A.stride = N;
        B.width = B.height = B.stride = N;
        C.width = C.height = C.stride = N;
        
        A.elements = new int[N * N];
        B.elements = new int[N * N];
        C.elements = new int[N * N];
        
        fill_matrix(A, s);
        fill_matrix(B, s);
        
        // Allocate device matrices
        Matrix d_A, d_B, d_C;
        d_A.width = d_A.height = d_A.stride = N;
        d_B.width = d_B.height = d_B.stride = N;
        d_C.width = d_C.height = d_C.stride = N;
        
        size_t bytes = N * N * sizeof(int);
        cudaMalloc(&d_A.elements, bytes);
        cudaMalloc(&d_B.elements, bytes);
        cudaMalloc(&d_C.elements, bytes);
        
        cudaMemcpy(d_A.elements, A.elements, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B.elements, B.elements, bytes, cudaMemcpyHostToDevice);
        
        // Benchmark Global
        dim3 blockGlobal(16, 16);
        dim3 gridGlobal((N + 15) / 16, (N + 15) / 16);
        benchmark_kernel("MatMulGlobal", (void(*)(Matrix, Matrix, Matrix))MatMulGlobal, 
                        d_A, d_B, d_C, gridGlobal, blockGlobal, runs);
        
        // Benchmark Shared
        dim3 blockShared(16, 16);
        dim3 gridShared(N / 16, N / 16);
        benchmark_kernel("MatMulShared", (void(*)(Matrix, Matrix, Matrix))MatMulShared, 
                        d_A, d_B, d_C, gridShared, blockShared, runs);
        
        // Benchmark Constant
        if (N * N <= MAX_CONST_SIZE){
            cudaMemcpyToSymbol(constB, B.elements, bytes);
            benchmark_kernel("MatMulConstant", (void(*)(Matrix, Matrix, Matrix))MatMulConstant, 
                            d_A, d_B, d_C, gridGlobal, blockGlobal, runs);
        } else {
            printf("MatMulConstant: Matrix too large (%dx%d=%d > %d)\n", N, N, N*N, MAX_CONST_SIZE);
        }
        
        // Copy result back for verification
        cudaMemcpy(C.elements, d_C.elements, bytes, cudaMemcpyDeviceToHost);
        
        // Free memory
        delete[] A.elements;
        delete[] B.elements;
        delete[] C.elements;
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }

    return 0;
}