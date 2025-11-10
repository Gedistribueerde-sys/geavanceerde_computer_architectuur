/******************************************************************************************
 * CUDA Matrix Multiplication Benchmark – Global / Shared / Constant memory
 * • Clean, tabular output
 * • Grid size printed for every kernel
 * • Separate timing of host to GPU and host to constant transfers
 * • Verification of correctness for each kernel
 ******************************************************************************************/

#include <iostream>
#include <iomanip>      // For std::setw, std::setprecision, std::fixed
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <sstream>      // For std::stringstream (new)
#include <string>       // For std::string (new)

#define BLOCK_SIZE 4
#define MAX_CONST_SIZE 16384          // 64 KB / 4 B per int

__constant__ int constB[MAX_CONST_SIZE];

struct Matrix {
    int width, height, stride;
    int* elements;
};

__device__ int  GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}
__device__ void SetElement(Matrix A, int row, int col, int value) {
    A.elements[row * A.stride + col] = value;
}
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix sub;
    sub.width = sub.height = BLOCK_SIZE;
    sub.stride = A.stride;
    sub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sub;
}

/* -------------------------- Kernels -------------------------- */

__global__ void MatMulGlobal(Matrix A, Matrix B, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= C.height || col >= C.width) return;

    int sum = 0;
    for (int k = 0; k < A.width; ++k)
        sum += GetElement(A, row, k) * GetElement(B, k, col);
    SetElement(C, row, col, sum);
}

__global__ void MatMulShared(Matrix A, Matrix B, Matrix C) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    Matrix Csub = GetSubMatrix(C, by, bx);

    int sum = 0;
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, by, m);
        Matrix Bsub = GetSubMatrix(B, m, bx);

        __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = GetElement(Asub, ty, tx);
        Bs[ty][tx] = GetElement(Bsub, ty, tx);
        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            sum += As[ty][e] * Bs[e][tx];
        __syncthreads();
    }

    int globalRow = by * BLOCK_SIZE + ty;
    int globalCol = bx * BLOCK_SIZE + tx;
    if (globalRow < C.height && globalCol < C.width)
        SetElement(Csub, ty, tx, sum);
}

__global__ void MatMulConstant(Matrix A, Matrix B, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= C.height || col >= C.width) return;

    int sum = 0;
    for (int k = 0; k < A.width; ++k)
        sum += GetElement(A, row, k) * constB[k * B.width + col];
    SetElement(C, row, col, sum);
}

/* Dummy kernel that only performs the global to shared copies – used to time the
   migration cost for the shared-memory version. */
__global__ void LoadToShared(Matrix A, Matrix B, Matrix C) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, by, m);
        Matrix Bsub = GetSubMatrix(B, m, bx);
        __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[ty][tx] = GetElement(Asub, ty, tx);
        Bs[ty][tx] = GetElement(Bsub, ty, tx);
        __syncthreads();
    }
}

/* -------------------------- Helpers -------------------------- */

void fill_matrix(Matrix& M, unsigned seed) {
    srand(seed);
    for (int i = 0; i < M.height * M.width; ++i)
        M.elements[i] = rand() % 1000;
}

/* CPU reference – returns true if the GPU result matches */
bool verify(const Matrix& gpuC, const Matrix& A, const Matrix& B) {
    int* ref = new int[A.height * B.width];
    std::memset(ref, 0, A.height * B.width * sizeof(int));

    for (int i = 0; i < A.height; ++i)
        for (int j = 0; j < B.width; ++j) {
            int sum = 0;
            for (int k = 0; k < A.width; ++k)
                sum += A.elements[i * A.stride + k] *
                       B.elements[k * B.stride + j];
            ref[i * B.width + j] = sum;
        }

    bool ok = std::memcmp(gpuC.elements, ref, A.height * B.width * sizeof(int)) == 0;
    delete[] ref;
    return ok;
}

/* Time a kernel (average over `runs`) */
float time_kernel(const char* name,
                  void (*kernel)(Matrix, Matrix, Matrix),
                  Matrix A, Matrix B, Matrix C,
                  dim3 grid, dim3 block,
                  int runs = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0.0f;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        if (kernel == (void(*)(Matrix,Matrix,Matrix))MatMulGlobal)
            MatMulGlobal<<<grid, block>>>(A, B, C);
        else if (kernel == (void(*)(Matrix,Matrix,Matrix))MatMulShared)
            MatMulShared<<<grid, block>>>(A, B, C);
        else if (kernel == (void(*)(Matrix,Matrix,Matrix))MatMulConstant)
            MatMulConstant<<<grid, block>>>(A, B, C);
        else if (kernel == (void(*)(Matrix,Matrix,Matrix))LoadToShared)
            LoadToShared<<<grid, block>>>(A, B, C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / runs;
}

/* Time a single memcpy (average over `runs`) */
float time_memcpy(cudaMemcpyKind kind, void* dst, const void* src,
                  size_t bytes, int runs = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total = 0.0f;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        cudaMemcpy(dst, src, bytes, kind);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / runs;
}

/* -------------------------- Main -------------------------- */

int main() {
    const int sizes[] = {8, 16, 32, 64, 96, 192, 384, 768};
    const int nSizes = sizeof(sizes)/sizeof(sizes[0]);
    const int benchRuns = 100;
    const int memRuns   = 10;

    // Set up output formatting for all subsequent std::cout
    std::cout << std::fixed << std::setprecision(4);

    for (int si = 0; si < nSizes; ++si) {
        int N = sizes[si];
        std::cout << "\n=== N = " << std::setw(4) << N << " x " << N << " ===\n";

        /* ---------- Host allocation ---------- */
        Matrix hA{ N, N, N, new int[N*N] };
        Matrix hB{ N, N, N, new int[N*N] };
        Matrix hC{ N, N, N, new int[N*N] };
        fill_matrix(hA, si*2);
        fill_matrix(hB, si*2+1);

        /* ---------- Device allocation ---------- */
        Matrix dA{ N, N, N, nullptr };
        Matrix dB{ N, N, N, nullptr };
        Matrix dC{ N, N, N, nullptr };
        size_t bytes = N*N*sizeof(int);
        cudaMalloc(&dA.elements, bytes);
        cudaMalloc(&dB.elements, bytes);
        cudaMalloc(&dC.elements, bytes);

        /* ---------- Transfer timings ---------- */
        float tA = time_memcpy(cudaMemcpyHostToDevice, dA.elements, hA.elements, bytes, memRuns);
        float tB = time_memcpy(cudaMemcpyHostToDevice, dB.elements, hB.elements, bytes, memRuns);
        
        float tConstCopy = 0.0f;
        bool constFits = (N*N <= MAX_CONST_SIZE);
        if (constFits) {
            // Time the copy to the *symbol* (which goes to constant memory)
            tConstCopy = time_memcpy(cudaMemcpyHostToDevice, (void*)constB,
                                     hB.elements, bytes, memRuns);
        }

        std::cout << "  Memory Transfers (avg over " << memRuns << " runs):\n";
        std::cout << "  ├ Host to Device (A) : " << std::right << std::setw(10) << tA << " ms\n";
        std::cout << "  ├ Host to Device (B) : " << std::right << std::setw(10) << tB << " ms\n";
        if (constFits) {
            std::cout << "  └ Host to Constant (B): " << std::right << std::setw(10) << tConstCopy << " ms\n";
        } else {
            std::cout << "  └ Host to Constant (B):  (too large)\n";
        }

        /* ---------- Ensure data on device (for kernel launch) ---------- */
        cudaMemcpy(dA.elements, hA.elements, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB.elements, hB.elements, bytes, cudaMemcpyHostToDevice);
        if (constFits) {
            // This copy is not timed, it just ensures data is there for the benchmark
            cudaMemcpyToSymbol(constB, hB.elements, bytes);
        }

        /* ---------- Grid / block configs ---------- */
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        // Grid for global/constant kernels (handles non-multiple sizes)
        dim3 gridGlob((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        // Grid for shared kernel (assumes N is multiple of BLOCK_SIZE)
        dim3 gridShrd(N / BLOCK_SIZE, N / BLOCK_SIZE);

        /* ---------- Format Grid Strings for Output ---------- */
        std::stringstream ssGlob, ssShrd;
        ssGlob << "(" << gridGlob.x << "x" << gridGlob.y << ")";
        ssShrd << "(" << gridShrd.x << "x" << gridShrd.y << ")";

        // Labels for the table
        std::string globLabel = "Global " + ssGlob.str();
        std::string shrdLabel = "Shared " + ssShrd.str();
        std::string loadLabel = "Load to Shared " + ssShrd.str();
        std::string constLabel = "Constant " + ssGlob.str();


        /* ---------- Benchmark table header ---------- */
        std::cout << "\n  Kernel Benchmarks (avg over " << benchRuns << " runs):\n";
        std::cout << "  ┌──────────────────────────────┬────────────┬────────────┐\n"
                  << "  │ " << std::left << std::setw(27) << "Kernel (grid)"
                  << "  │ " << std::right << std::setw(8) << "Time (ms)"
                  << "  │ " << std::left << std::setw(10) << "Verify" << " │\n"
                  << "  ├──────────────────────────────┼────────────┼────────────┤\n";

        /* ----- Global ----- */
        float tGlob = time_kernel("Global", (void(*)(Matrix,Matrix,Matrix))MatMulGlobal,
                                  dA, dB, dC, gridGlob, block, benchRuns);
        cudaMemset(dC.elements, 0, bytes); // Clear C for verification
        MatMulGlobal<<<gridGlob, block>>>(dA, dB, dC);
        cudaDeviceSynchronize();
        cudaMemcpy(hC.elements, dC.elements, bytes, cudaMemcpyDeviceToHost);
        bool okGlob = verify(hC, hA, hB);
        std::cout << "  │ " << std::left << std::setw(27) << globLabel
                  << "  │ " << std::right << std::setw(10) << tGlob
                  << " │ " << std::left << std::setw(10) << (okGlob ? "PASS" : "FAIL") << " │\n";

        /* ----- Shared ----- */
        // Only run shared if N is a multiple of BLOCK_SIZE
        if (N % BLOCK_SIZE == 0) {
            float tShrd = time_kernel("Shared", (void(*)(Matrix,Matrix,Matrix))MatMulShared,
                                      dA, dB, dC, gridShrd, block, benchRuns);
            cudaMemset(dC.elements, 0, bytes); // Clear C for verification
            MatMulShared<<<gridShrd, block>>>(dA, dB, dC);
            cudaDeviceSynchronize();
            cudaMemcpy(hC.elements, dC.elements, bytes, cudaMemcpyDeviceToHost);
            bool okShrd = verify(hC, hA, hB);
            std::cout << "  │ " << std::left << std::setw(27) << shrdLabel
                      << "  │ " << std::right << std::setw(10) << tShrd
                      << " │ " << std::left << std::setw(10) << (okShrd ? "PASS" : "FAIL") << " │\n";
            
            /* ----- Global to Shared migration ----- */
            float tLoad = time_kernel("LoadToShrd", (void(*)(Matrix,Matrix,Matrix))LoadToShared,
                                      dA, dB, dC, gridShrd, block, benchRuns);
            std::cout << "  │ " << std::left << std::setw(27) << loadLabel
                      << "  │ " << std::right << std::setw(9) << tLoad
                      << "  │ " << std::left << std::setw(10) << "-" << " │\n";
        } else {
            std::cout << "  │ " << std::left << std::setw(27) << "Shared (N not multiple)"
                      << "  │ " << std::right << std::setw(10) << "-"
                      << " │ " << std::left << std::setw(10) << "-" << " │\n";
            std::cout << "  │ " << std::left << std::setw(27) << "Load to Shared (N not mult)"
                      << "  │ " << std::right << std::setw(10) << "-"
                      << " │ " << std::left << std::setw(10) << "-" << " │\n";
        }


        /* ----- Constant (if fits) ----- */
        if (constFits) {
            float tConst = time_kernel("Constant", (void(*)(Matrix,Matrix,Matrix))MatMulConstant,
                                       dA, dB, dC, gridGlob, block, benchRuns);
            cudaMemset(dC.elements, 0, bytes); // Clear C for verification
            MatMulConstant<<<gridGlob, block>>>(dA, dB, dC);
            cudaDeviceSynchronize();
            cudaMemcpy(hC.elements, dC.elements, bytes, cudaMemcpyDeviceToHost);
            bool okConst = verify(hC, hA, hB);
            std::cout << "  │ " << std::left << std::setw(27) << constLabel
                      << "  │ " << std::right << std::setw(10) << tConst
                      << " │ " << std::left << std::setw(10) << (okConst ? "PASS" : "FAIL") << " │\n";
        } else {
            std::cout << "  │ " << std::left << std::setw(27) << "Constant (too large)"
                      << "  │ " << std::right << std::setw(10) << "-"
                      << " │ " << std::left << std::setw(10) << "-" << " │\n";
        }

        std::cout << "  └──────────────────────────────┴────────────┴────────────┘\n";

        /* ---------- Cleanup ---------- */
        delete[] hA.elements; delete[] hB.elements; delete[] hC.elements;
        cudaFree(dA.elements); cudaFree(dB.elements); cudaFree(dC.elements);
    }
    return 0;
}