#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <iomanip>      // For std::setw, std::setprecision
#include <cstdlib>      // For malloc, free, srand, rand
#include <ctime>        // For time()
#include <cuda_runtime.h> // CUDA Runtime API

// --- Image Dimensions ---
// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 960       // VR width
#define N 1280      // VR height
#define C 3         // Colors
#define OFFSET 16   // Header length

// --- CUDA Error Checking Utility ---
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}


// --- Image Loading/Saving ---
uint8_t* get_full_image_buffer(void){
    FILE *imageFile;
    imageFile=fopen("./anna.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the *entire* file (header + data)
    uint8_t* full_buffer = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    if (!full_buffer) {
         perror("ERROR: Cannot allocate memory");
         exit(EXIT_FAILURE);
    }

    // Read the image (header + data)
    fread(full_buffer, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);

    // Close the file
    fclose(imageFile);

    // Return the pointer to the START of the allocation
    return full_buffer;
}

void save_image_array(uint8_t* image_array ,int a){
    FILE *imageFile;
    if (a == 1) imageFile=fopen("./output_image_coalesced.ppm","wb");
    if (a == 2) imageFile=fopen("./output_image_uncoalesced.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(imageFile,"P6\n");
    fprintf(imageFile,"%d %d\n", M, N);
    fprintf(imageFile,"255\n");
    // Write the image data (M*N*C bytes)
    fwrite(image_array, 1, M*N*C, imageFile);
    fclose(imageFile);
}


// --- KERNEL 1: COALESCED ---
// Operates on a PLANAR array (RRR...GGG...BBB...)
__global__ void invertRedCoalesced(unsigned char* redChannel, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < numPixels; i += blockDim.x * gridDim.x) {
        redChannel[i] = 255 - redChannel[i];
    }
}

// --- KERNEL 2: UNCOALESCED ---
// Operates on an INTERLEAVED array (RGBGBRGB...)R
__global__ void invertRedUncoalesced(unsigned char* interleavedImage, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < numPixels; i += blockDim.x * gridDim.x) {
        int redIndex = i * C; // C is 3 (RGB)
        interleavedImage[redIndex] = 255 - interleavedImage[redIndex];
    }
}


int main (void) {

    // === Part 1: Process the 'anna.ppm' image ===
    std::cout << "--- Processing Image File (anna.ppm) ---" << std::endl;
    int numPixels = M * N;
    int blockSize = 256;
    int gridSize = (numPixels + blockSize - 1) / blockSize;

    uint8_t* full_image_buffer_host = get_full_image_buffer();
    uint8_t* image_array_host_interleaved = full_image_buffer_host + OFFSET;

    // 1. Create the planar (RRR...) host array for the coalesced test
    uint8_t* image_array_host_planar_R = (uint8_t*)malloc(numPixels * sizeof(uint8_t));
    for (int i = 0; i < numPixels; ++i) {
        image_array_host_planar_R[i] = image_array_host_interleaved[i * C];
    }
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float milliseconds = 0;

    // --- Test 1: Coalesced Kernel (on RRR... data) ---
    std::cout << "Running Coalesced Kernel..." << std::endl;
    uint8_t* d_planar_R;
    checkCudaErrors(cudaMalloc(&d_planar_R, numPixels * sizeof(uint8_t)));
    checkCudaErrors(cudaMemcpy(d_planar_R, image_array_host_planar_R, numPixels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start, 0));
    invertRedCoalesced<<<gridSize, blockSize>>>(d_planar_R, numPixels);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    
    std::cout << "Time for invertRedCoalesced: " << milliseconds << " ms" << std::endl;

    // Get result back and save image
    checkCudaErrors(cudaMemcpy(image_array_host_planar_R, d_planar_R, numPixels * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    uint8_t* output_image_coalesced = (uint8_t*)malloc(numPixels * C * sizeof(uint8_t));
    for (int i = 0; i < numPixels; ++i) {
        output_image_coalesced[i * C]     = image_array_host_planar_R[i];
        output_image_coalesced[i * C + 1] = image_array_host_interleaved[i * C + 1];
        output_image_coalesced[i * C + 2] = image_array_host_interleaved[i * C + 2];
    }
    save_image_array(output_image_coalesced, 1);
    free(output_image_coalesced);
    free(image_array_host_planar_R);
    checkCudaErrors(cudaFree(d_planar_R));


    // --- Test 2: Uncoalesced Kernel (on RGBRGB... data) ---
    std::cout << "\nRunning Uncoalesced Kernel..." << std::endl;
    uint8_t* d_interleaved_RGB;
    checkCudaErrors(cudaMalloc(&d_interleaved_RGB, numPixels * C * sizeof(uint8_t)));
    checkCudaErrors(cudaMemcpy(d_interleaved_RGB, image_array_host_interleaved, numPixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start, 0));
    invertRedUncoalesced<<<gridSize, blockSize>>>(d_interleaved_RGB, numPixels);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Time for invertRedUncoalesced: " << milliseconds << " ms" << std::endl;

    // Get result back and save image
    checkCudaErrors(cudaMemcpy(image_array_host_interleaved, d_interleaved_RGB, numPixels * C * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    save_image_array(image_array_host_interleaved, 2);
    checkCudaErrors(cudaFree(d_interleaved_RGB));


    // === Part 2: Benchmark with different array sizes ===
    
    std::cout << "\n\n--- Benchmarking Coalesced vs. Uncoalesced ---" << std::endl;
    std::cout << std::setw(12) << "Num Pixels"
              << std::setw(12) << "Coalesced (ms)"
              << std::setw(14) << "Uncoalesced (ms)"
              << std::setw(10) << "Slowdown" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    srand(time(NULL));

    for (int p = 20; p <= 26; ++p) {
        int N_pixels = 1 << p;
        int N_bytes_planar = N_pixels * sizeof(uint8_t);
        int N_bytes_interleaved = N_pixels * C * sizeof(uint8_t);

        uint8_t* h_planar_R = (uint8_t*)malloc(N_bytes_planar);
        uint8_t* h_interleaved_RGB = (uint8_t*)malloc(N_bytes_interleaved);
        if (!h_planar_R || !h_interleaved_RGB) {
            std::cerr << "Failed to allocate host memory for benchmark!" << std::endl;
            break;
        }

        for (int i = 0; i < N_pixels; ++i) {
            uint8_t r = rand() % 256;
            h_planar_R[i] = r;
            h_interleaved_RGB[i * C] = r;
            h_interleaved_RGB[i * C + 1] = rand() % 256;
            h_interleaved_RGB[i * C + 2] = rand() % 256;
        }

        uint8_t* d_planar, *d_interleaved;
        checkCudaErrors(cudaMalloc(&d_planar, N_bytes_planar));
        checkCudaErrors(cudaMalloc(&d_interleaved, N_bytes_interleaved));

        int benchGridSize = (N_pixels + blockSize - 1) / blockSize;
        float msCoalesced = 0, msUncoalesced = 0;

        // Time Coalesced
        checkCudaErrors(cudaMemcpy(d_planar, h_planar_R, N_bytes_planar, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaEventRecord(start, 0));
        invertRedCoalesced<<<benchGridSize, blockSize>>>(d_planar, N_pixels);
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&msCoalesced, start, stop));

        // Time Uncoalesced
        checkCudaErrors(cudaMemcpy(d_interleaved, h_interleaved_RGB, N_bytes_interleaved, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaEventRecord(start, 0));
        invertRedUncoalesced<<<benchGridSize, blockSize>>>(d_interleaved, N_pixels);
        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&msUncoalesced, start, stop));

        std::cout << std::setw(12) << N_pixels
                  << std::setw(12) << std::fixed << std::setprecision(4) << msCoalesced
                  << std::setw(14) << std::fixed << std::setprecision(4) << msUncoalesced
                  << std::setw(9) << std::fixed << std::setprecision(2) << (msUncoalesced / msCoalesced) << "x"
                  << std::endl;

        free(h_planar_R);
        free(h_interleaved_RGB);
        checkCudaErrors(cudaFree(d_planar));
        checkCudaErrors(cudaFree(d_interleaved));
    }


    free(full_image_buffer_host);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}


/** TEXTUAL OUTPUT

--- Processing Image File (anna.ppm) ---
Running Coalesced Kernel...
Time for invertRedCoalesced: 1.50403 ms

Running Uncoalesced Kernel...
Time for invertRedUncoalesced: 0.060384 ms


--- Benchmarking Coalesced vs. Uncoalesced ---
  Num PixelsCoalesced (ms)Uncoalesced (ms)  Slowdown
---------------------------------------------------------
     1048576      0.0472        0.0458     0.97x
     2097152      0.0567        0.0754     1.33x
     4194304      0.1514        0.1087     0.72x
     8388608      0.1308        0.1864     1.43x
    16777216      0.2332        0.3509     1.50x
    33554432      0.5128        0.6697     1.31x
    67108864      0.9152        1.3090     1.43x


*/