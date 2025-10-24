/* 
 * Code snippet for importing / exporting image data.
 * 
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 * 
 */
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h>
#include <chrono>
// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 960       // VR width
#define N 1280       // VR height
#define C 3         // Colors
#define OFFSET 16   // Header length


uint8_t* get_image_array(void){
    /*
     * Get the data of an (RGB) image as a 1D array.
     * 
     * Returns: Flattened image array.
     * 
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the firts row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     * 
     */        
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./anna.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
    
    // Read the image
    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
    
    // Close the file
    fclose(imageFile);
        
    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}


void save_image_array(uint8_t* image_array){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./output_image.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    
    // Configure the file
    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
    
    // Write the image
    fwrite(image_array, 1, M*N*C, imageFile);
    
    // Close the file
    fclose(imageFile);
}

__global__ void convert_image(uint8_t* input_img, uint8_t* output_img, int total_size){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >1 && tid < total_size -2 && tid % 3 == 0){
        output_img[tid] = 0.1 * input_img[tid - 2] + 0.25 * input_img[tid - 1] + 0.5 * input_img[tid] + 0.25 * input_img[tid + 1] + 0.1 * input_img[tid + 2];
    } else if (tid < total_size){
        output_img[tid] = 255 - input_img[tid];
    }


}

int main() {
    // --- Setup image buffers (your code) ---
    uint8_t* image_array = get_image_array(); // assume returns M*N*C bytes
    const size_t img_size = M * N * C * sizeof(uint8_t);

    uint8_t* new_image_array = (uint8_t*)std::malloc(img_size);
    if (!new_image_array) { std::cerr << "malloc failed\n"; return 1; }

    uint8_t *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  img_size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, img_size));
    CUDA_CHECK(cudaMemcpy(d_input, image_array, img_size, cudaMemcpyHostToDevice));

    // --- Decide sweep of threads-per-block, capped by device capability ---
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int maxTPB = prop.maxThreadsPerBlock; // commonly 1024

    std::vector<int> tpb_candidates = {32, 64, 128, 256, 512, 1024};
    tpb_candidates.erase(
        std::remove_if(tpb_candidates.begin(), tpb_candidates.end(),
                       [&](int t){ return t > maxTPB; }),
        tpb_candidates.end());

    // --- Timing setup ---
    const int warmup_runs = 5;
    const int timed_runs  = 30;

    std::cout << "device: " << prop.name << "\n";
    std::cout << "img_size (bytes): " << img_size << "\n";
    std::cout << "TPB, Blocks, Avg_ms, StdDev_ms\n";

    for (int threadsPerBlock : tpb_candidates) {
        const size_t nElems = img_size / sizeof(uint8_t); // number of pixels/bytes
        const int blocksPerGrid = static_cast<int>((nElems + threadsPerBlock - 1) / threadsPerBlock);

        // warm-up (not timed)
        for (int i = 0; i < warmup_runs; ++i) {
            convert_image<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, nElems);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // events for timing
        cudaEvent_t startEvt, stopEvt;
        CUDA_CHECK(cudaEventCreate(&startEvt));
        CUDA_CHECK(cudaEventCreate(&stopEvt));

        std::vector<float> times_ms;
        times_ms.reserve(timed_runs);

        for (int r = 0; r < timed_runs; ++r) {
            CUDA_CHECK(cudaEventRecord(startEvt, 0));
            convert_image<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, nElems);
            CUDA_CHECK(cudaEventRecord(stopEvt, 0));
            CUDA_CHECK(cudaEventSynchronize(stopEvt));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, startEvt, stopEvt)); // milliseconds
            times_ms.push_back(ms);
        }

        CUDA_CHECK(cudaEventDestroy(startEvt));
        CUDA_CHECK(cudaEventDestroy(stopEvt));

        // stats: average + stddev
        const double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        const double avg = sum / times_ms.size();
        double var = 0.0;
        for (float t : times_ms) { double d = t - avg; var += d * d; }
        var /= times_ms.size();
        const double stddev = std::sqrt(var);

        std::cout << threadsPerBlock << ", " << blocksPerGrid << ", "
                  << avg << ", " << stddev << "\n";
    }

    // Optionally copy back once (not timed)
    CUDA_CHECK(cudaMemcpy(new_image_array, d_output, img_size, cudaMemcpyDeviceToHost));
    save_image_array(new_image_array);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    std::free(new_image_array);
    return 0;
}