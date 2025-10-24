#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h>
#include <vector>       // For testing different thread counts

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

__global__ void invert_image(uint8_t* input_img, uint8_t* output_img, int total_size){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < total_size){
        output_img[tid] = 255 - input_img[tid];
    }
}

__global__ void invert_image_stride(uint8_t* input_img, uint8_t* output_img, int total_size){
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < total_size; i += stride) {
        output_img[i] = 255 - input_img[i];
    }
}

void run_performance_test(const std::string& title, 
                          const std::vector<int>& thread_counts,
                          uint8_t* d_input, 
                          uint8_t* d_output, 
                          int total_elements, 
                          cudaEvent_t& start, 
                          cudaEvent_t& stop,
                          int num_iterations)
{
    std::cout << "--- " << title << " (Avg. over " << num_iterations << " runs) ---" << std::endl;
    std::cout << "Threads/Block | Blocks/Grid | Avg. Time (ms)" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    for (int threadsPerBlock : thread_counts) {
        
        int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
        
        double total_milliseconds = 0.0;

        // for loop for averaging
        for (int i = 0; i < num_iterations; ++i) {
            cudaEventRecord(start);
            
            invert_image<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, total_elements);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_milliseconds += milliseconds;
        }

        float avg_milliseconds = (float)(total_milliseconds / num_iterations);
        
        printf("%-13d | %-11d | %f\n", threadsPerBlock, blocksPerGrid, avg_milliseconds);
    }
    
    std::cout << "-----------------------------------------------\n" << std::endl;
}


void run_striding_performance_test(const std::string& title, 
                          const std::vector<int>& thread_counts,
                          uint8_t* d_input, 
                          uint8_t* d_output, 
                          int total_elements, 
                          cudaEvent_t& start, 
                          cudaEvent_t& stop,
                          int num_iterations)
{
    std::cout << "--- " << title << " (Avg. over " << num_iterations << " runs) ---" << std::endl;
    std::cout << "Threads/Block | Blocks/Grid | Avg. Time (ms)" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    for (int threadsPerBlock: thread_counts) {
        int numThreads = total_elements / 16;
        int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

        double total_milliseconds = 0.0;

        for (int i = 0; i < num_iterations; ++i) {
            cudaEventRecord(start);

            invert_image_stride<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, total_elements);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0; 
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_milliseconds += milliseconds;
        }

        float avg_milliseconds = (float)(total_milliseconds / num_iterations);

        printf("%-13d | %-11d | %f\n", threadsPerBlock, blocksPerGrid, avg_milliseconds);
    }

    std::cout << "-----------------------------------------------\n" << std::endl;
}

int main (void) {
    
    // Read the image
    uint8_t* image_array = get_image_array();
    
    // Allocate host output memory
    uint8_t* new_image_array = (uint8_t*)malloc(M*N*C);
    if (new_image_array == NULL) {
        perror("ERROR: Cannot allocate host output memory");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    uint8_t *d_input, *d_output;
    size_t img_size = M * N * C * sizeof(uint8_t);
    int total_elements = M * N * C;

    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);

    // Copy image from host to device
    cudaMemcpy(d_input, image_array, img_size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Define the different thread counts
    std::vector<int> threads_full_warp = {32, 64, 128, 256, 512, 1024};
    std::vector<int> threads_partial_warp = {16, 20, 100, 200, 400};

    // Testing execution times
    std::cout << "Image size: " << M << "x" << N << std::endl << std::endl;

    int num_iterations = 100;
    run_performance_test("Performance Full Warps", 
        threads_full_warp, 
        d_input, d_output, 
        total_elements, 
        start, stop, num_iterations);

    run_performance_test("Performance Partial Warps", 
        threads_partial_warp, 
        d_input, d_output, 
        total_elements, 
        start, stop, num_iterations);

    run_striding_performance_test("Performance Full Warps - Striding",
        threads_full_warp,
        d_input, d_output,
        total_elements,
        start, stop, num_iterations);

    run_striding_performance_test("Performance Partial Warps - Striding",
        threads_partial_warp,
        d_input, d_output,
        total_elements,
        start, stop, num_iterations);

   
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the final result (from the last run) back to host
    cudaMemcpy(new_image_array, d_output, img_size, cudaMemcpyDeviceToHost);
    
    // Save the inverted image
    save_image_array(new_image_array);

    // Free all allocated memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(new_image_array);
    free(image_array - OFFSET); // Correctly free the original image pointer
    
    return 0;
}