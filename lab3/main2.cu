#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h>
#include <vector>       // For testing different thread counts
#include <string>       // For titles
#include <numeric>      // For std::accumulate (though manual sum is fine)

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

    if (tid >= 6 && tid < total_size - 6 && tid % 3 == 0) {
        output_img[tid] = 0.1 * input_img[tid - 6] +
                          0.25 * input_img[tid - 3] +
                          0.5 * input_img[tid] +
                          0.25 * input_img[tid + 3] +
                          0.1 * input_img[tid + 6];
    
    } else if (tid < total_size) {
        output_img[tid] = 255 - input_img[tid];
    }
}

__global__ void convert_image_planar(uint8_t* input_img, uint8_t* output_img, int num_pixels) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = num_pixels * 3;

    if (tid >= total_elements) return;

    if (tid < num_pixels) {
        if (tid >= 2 && tid < num_pixels - 2) {
            output_img[tid] = 0.1 * input_img[tid - 2] + 0.25 * input_img[tid - 1] + 
                              0.5 * input_img[tid] + 0.25 * input_img[tid + 1] + 
                              0.1 * input_img[tid + 2];
        } else {
            // Just copy the edges
            output_img[tid] = input_img[tid];
        }
    } else {

        output_img[tid] = 255 - input_img[tid];
    }
}



void run_divergent_performance_test(
                          const std::string& title, 
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

        for (int i = 0; i < num_iterations; ++i) {
            cudaEventRecord(start);
            
            convert_image<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, total_elements);
            
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

void run_planar_performance_test(
                          const std::string& title, 
                          const std::vector<int>& thread_counts,
                          uint8_t* d_input, 
                          uint8_t* d_output, 
                          int total_elements,
                          int num_pixels, 
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

        for (int i = 0; i < num_iterations; ++i) {
            cudaEventRecord(start);
            
            convert_image_planar<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, num_pixels);
            
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
    
    const int NUM_ITERATIONS = 100;

    uint8_t* image_array = get_image_array();

    int num_pixels = M * N;
    int total_elements = M * N * C;
    size_t img_size_bytes = total_elements * sizeof(uint8_t);
    
    uint8_t* planar_format = (uint8_t*)malloc(img_size_bytes);
    if (planar_format == NULL) {
        perror("ERROR: Cannot allocate host planar memory");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_pixels; i++){
        planar_format[i] = image_array[i*3];
        planar_format[i + num_pixels] = image_array[i*3+1];
        planar_format[i + 2*num_pixels] = image_array[i*3+2];
    }
    
    uint8_t* new_image_array = (uint8_t*)malloc(img_size_bytes);
    if (new_image_array == NULL) {
        perror("ERROR: Cannot allocate host output memory");
        exit(EXIT_FAILURE);
    }
    uint8_t* planar_output_array = (uint8_t*)malloc(img_size_bytes);


    uint8_t *d_input, *d_output;
    cudaMalloc((void**)&d_input, img_size_bytes);
    cudaMalloc((void**)&d_output, img_size_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<int> threads_full_warp = {32, 64, 128, 256, 512, 1024};
    std::vector<int> threads_partial_warp = {16, 20, 100, 200, 400};

    std::cout << "Image size: " << M << "x" << N << std::endl;

    cudaMemcpy(d_input, image_array, img_size_bytes, cudaMemcpyHostToDevice);

    run_divergent_performance_test("Performance Full Warps (RGB)", 
        threads_full_warp, 
        d_input, d_output, 
        total_elements, 
        start, stop,
        NUM_ITERATIONS);

    run_divergent_performance_test("Performance Partial Warps (RGB)", 
        threads_partial_warp, 
        d_input, d_output, 
        total_elements, 
        start, stop,
        NUM_ITERATIONS);

    std::cout << "\nRunning planar" << std::endl;
    cudaMemcpy(d_input, planar_format, img_size_bytes, cudaMemcpyHostToDevice);

    run_planar_performance_test("Performance Full Warps (Planar)",
        threads_full_warp,
        d_input, d_output,
        total_elements, num_pixels,
        start, stop,
        NUM_ITERATIONS);

    run_planar_performance_test("Performance Partial Warps (Planar)",
        threads_partial_warp,
        d_input, d_output,
        total_elements, num_pixels,
        start, stop,
        NUM_ITERATIONS);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(planar_output_array, d_output, img_size_bytes, cudaMemcpyDeviceToHost);
    

    for (int i = 0; i < num_pixels; i++){
        new_image_array[i*3]   = planar_output_array[i];
        new_image_array[i*3+1] = planar_output_array[i + num_pixels];
        new_image_array[i*3+2] = planar_output_array[i + 2*num_pixels];
    }
    
    save_image_array(new_image_array);

    cudaFree(d_input);
    cudaFree(d_output);
    free(new_image_array);
    free(planar_output_array);
    free(planar_format);
    free(image_array - OFFSET);
    
    return 0;
}