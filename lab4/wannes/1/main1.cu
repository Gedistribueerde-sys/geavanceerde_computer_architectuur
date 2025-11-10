/* 
 * Code snippet for importing / exporting image data.
 * 
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 * 
 */
#include <cstdint>      // Data types
#include <cstdio>       // File operations
#include <cstdlib>      // malloc, free, exit

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


void save_image_array(uint8_t* image_array, const char filename[] = "./output_image.ppm"){
    /*
     * Save the data of an (RGB) image as a pixel map.
     * 
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     * 
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen(filename,"wb");
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

__global__ void invert_red_coalesced(uint8_t* planar_input, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        planar_input[idx] = 255 - planar_input[idx];
    }
}

__global__ void invert_red_uncoalesced(uint8_t* interleaved_input, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int pixel_index = idx * 3;
        interleaved_input[pixel_index] = 255 - interleaved_input[pixel_index];
    }
}

void interleaved_to_planar(uint8_t* image_array, uint8_t* planar_image){
    for (int i = 0; i < M*N; i++) {
        planar_image[i] = image_array[i*3];    
        planar_image[i + M*N] = image_array[i*3 + 1]; 
        planar_image[i + 2*M*N] = image_array[i*3 + 2];
    }
}

void planar_to_interleaved(uint8_t* planar_image, uint8_t* image_array){
    for (int i = 0; i < M*N; i++) {
        image_array[i*3] = planar_image[i];    
        image_array[i*3 + 1] = planar_image[i + M*N]; 
        image_array[i*3 + 2] = planar_image[i + 2*M*N];
    }
}

// Duplicate image to create larger input (tiles the image)
void create_scaled_image(uint8_t* src, uint8_t* dst, int src_pixels, int scale_factor, bool is_planar) {
    int total_pixels = src_pixels * scale_factor;
    
    if (is_planar) {
        for (int color = 0; color < C; color++) {
            for (int tile = 0; tile < scale_factor; tile++) {
                for (int i = 0; i < src_pixels; i++) {
                    dst[color * total_pixels + tile * src_pixels + i] = src[color * src_pixels + i];
                }
            }
        }
    } else {
        for (int tile = 0; tile < scale_factor; tile++) {
            for (int i = 0; i < src_pixels * C; i++) {
                dst[tile * src_pixels * C + i] = src[i];
            }
        }
    }
}

void benchmark_with_size(uint8_t* image_array, uint8_t* planar_image, int num_pixels, int scale_factor, const char* size_label) {
    printf("\n--- Input Size: %s (%d pixels, %.2f MB) ---\n", size_label, num_pixels, (num_pixels * C) / (1024.0 * 1024.0));
    int num_runs = 100;
    printf("--- Number of runs: %d ---\n", num_runs);

    // Create scaled versions
    uint8_t* scaled_interleaved = (uint8_t*)malloc(num_pixels * C * sizeof(uint8_t));
    uint8_t* scaled_planar = (uint8_t*)malloc(num_pixels * C * sizeof(uint8_t));
    
    create_scaled_image(image_array, scaled_interleaved, M*N, scale_factor, false);
    create_scaled_image(planar_image, scaled_planar, M*N, scale_factor, true);
    
    // Test different thread configurations
    int thread_configs[] = {64, 128, 256, 512, 1024};
    int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
    
    // Allocate device memory
    uint8_t *d_interleaved, *d_planar;
    cudaMalloc((void**)&d_interleaved, num_pixels * C * sizeof(uint8_t));
    cudaMalloc((void**)&d_planar, num_pixels * C * sizeof(uint8_t));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("%-15s %-20s %-20s %-15s\n", "Threads/Block", "Uncoalesced (ms)", "Coalesced (ms)", "Speedup");
    printf("------------------------------------------------------------------------\n");
    
    for (int i = 0; i < num_configs; i++) {
        int threadsPerBlock = thread_configs[i];
        int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;
        
        // Warm-up runs
        cudaMemcpy(d_interleaved, scaled_interleaved, num_pixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);
        invert_red_uncoalesced<<<blocksPerGrid, threadsPerBlock>>>(d_interleaved, num_pixels);
        cudaDeviceSynchronize();
        
        cudaMemcpy(d_planar, scaled_planar, num_pixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);
        invert_red_coalesced<<<blocksPerGrid, threadsPerBlock>>>(d_planar, num_pixels);
        cudaDeviceSynchronize();
        
        // Benchmark uncoalesced kernel
        float total_uncoalesced_time = 0.0f;
        
        for (int run = 0; run < num_runs; run++) {
            cudaMemcpy(d_interleaved, scaled_interleaved, num_pixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            invert_red_uncoalesced<<<blocksPerGrid, threadsPerBlock>>>(d_interleaved, num_pixels);
            cudaEventRecord(stop);
            
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_uncoalesced_time += milliseconds;
        }
        float avg_uncoalesced_time = total_uncoalesced_time / num_runs;
        
        // Benchmark coalesced kernel
        float total_coalesced_time = 0.0f;
        
        for (int run = 0; run < num_runs; run++) {
            cudaMemcpy(d_planar, scaled_planar, num_pixels * C * sizeof(uint8_t), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            invert_red_coalesced<<<blocksPerGrid, threadsPerBlock>>>(d_planar, num_pixels);
            cudaEventRecord(stop);
            
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_coalesced_time += milliseconds;
        }
        float avg_coalesced_time = total_coalesced_time / num_runs;
        
        float speedup = avg_uncoalesced_time / avg_coalesced_time;
        
        printf("%-15d %-20.6f %-20.6f %-15.2fx\n", 
               threadsPerBlock, avg_uncoalesced_time, avg_coalesced_time, speedup);
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_interleaved);
    cudaFree(d_planar);
    free(scaled_interleaved);
    free(scaled_planar);
}

void benchmark_kernels(uint8_t* image_array, uint8_t* planar_image) {
    printf("\n========================================\n");
    printf("PERFORMANCE BENCHMARK: Coalesced vs Uncoalesced Memory Access\n");
    printf("========================================\n");
    printf("Base Image Size: %dx%d pixels\n", M, N);
    
    // Test with original size (1x)
    benchmark_with_size(image_array, planar_image, M*N, 1, "1x (Original)");
    
    // Test with 2x size
    benchmark_with_size(image_array, planar_image, M*N*2, 2, "2x");
    
    // Test with 4x size
    benchmark_with_size(image_array, planar_image, M*N*4, 4, "4x");
    
}


int main (void) {
    
    // Read the image
    uint8_t* image_array = get_image_array();
    
    // Allocate output
    uint8_t* planar_image = (uint8_t*)malloc(M*N*C*sizeof(uint8_t));

    interleaved_to_planar(image_array, planar_image);

    // Run comprehensive benchmark
    benchmark_kernels(image_array, planar_image);

    // Initialize device pointers
    uint8_t *d_input_image;
    cudaMalloc((void**)&d_input_image, M*N*C*sizeof(uint8_t));

    // uncoalesced version
    // Copy input image to device
    cudaMemcpy(d_input_image, image_array, M*N*C*sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (M * N + threadsPerBlock - 1) / threadsPerBlock;
    invert_red_uncoalesced<<<blocksPerGrid, threadsPerBlock>>>(d_input_image, M*N);

    // Copy output image back to host
    cudaMemcpy(image_array, d_input_image, M*N*C*sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Save the image
    save_image_array(image_array, "uncoalesced_output.ppm");

    // coalesced version
    // Copy input image to device
    cudaMemcpy(d_input_image, planar_image, M*N*C*sizeof(uint8_t), cudaMemcpyHostToDevice);
    // Launch kernel
    invert_red_coalesced<<<blocksPerGrid, threadsPerBlock>>>(d_input_image, M*N);
    // Copy output image back to host
    cudaMemcpy(planar_image, d_input_image, M*N*C*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // Save the image
    planar_to_interleaved(planar_image, image_array);
    save_image_array(image_array, "coalesced_output.ppm");

    // Free device memory
    cudaFree(d_input_image);

    free(image_array - OFFSET);
    free(planar_image);
    
    return 0;
}