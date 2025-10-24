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

/ ---- MAIN ----
int main() {
    const int totalPixels = M * N * C;
    const size_t totalBytes = totalPixels * sizeof(uint8_t);

    // lees invoerbeeld
    uint8_t* h_in = get_image_array();
    uint8_t* h_out = (uint8_t*)malloc(totalBytes);

    // device geheugen
    uint8_t *d_in, *d_out;
    cudaCheck(cudaMalloc(&d_in, totalBytes));
    cudaCheck(cudaMalloc(&d_out, totalBytes));
    cudaCheck(cudaMemcpy(d_in, h_in, totalBytes, cudaMemcpyHostToDevice));

    // events voor timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;

    // meet uitvoeringstijd
    cudaEventRecord(start);
    convert_image<<<blocks, threads>>>(d_in, d_out, N, M);
    cudaEventRecord(stop);
    cudaCheck(cudaEventSynchronize(stop));

    float ms = 0;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Thread divergence kernel time: " << ms << " ms\n";

    // kopieer resultaat terug
    cudaCheck(cudaMemcpy(h_out, d_out, totalBytes, cudaMemcpyDeviceToHost));

    // opslaan
    save_image_array(h_out, "output_divergence.ppm");

    // cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Output saved as output_divergence.ppm\n";
    return 0;
}