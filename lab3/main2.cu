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


int main (void) {
    
    // Read the image
    uint8_t* image_array = get_image_array();
    
    // Allocate output
    uint8_t* new_image_array = (uint8_t*)malloc(M*N*C);

    uint8_t *d_input, *d_output;
    size_t img_size = M * N * C * sizeof(uint8_t);

    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);

    cudaMemcpy(d_input, image_array, img_size, cudaMemcpyHostToDevice);
    
    // Convert to grayscale using only the red color component
    // for(int i=0; i<M*N*C; i++){
    //     new_image_array[i] = image_array[i/3*3];
    // }

    int threadsPerBlock = 256;
    int blocksPerGrid = (M * N * C + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Num of threads: " << threadsPerBlock << " Num of blocks: " << blocksPerGrid << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    convert_image<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, img_size);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    std::cout << "Kernel execution time: " << elapsed.count() << " ms" << std::endl;
    cudaMemcpy(new_image_array, d_output, img_size, cudaMemcpyDeviceToHost);

    
    // Save the image
    save_image_array(new_image_array);


    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}