/*
 * Code snippet for importing / exporting image data.
 *
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 *
 */
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h> // CUDA Runtime API

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

__global__ void invertRedCoalesced(unsigned char* redChannel, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    if (idx < size) {
        redChannel[idx] = 255 - redChannel[idx];  // Invert red component
    }
}

__global__ void invertRedUncoalesced(unsigned char* redChannel, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        redChannel[i] = 255 - redChannel[i];  // Same inversion but uncoalesced access pattern
    }
}

int main (void) {

    // Read the image
    uint8_t* image_array_host = get_image_array();

    // Allocate memory for the red channel on the host
    uint8_t* red_channel_host = (uint8_t*)malloc(M * N * sizeof(uint8_t));

    // Extract the red channel from the interleaved RGB array
    for (int i = 0; i < M * N; ++i) {
        red_channel_host[i] = image_array_host[i * C]; // Red component is at index 0, 3, 6, ...
    }

    // Allocate device memory for the red channel
    uint8_t* red_channel_device;
    cudaMalloc(&red_channel_device, M * N * sizeof(uint8_t));

    // Copy red channel from host to device
    cudaMemcpy(red_channel_device, red_channel_host, M * N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (M * N + blockSize - 1) / blockSize;

    // --- Time invertRedCoalesced kernel ---
    cudaEvent_t startCoalesced, stopCoalesced;
    cudaEventCreate(&startCoalesced);
    cudaEventCreate(&stopCoalesced);

    std::cout << "Running invertRedCoalesced kernel..." << std::endl;
    cudaEventRecord(startCoalesced, 0);
    invertRedCoalesced<<<gridSize, blockSize>>>(red_channel_device, M, N);
    cudaEventRecord(stopCoalesced, 0);
    cudaEventSynchronize(stopCoalesced);

    float millisecondsCoalesced = 0;
    cudaEventElapsedTime(&millisecondsCoalesced, startCoalesced, stopCoalesced);
    std::cout << "Time for invertRedCoalesced: " << millisecondsCoalesced << " ms" << std::endl;

    // Copy the inverted red channel back to host to save it or use for other operations
    // Note: If you want to compare coalesced vs uncoalesced, you'll need to reset red_channel_device
    // or use a separate device array for the uncoalesced version.
    cudaMemcpy(red_channel_host, red_channel_device, M * N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Reconstruct the image with the inverted red channel for coalesced version
    uint8_t* output_image_coalesced = (uint8_t*)malloc(M * N * C * sizeof(uint8_t));
    for (int i = 0; i < M * N; ++i) {
        output_image_coalesced[i * C] = red_channel_host[i];       // Inverted Red
        output_image_coalesced[i * C + 1] = image_array_host[i * C + 1]; // Original Green
        output_image_coalesced[i * C + 2] = image_array_host[i * C + 2]; // Original Blue
    }
    save_image_array(output_image_coalesced); // Saves as output_image.ppm

    // --- Reset device memory and prepare for invertRedUncoalesced ---
    // Copy original red channel from host to device again
    cudaMemcpy(red_channel_device, red_channel_host, M * N * sizeof(uint8_t), cudaMemcpyHostToDevice);


    // --- Time invertRedUncoalesced kernel ---
    cudaEvent_t startUncoalesced, stopUncoalesced;
    cudaEventCreate(&startUncoalesced);
    cudaEventCreate(&stopUncoalesced);

    std::cout << "\nRunning invertRedUncoalesced kernel..." << std::endl;
    cudaEventRecord(startUncoalesced, 0);
    invertRedUncoalesced<<<gridSize, blockSize>>>(red_channel_device, M, N);
    cudaEventRecord(stopUncoalesced, 0);
    cudaEventSynchronize(stopUncoalesced);

    float millisecondsUncoalesced = 0;
    cudaEventElapsedTime(&millisecondsUncoalesced, startUncoalesced, stopUncoalesced);
    std::cout << "Time for invertRedUncoalesced: " << millisecondsUncoalesced << " ms" << std::endl;

    // Copy the inverted red channel back to host
    cudaMemcpy(red_channel_host, red_channel_device, M * N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Reconstruct the image with the inverted red channel for uncoalesced version
    uint8_t* output_image_uncoalesced = (uint8_t*)malloc(M * N * C * sizeof(uint8_t));
    for (int i = 0; i < M * N; ++i) {
        output_image_uncoalesced[i * C] = red_channel_host[i];       // Inverted Red
        output_image_uncoalesced[i * C + 1] = image_array_host[i * C + 1]; // Original Green
        output_image_uncoalesced[i * C + 2] = image_array_host[i * C + 2]; // Original Blue
    }
    // You might want to save this to a different file name, e.g., "output_uncoalesced.ppm"
    // For now, it will overwrite "output_image.ppm" or you can comment this out.
    // save_image_array(output_image_uncoalesced);


    // Clean up
    cudaFree(red_channel_device);
    free(red_channel_host);
    free(image_array_host);
    free(output_image_coalesced);
    free(output_image_uncoalesced);
    cudaEventDestroy(startCoalesced);
    cudaEventDestroy(stopCoalesced);
    cudaEventDestroy(startUncoalesced);
    cudaEventDestroy(stopUncoalesced);

    return 0;
}
