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


void save_image_array(uint8_t* image_array ,int a){
    /*
     * Save the data of an (RGB) image as a pixel map.
     *
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     *
     */
    // Try opening the file
    FILE *imageFile;
    if (a ==1)imageFile=fopen("./output_image.ppm","wb");
    if (a ==2)imageFile=fopen("./output_image2.ppm","wb");
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
    // Reconstruct the image with the inverted red channel for coalesced version
    uint8_t* output_image_coalesced = (uint8_t*)malloc(M * N * C * sizeof(uint8_t));
    for (int i = 0; i < M * N; ++i) {
        output_image_coalesced[i * C] = red_channel_host[i];       // Inverted Red
        output_image_coalesced[i * C + 1] = image_array_host[i * C + 1]; // Original Green
        output_image_coalesced[i * C + 2] = image_array_host[i * C + 2]; // Original Blue
    }
    save_image_array(output_image_coalesced, 1); // Saves as output_image.ppm

    // --- Reset device memory and prepare for invertRedUncoalesced ---
    // IMPORTANT: Re-extract the original red channel into a temporary host buffer
    //            or directly from image_array_host to copy to the device.
    //            The previous cudaMemcpy copied the *inverted* data.

    // Allocate a temporary host buffer for the original red channel
    uint8_t* original_red_channel_for_device = (uint8_t*)malloc(M * N * sizeof(uint8_t));
    for (int i = 0; i < M * N; ++i) {
        original_red_channel_for_device[i] = image_array_host[i * C]; // Get original Red component
    }

    // Copy the ORIGINAL red channel from this temporary host buffer to device
    cudaMemcpy(red_channel_device, original_red_channel_for_device, M * N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Free the temporary host buffer
    free(original_red_channel_for_device);

    // Now red_channel_device contains the original red channel, ready for uncoalesced inversion.

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

    // Copy the inverted red channel (from uncoalesced) back to host
    cudaMemcpy(red_channel_host, red_channel_device, M * N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Reconstruct the image with the inverted red channel for uncoalesced version
    uint8_t* output_image_uncoalesced = (uint8_t*)malloc(M * N * C * sizeof(uint8_t));
    for (int i = 0; i < M * N; ++i) {
        output_image_uncoalesced[i * C] = red_channel_host[i];       // Inverted Red
        output_image_uncoalesced[i * C + 1] = image_array_host[i * C + 1]; // Original Green
        output_image_uncoalesced[i * C + 2] = image_array_host[i * C + 2]; // Original Blue
    }
    save_image_array(output_image_uncoalesced, 2);

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

 std::cout << "\n--- Vergelijking van coalesced vs. uncoalesced op een willekeurige array ---" << std::endl;

    int randomArraySize = 1 << 10; // 2^10 = 1024
    uint8_t* random_data_host = (uint8_t*)malloc(randomArraySize * sizeof(uint8_t));

    // Vul de array met willekeurige getallen
    srand(time(NULL)); // Initialiseer de random number generator
    for (int i = 0; i < randomArraySize; ++i) {
        random_data_host[i] = rand() % 256; // Willekeurig getal tussen 0 en 255
    }

    uint8_t* random_data_device;
    cudaMalloc(&random_data_device, randomArraySize * sizeof(uint8_t));

    // Kopieer de willekeurige data van host naar device
    cudaMemcpy(random_data_device, random_data_host, randomArraySize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Definieer grid en block dimensies voor de willekeurige array
    int blockSizeRandom = 256;
    int gridSizeRandom = (randomArraySize + blockSizeRandom - 1) / blockSizeRandom;

    // --- Time invertRedCoalesced kernel op willekeurige array ---
    cudaEvent_t startCoalescedRandom, stopCoalescedRandom;
    cudaEventCreate(&startCoalescedRandom);
    cudaEventCreate(&stopCoalescedRandom);

    std::cout << "Running invertRedCoalesced kernel on random array..." << std::endl;
    cudaEventRecord(startCoalescedRandom, 0);
    invertRedCoalesced<<<gridSizeRandom, blockSizeRandom>>>(random_data_device, randomArraySize, 1); // Hoogte is 1 voor 1D array
    cudaEventRecord(stopCoalescedRandom, 0);
    cudaEventSynchronize(stopCoalescedRandom);

    float millisecondsCoalescedRandom = 0;
    cudaEventElapsedTime(&millisecondsCoalescedRandom, startCoalescedRandom, stopCoalescedRandom);
    std::cout << "Time for invertRedCoalesced on random array: " << millisecondsCoalescedRandom << " ms" << std::endl;

    // (Optioneel: kopieer terug naar host om te controleren, maar niet nodig voor tijdmeting)
    // cudaMemcpy(random_data_host, random_data_device, randomArraySize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Reset device geheugen met de originele willekeurige data voor de uncoalesced test
    cudaMemcpy(random_data_device, random_data_host, randomArraySize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // --- Time invertRedUncoalesced kernel op willekeurige array ---
    cudaEvent_t startUncoalescedRandom, stopUncoalescedRandom;
    cudaEventCreate(&startUncoalescedRandom);
    cudaEventCreate(&stopUncoalescedRandom);

    std::cout << "\nRunning invertRedUncoalesced kernel on random array..." << std::endl;
    cudaEventRecord(startUncoalescedRandom, 0);
    invertRedUncoalesced<<<gridSizeRandom, blockSizeRandom>>>(random_data_device, randomArraySize, 1); // Hoogte is 1 voor 1D array
    cudaEventRecord(stopUncoalescedRandom, 0);
    cudaEventSynchronize(stopUncoalescedRandom);

    float millisecondsUncoalescedRandom = 0;
    cudaEventElapsedTime(&millisecondsUncoalescedRandom, startUncoalescedRandom, stopUncoalescedRandom);
    std::cout << "Time for invertRedUncoalesced on random array: " << millisecondsUncoalescedRandom << " ms" << std::endl;

    // Clean up voor de willekeurige array
    cudaFree(random_data_device);
    free(random_data_host);
    cudaEventDestroy(startCoalescedRandom);
    cudaEventDestroy(stopCoalescedRandom);
    cudaEventDestroy(startUncoalescedRandom);
    cudaEventDestroy(stopUncoalescedRandom);


    return 0;
}
