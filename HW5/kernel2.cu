#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int* d_data, float stepX, float stepY, float lowerX, float lowerY, int count, int pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float c_x = lowerX + thisX * stepX;
    float c_y = lowerY + thisY * stepY;
    float z_x = c_x;
    float z_y = c_y;

    int iter;
    for (iter = 0; iter < count; ++iter){
	if (z_x * z_x + z_y * z_y > 4.f) break;

	float new_x = z_x * z_x - z_y * z_y;
	float new_y = 2.f * z_x * z_y;
	z_x = c_x + new_x;
	z_y = c_y + new_y;
    }

    int* row = (int*)((char*)d_data + thisY * pitch);
    row[thisX] = iter;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int N = resX * resY;
    int size = N * sizeof(int);
    size_t pitch = 0;

    int *data;
    cudaHostAlloc(&data, size, cudaHostAllocMapped);
    int *d_data;
    cudaMallocPitch(&d_data, &pitch, resX * sizeof(int), resY);

    dim3 threadsPerBlock(25, 25);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(d_data, stepX, stepY, lowerX, lowerY, maxIterations, pitch);

    cudaMemcpy2D(data, resX * sizeof(int), d_data, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, data, size);
    cudaFree(d_data);
    cudaFreeHost(data);
}
