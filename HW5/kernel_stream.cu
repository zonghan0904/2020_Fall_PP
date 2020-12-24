#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int* d_data, int width, float stepX, float stepY, float lowerX, float lowerY, int count, int index, int ChunkSize) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = thisX + thisY * width;

    if (idx / ChunkSize == index){
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

    	d_data[idx % ChunkSize] = iter;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int N = resX * resY;
    int size = N * sizeof(int);
    const int nStreams = 8;

    int *data;
    cudaHostAlloc(&data, size, cudaHostAllocMapped);
    int *d_data;
    cudaMalloc(&d_data, size);

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
	cudaStreamCreate(&streams[i]);
    }

    const int ChunkSize = size / nStreams;
    int offset = 0;
    dim3 threadsPerBlock(25, 25);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    for(int i = 0; i < nStreams; i++){
	offset = ChunkSize * i;
	mandelKernel<<<numBlocks, threadsPerBlock>>>(d_data + offset, resX, stepX, stepY, lowerX, lowerY, maxIterations, i, ChunkSize);
	cudaMemcpyAsync(data + offset, d_data + offset, ChunkSize, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();
    memcpy(img, data, size);
    for (int i = 0; i < nStreams; i++){
	cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);
    cudaFreeHost(data);
}
