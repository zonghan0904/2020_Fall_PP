#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int mem_size = imageHeight * imageWidth * sizeof(float);

    // create command queue
    cl_command_queue myqueue;
    myqueue = clCreateCommandQueue(*context, *device, 0, &status);

    // allocate device memory
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
    cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY, mem_size, NULL, &status);
    cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, mem_size, NULL, &status);

    // copy data from host memory to device memory
    status = clEnqueueWriteBuffer(myqueue, d_filter, CL_TRUE, 0, filterSize, (void *)filter, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer");
    status = clEnqueueWriteBuffer(myqueue, d_inputImage, CL_TRUE, 0, mem_size, (void *)inputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer");

    // create kernel function
    cl_kernel mykernel = clCreateKernel(*program, "convolution", status);
    CHECK(status, "clCreateKernel");

    // set kernel function args
    clSetKernelArg(mykernel, 0, sizeof(cl_int), (void *)&filterWidth);
    clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void *)&d_filter);
    clSetKernelArg(mykernel, 2, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(mykernel, 4, sizeof(cl_mem), (void *)&d_inputImage);
    clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void *)&d_outputImage);

    // workgroups parameter
    size_t localws[2] = {25, 25};
    size_t globalws[2] = {imageWidth, imageHeight};

    // execute kernel function
    status = clEnqueueNDRangeKernel(myqueue, mykernel, 2, 0, globalws, localws, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // copy data from device memory to host memory
    status = clEnqueueReadBuffer(myqueue, d_outputImage, CL_TRUE, 0, mem_size, (void *)outputImage, NULL, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");

    // release opencl object
    status = clReleaseCommandQueue(myqueue);
    status = clReleaseMemObject(d_filter);
    status = clReleaseMemObject(d_inputImage);
    status = clReleaseMemObject(d_outputImage);
    status = clReleaseKernel(mykernel);
}
