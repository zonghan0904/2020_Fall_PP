#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hostFE.h"
#include "helper.h"

/*
 * gaussianFilter: http://www.songho.ca/dsp/cannyedge/cannyedge.html
 * Determine the size of kernel (odd #)
 * 0.0 <= sigma < 0.5 : 3
 * 0.5 <= sigma < 1.0 : 5
 * 1.0 <= sigma < 1.5 : 7
 * 1.5 <= sigma < 2.0 : 9
 * 2.0 <= sigma < 2.5 : 11
 * 2.5 <= sigma < 3.0 : 13 ...
 * kernel size = 2 * int(2 * sigma) + 3;
 */
void gaussian_kernel(float *kernel,
		     const int n,
		     const float mean,
		     const float sigma)
{
  int i, j;
  size_t c = 0;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      kernel[c++] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);
  }
}

/*
 * Links:
 * http://en.wikipedia.org/wiki/Canny_edge_detector
 * http://www.tomgibara.com/computer-vision/CannyEdgeDetector.java
 * http://fourier.eng.hmc.edu/e161/lectures/canny/node1.html
 * http://www.songho.ca/dsp/cannyedge/cannyedge.html
 *
 * Note: T1 and T2 are lower and upper thresholds.
 */
float * canny_edge_detection(const float    *in,
                             const int      width,
		             const int      height,
			     const int      t1,
			     const int      t2,
			     const float    sigma,
			     cl_device_id   *device,
			     cl_context	    *context,
			     cl_program	    *program)
{
  cl_int status;
  float *retval;
  int dataSize = width * height * sizeof(float);
  int zero = 0;
  int one = 1;
  int three = 3;
  const int n = 2 * (int) (2 * sigma) + 3;
  const float mean = (float) floor(n / 2.0);

  size_t localws[2] = {32, 32};
  size_t globalws[2] = {width, height};

  const float Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const float Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  cl_command_queue myqueue;
  myqueue = clCreateCommandQueue(*context, *device, 0, &status);

  // /* allocate device memory */
  cl_mem d_pixels = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
  cl_mem d_out = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
  cl_mem d_Gx = clCreateBuffer(*context, CL_MEM_READ_ONLY, 9 * sizeof(float), NULL, &status);
  cl_mem d_Gy = clCreateBuffer(*context, CL_MEM_READ_ONLY, 9 * sizeof(float), NULL, &status);
  cl_mem d_after_Gx = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
  cl_mem d_after_Gy = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
  cl_mem d_G = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
  cl_mem d_nms = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
  cl_mem d_kernel = clCreateBuffer(*context, CL_MEM_READ_ONLY, n * n * sizeof(float), NULL, &status);
  cl_mem d_thre = clCreateBuffer(*context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
  cl_mem d_hyster = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);

  // /* allocate host memory */
  float *kernel = (float*)malloc(n * n * sizeof(float));

  // /* copy input data from host to device */
  status = clEnqueueWriteBuffer(myqueue, d_pixels, CL_TRUE, 0, dataSize, (void *)in, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");
  status = clEnqueueWriteBuffer(myqueue, d_Gx, CL_TRUE, 0, 9 * sizeof(float), (void *)Gx, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");
  status = clEnqueueWriteBuffer(myqueue, d_Gy, CL_TRUE, 0, 9 * sizeof(float), (void *)Gy, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");

  // /* Gaussian Filter */
  gaussian_kernel(kernel, n, mean, sigma);
  status = clEnqueueWriteBuffer(myqueue, d_kernel, CL_TRUE, 0, n * n * sizeof(float), (void *)kernel, 0, NULL, NULL);
  CHECK(status, "clEnqueueWriteBuffer");
  cl_kernel conv_kernel = clCreateKernel(*program, "convolution", status);
  CHECK(status, "clCreateKernel");
  clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), (void *)&d_pixels);
  clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), (void *)&d_out);
  clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), (void *)&d_kernel);
  clSetKernelArg(conv_kernel, 3, sizeof(cl_int), (void *)&width);
  clSetKernelArg(conv_kernel, 4, sizeof(cl_int), (void *)&height);
  clSetKernelArg(conv_kernel, 5, sizeof(cl_int), (void *)&n);
  clSetKernelArg(conv_kernel, 6, sizeof(cl_int), (void *)&one);
  status = clEnqueueNDRangeKernel(myqueue, conv_kernel, 2, 0, globalws, localws, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");

  // /* Sobel Filter */
  clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), (void *)&d_out);
  clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), (void *)&d_after_Gx);
  clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), (void *)&d_Gx);
  clSetKernelArg(conv_kernel, 5, sizeof(cl_int), (void *)&three);
  clSetKernelArg(conv_kernel, 6, sizeof(cl_int), (void *)&zero);
  status = clEnqueueNDRangeKernel(myqueue, conv_kernel, 2, 0, globalws, localws, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");
  clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), (void *)&d_after_Gy);
  clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), (void *)&d_Gy);
  status = clEnqueueNDRangeKernel(myqueue, conv_kernel, 2, 0, globalws, localws, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");
  cl_kernel sobel_kernel = clCreateKernel(*program, "calculate_G", status);
  CHECK(status, "clCreateKernel");
  clSetKernelArg(sobel_kernel, 0, sizeof(cl_mem), (void *)&d_G);
  clSetKernelArg(sobel_kernel, 1, sizeof(cl_mem), (void *)&d_after_Gx);
  clSetKernelArg(sobel_kernel, 2, sizeof(cl_mem), (void *)&d_after_Gy);
  clSetKernelArg(sobel_kernel, 3, sizeof(cl_int), (void *)&width);
  clSetKernelArg(sobel_kernel, 4, sizeof(cl_int), (void *)&height);
  status = clEnqueueNDRangeKernel(myqueue, sobel_kernel, 2, 0, globalws, localws, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");

  // /* Non-maximum suppression */
  cl_kernel nms_kernel = clCreateKernel(*program, "non_maximum_sup", status);
  CHECK(status, "clCreateKernel");
  clSetKernelArg(nms_kernel, 0, sizeof(cl_mem), (void *)&d_nms);
  clSetKernelArg(nms_kernel, 1, sizeof(cl_mem), (void *)&d_G);
  clSetKernelArg(nms_kernel, 2, sizeof(cl_mem), (void *)&d_after_Gx);
  clSetKernelArg(nms_kernel, 3, sizeof(cl_mem), (void *)&d_after_Gy);
  clSetKernelArg(nms_kernel, 4, sizeof(cl_int), (void *)&width);
  clSetKernelArg(nms_kernel, 5, sizeof(cl_int), (void *)&height);
  status = clEnqueueNDRangeKernel(myqueue, nms_kernel, 2, 0, globalws, localws, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");

  // /* threshold */
  cl_kernel thre_kernel = clCreateKernel(*program, "threshold", status);
  CHECK(status, "clCreateKernel");
  clSetKernelArg(thre_kernel, 0, sizeof(cl_mem), (void *)&d_nms);
  clSetKernelArg(thre_kernel, 1, sizeof(cl_mem), (void *)&d_thre);
  clSetKernelArg(thre_kernel, 2, sizeof(cl_int), (void *)&width);
  clSetKernelArg(thre_kernel, 3, sizeof(cl_int), (void *)&t2);
  status = clEnqueueNDRangeKernel(myqueue, thre_kernel, 2, 0, globalws, localws, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");

  // /* hystersis */
  cl_kernel hyster_kernel = clCreateKernel(*program, "hysteresis", status);
  CHECK(status, "clCreateKernel");
  clSetKernelArg(hyster_kernel, 0, sizeof(cl_mem), (void *)&d_nms);
  clSetKernelArg(hyster_kernel, 1, sizeof(cl_mem), (void *)&d_thre);
  clSetKernelArg(hyster_kernel, 2, sizeof(cl_mem), (void *)&d_hyster);
  clSetKernelArg(hyster_kernel, 3, sizeof(cl_int), (void *)&width);
  clSetKernelArg(hyster_kernel, 4, sizeof(cl_int), (void *)&height);
  clSetKernelArg(hyster_kernel, 5, sizeof(cl_int), (void *)&t1);
  clSetKernelArg(hyster_kernel, 6, sizeof(cl_int), (void *)&t2);
  status = clEnqueueNDRangeKernel(myqueue, hyster_kernel, 2, 0, globalws, localws, 0, NULL, NULL);
  CHECK(status, "clEnqueueNDRangeKernel");

  // /* copy output data from device to host */
  retval = (float*)malloc(dataSize);
  status = clEnqueueReadBuffer(myqueue, d_hyster, CL_TRUE, 0, dataSize, (void *)retval, NULL, NULL, NULL);
  CHECK(status, "clEnqueueReadBuffer");

  /* deallocate both CPU's and GPU's memory */
  status = clReleaseCommandQueue(myqueue);
  status = clReleaseMemObject(d_pixels);
  status = clReleaseMemObject(d_out);
  status = clReleaseMemObject(d_Gx);
  status = clReleaseMemObject(d_Gy);
  status = clReleaseMemObject(d_after_Gx);
  status = clReleaseMemObject(d_after_Gy);
  status = clReleaseMemObject(d_G);
  status = clReleaseMemObject(d_nms);
  status = clReleaseMemObject(d_kernel);
  status = clReleaseMemObject(d_thre);
  status = clReleaseMemObject(d_hyster);
  status = clReleaseKernel(conv_kernel);
  status = clReleaseKernel(sobel_kernel);
  status = clReleaseKernel(nms_kernel);
  status = clReleaseKernel(thre_kernel);
  status = clReleaseKernel(hyster_kernel);
  free(kernel);

  return retval;
}
