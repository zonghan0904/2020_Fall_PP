#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "../common/bmpfuncs.h"
#include "../common/CycleTimer.h"
#include "helper.h"
#include "hostFE.h"

#define CANNY_LOWER 45
#define CANNY_UPPER 50
#define CANNY_SIGMA 1.0

typedef unsigned char uchar;

int main(void){
     const char *inputFile = "../common/input.bmp";
     const char *outputFile = "output.bmp";
     int imageWidth, imageHeight;
     float *inputImage = readImage(inputFile, &imageWidth, &imageHeight);
     int dataSize = imageWidth * imageHeight * sizeof(float);
     float *outputImage = (float*)malloc(dataSize);

     cl_program program;
     cl_device_id device;
     cl_context context;
     initCL(&device, &context, &program);

     double start_time, end_time;
     double timer = 0;
     double record[10] = {0};

     for (int i = 0; i < 10; i++){
	memset(outputImage, 0, dataSize);
	start_time = currentSeconds();
     	outputImage = canny_edge_detection(inputImage, imageWidth, imageHeight, CANNY_LOWER, CANNY_UPPER, CANNY_SIGMA, &device, &context, &program);
     	end_time = currentSeconds();
	record[i] = end_time - start_time;
     }
     qsort(record, 10, sizeof(double), compare);

     for (int i = 3; i < 7; i++){
	timer += record[i];
     }
     timer /= 4;

     printf("[execution time]: \t\t[%.3f] ms\n\n", timer * 1000);

     storeImage(outputImage, outputFile, imageHeight, imageWidth, inputFile);
     free(outputImage);
}
