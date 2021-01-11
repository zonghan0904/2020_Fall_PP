# Canny Edge Detection Performance Comparison Between Parallel Programming and Serial Programming
Implementation of Canny Edge Detection in serial, Pthread, OpenMP, CUDA and OpenCL.

## Platform
* CPU: Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz
* GPU: NVIDIA GTX 1060 6GB
* OS: Ubuntu 18.04
* Others: g++-10, clang++-11, libomp5, cuda10.2 and OpenCL1.1 have been installed

## Build
Changing directory to any directory which is named corresponding implementation in this repository.
```
$ make [clean]
```
Use clean if you want to clean build.

## Example
### For serial implementation
`$ cd Serial`
`$ make`
`$ ./serial`

### For CUDA implementation
`$ cd CUDA`
`$ make`
`$ ./cuda`

## Result
It should output the average execution time of corresponding implementation.
Also, it will generate an output image of Canny Edge Detection result.

![](https://user-images.githubusercontent.com/40656204/104148959-658e4f80-540f-11eb-9534-512cd45bebec.jpg)

