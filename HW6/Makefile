default: conv

NVCC = nvcc
CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
ifdef USE_CUDA
	$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) -lOpenCL
else
	$(CC) -o $@ $(OBJS) $(FLAGS)
endif

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

hostFE.o: hostFE.cu hostFE.h
ifdef USE_CUDA
	${NVCC} ${CUDA_COMPILE_FLAGS} -c hostFE.cu -o $@
endif

clean:
	rm -f conv *.o output.bmp ref.bmp
