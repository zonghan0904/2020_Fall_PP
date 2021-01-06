NVCC = nvcc
CC = gcc
CFLAG = -O3 -I../common -I./
LFLAG = -lm
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

EXE = cuda
SRC = cuda.c

SRC += ../common/bmpfuncs.c

OBJ = ${SRC:.c=.o} kernel.o

all: ${EXE}

${EXE}: ${OBJ}
	$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJ)

%.o: %.c
	${CC} ${CFLAG} -c $^ -o $@

kernel.o: kernel.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel.cu -o $@

.PHONY: clean

clean:
	rm -rf ${EXE} ${OBJ} *.bmp
