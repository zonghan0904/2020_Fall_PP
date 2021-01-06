CC = gcc
CFLAG = -O3 -I../common -I./ -m64 -ffloat-store
LFLAG = -lm -lOpenCL

EXE = opencl
SRC = opencl.c

SRC += ../common/bmpfuncs.c
SRC += ./hostFE.c
SRC += ./helper.c

OBJ = ${SRC:.c=.o}

all: ${EXE}

${EXE}: ${OBJ}
	$(CC) -o $@ $(OBJ) ${LFLAG}

%.o: %.c
	${CC} ${CFLAG} -c $^ -o $@

.PHONY: clean

clean:
	rm -rf ${EXE} ${OBJ} *.bmp
