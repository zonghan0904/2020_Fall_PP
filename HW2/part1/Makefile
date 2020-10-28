CXX=gcc
CFLAG=-Wall -g3
LFLAG=-lpthread
EXE=pi.out
SRC=pi.c

all: ${EXE}

${EXE}:${SRC}
	${CXX} ${CFLAG} $^ -o $@ ${LFLAG}

.PHONY: clean

clean:
	rm -rf ${EXE}
