CXX=g++
CFLAG=-Wall -g
EXE=pi.out
SRC=pi.cpp

all: ${EXE}

${EXE}:${SRC}
	${CXX} ${CFLAG} $^ -o $@

.PHONY: clean

clean:
	rm -rf ${EXE}
