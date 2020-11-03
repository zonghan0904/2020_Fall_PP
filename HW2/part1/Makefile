CXX=gcc
CFLAG=-fPIC -std=c99 -g3 -O3 -I. -I${INC} -mavx2 -march=native -Wall -Wextra -pedantic -Wshadow
LFLAG=-lpthread
EXE=pi.out
SRC=pi.c

SRC+=./SIMDxorshift/src/simdxorshift128plus.c
SRC+=./SIMDxorshift/src/xorshift128plus.c

INC=./SIMDxorshift/include

OBJ=${SRC:.c=.o}

all: ${EXE}

${EXE}:${OBJ}
	${CXX} ${CFLAG} $^ -o $@ ${LFLAG}

%.o:%.c
	${CXX} ${CFLAG} -c $^ -o $@

.PHONY: clean

clean:
	rm -rf ${EXE} *.out ${OBJ}
