CXX=mpicxx
SRC=main.cc
SRC+= matrix.cc

EXE=matmul

OBJ=${SRC:.cc=.o}

all: ${EXE}

${EXE}: ${OBJ}
	${CXX} -o $@ $^

%.o: %.cc
	${CXX} -c $^ -o $@

.PHONY: clean

clean:
	rm -rf ${OBJ} ${EXE}
