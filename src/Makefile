objects = main.o coulomb.o jacobi_eigen.o submatrix.o

all: $(objects)
	nvcc $(objects) -o app

%.o: %.cpp
	nvcc -x cu -I. -dc $< -o $@

clean:
	rm -f *.o app
