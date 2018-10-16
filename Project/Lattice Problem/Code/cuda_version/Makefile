# Get the host
CC=nvcc

#CLINKS= -larmadillo -fopenmp
#CFLAGS= -g -std=c++0x -march=native
#CFLAGS= -g -std=c++0x -O3 -march=native


all : example3D growthCurve3D

example3D : example3D.cu colonies3D.o
	$(CC) -o example3D example3D.cu colonies3D.o $(CLINKS)

growthCurve3D : growthCurve3D.cpp colonies3D.o
	$(CC) -o growthCurve3D growthCurve3D.cpp colonies3D.o $(CLINKS)

colonies3D.o : colonies3D.cu colonies3D.hpp
	$(CC) -c colonies3D.cu


clean :
	rm -f *.o
	rm -f example3D
	rm -f growthCurve3D
