# Get the host
CC=nvcc

CFLAGS= -std=c++11 -lcurand


all : example3D benchmark

example3D : example3D.cu colonies3D.o
	$(CC) -o example3D example3D.cu colonies3D.o

benchmark : benchmark.cu
	$(CC) -o benchmark benchmark.cu

colonies3D.o : colonies3D.cu colonies3D.hpp
	$(CC) -c colonies3D.cu


clean :
	rm -f *.o
	rm -f example3D
	rm -f benchmark
