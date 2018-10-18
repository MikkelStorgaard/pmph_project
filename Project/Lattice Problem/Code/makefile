# Get the host
OCC=g++
CC=nvcc

CLINKS= -larmadillo
CFLAGS= -g -std=c++11
#CFLAGS= -g -std=c++0x -O3 -march=native


all : example3D growthCurve3D

example3D : example3D.cpp colonies3D.o
	$(CC) $(CFLAGS) -o example3D example3D.cpp colonies3D.o $(CLINKS)

growthCurve3D : growthCurve3D.cpp colonies3D.o
	$(CC) $(CFLAGS) -o growthCurve3D growthCurve3D.cpp colonies3D.o $(CLINKS)

colonies3D.o : colonies3D.cpp colonies3D.hpp
	$(CC) $(CFLAGS) -c colonies3D.cpp $(CLINKS)


clean :
	rm -f *.o
	rm -f example3D
	rm -f growthCurve3D
