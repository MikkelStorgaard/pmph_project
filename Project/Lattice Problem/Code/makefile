# Get the host
CC=g++

CLINKS= -larmadillo -fopenmp
CFLAGS= -g -std=c++0x -march=native
#CFLAGS= -g -std=c++0x -O3 -march=native


all : example3D

example3D : example3D.cpp colonies3D.o
	$(CC) $(CFLAGS) -o example3D example3D.cpp colonies3D.o $(CLINKS)

colonies3D.o : colonies3D.cpp colonies3D.hpp
	$(CC) $(CFLAGS) -c colonies3D.cpp $(CLINKS)


clean :
	rm -f *.o
	rm -f example3D
