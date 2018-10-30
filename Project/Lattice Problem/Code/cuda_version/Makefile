# Get the host
CC=nvcc

#CLINKS= -larmadillo -fopenmp
#CFLAGS= -g -std=c++0x -march=native
CFLAGS= -std=c++11 -lcurand


all : example3D test growthCurve3D

example3D : example3D.cu colonies3D.o
	$(CC) $(CFLAGS) -o example3D example3D.cu colonies3D.o $(CLINKS)

test : example3D.cu colonies_new.o
	$(CC) $(CFLAGS) -o test example3D.cu colonies_new.o $(CLINKS)

growthCurve3D : growthCurve3D.cpp colonies3D.o
	$(CC) $(CFLAGS) -o growthCurve3D growthCurve3D.cpp colonies3D.o $(CLINKS)

colonies3D.o : colonies3D.cu colonies3D.hpp
	$(CC) $(CFLAGS) -c colonies3D.cu

colonies_new.o : colonies_new.cu colonies3D.hpp
	$(CC) $(CFLAGS) -c colonies_new.cu


clean :
	rm -f *.o
	rm -f example3D
	rm -f growthCurve3D
