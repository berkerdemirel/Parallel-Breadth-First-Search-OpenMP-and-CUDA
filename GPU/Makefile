bfs:
	gcc graphio.c -c -O3
	gcc mmio.c -c  -O3
	nvcc -O3 -c bfs.cu -o bfs.o
	nvcc -arch=sm_30 -std=c++11 -o bfs main.cpp mmio.o graphio.o bfs.o -O3 -Xcompiler -fopenmp
	rm -f *.o