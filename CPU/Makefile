wbfs: wbfs.cpp
	gcc graphio.c -c -O3 -g
	gcc mmio.c -c  -O3 -g
  #gcc graph.c -c -O3
	g++ wbfs.cpp -fopenmp -c -O3 -std=c++11 -g
	g++ -o wbfs wbfs.o mmio.o graphio.o -O3 -fopenmp -std=c++11 -g 
	rm -f *.o

qbfs: qbfs.cpp
	gcc graphio.c -c -O3 -g
	gcc mmio.c -c  -O3 -g
  #gcc graph.c -c -O3
	g++ qbfs.cpp -fopenmp -c -O3 -std=c++11 -g
	g++ -o qbfs qbfs.o mmio.o graphio.o -O3 -fopenmp -std=c++11 -g
	rm -f *.o

hybrid: hybrid.cpp
	gcc graphio.c -c -O3 
	gcc mmio.c -c  -O3 
  #gcc graph.c -c -O3
	g++ hybrid.cpp -fopenmp -c -O3 -std=c++11 
	g++ -o hybrid hybrid.o mmio.o graphio.o -O3 -fopenmp -std=c++11 
	rm -f *.o

preqbfs: preqbfs.cpp
	gcc graphio.c -c -O3 -g
	gcc mmio.c -c  -O3 -g
  #gcc graph.c -c -O3
	g++ preqbfs.cpp -fopenmp -c -O3 -std=c++11 -g
	g++ -o preqbfs preqbfs.o mmio.o graphio.o -O3 -fopenmp -std=c++11 -g
	rm -f *.o
