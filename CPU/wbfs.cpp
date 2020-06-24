extern "C"
{
#include "graphio.h"
#include "graph.h"
}
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <limits.h>
#include <random>
#include <vector>

#define N 1

char gfile[2048];

using namespace std;

void printArray(int *x, int len) {
	for (int i = 0; i < len; i++) {
		cout << x[i] << " ";
	}
	cout << endl;
}

bool topDown(etype *row, vtype *col, int* distance, int &level, int nov) {
	bool improvement = false;

	#pragma omp parallel
	{
		#pragma omp for reduction(||:improvement) schedule(guided, 32)
		for (int v = 0; v < nov; v++) {
			if (distance[v] == level) {
				for (int j = row[v]; j < row[v + 1]; j++) {
					int u = col[j];
					if (distance[u] < 0) {
						distance[u] = level + 1;
						improvement = true;
					}
				}
			}
		}
	}

	if (improvement) {
		level++;
	}

	return improvement;
}

int main(int argc, char *argv[])
{
	//GRAPH READ
	etype *row_ptr;
	vtype *col_ind;
	ewtype *ewghts;
	vwtype *vwghts;
	vtype nov, source;
	double start, end, total = 0;

	const char* fname = argv[1]; // matrix file name
	strcpy(gfile, fname);
	int zerobased = atoi(argv[2]);
	int dummy;
	if (read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0, zerobased, &dummy) == -1)
	{
		printf("error in graph read\n");
		exit(1);
	}
	/*cout << "Vertices: " << nov << endl;
	cout << "Edges: " << row_ptr[nov] << endl;
	cout << "Avg degree: " << double(row_ptr[nov]) / double(nov) << endl; 
	int maxDegree = 0;
	int minDegree = INT_MAX;
	int d;
    for (int i = 0; i < nov; i++) {
		d = row_ptr[i + 1] - row_ptr[i];
		if (maxDegree < d) maxDegree = d;
		if (minDegree > d) minDegree = d;
    }
	cout << "Min & max degrees: " << minDegree << " & " << maxDegree << endl;*/

	vector<int> sources;
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<std::mt19937::result_type> rand(0, nov - 1);
	for (int i = 0; i < N; i++) {
		sources.push_back(rand(rng));
	}

	cout << "Sources: ";
	for(int i = 0; i < N; i++) {
		cout << sources[i] << " ";
	}
	cout << endl;

	for (int t = 1; t <= 16; t = t * 2) {
	total = 0;
	for (int n = 0; n < N; n++) {
		omp_set_dynamic(0);
		omp_set_num_threads(t);

		int *distance = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance[i] = -1;
		}

		source = sources[n];
		distance[source] = 0;

		int level = 0;
		bool improvement = true;
		start = omp_get_wtime();
		do {
			improvement = topDown(row_ptr, col_ind, distance, level, nov);
		} while (improvement);
		end = omp_get_wtime();
		total += end - start;
		
		int traversed = 0;
		for (int i = 0; i < nov; i++) {
			if (distance[i] != -1) {
				traversed++;
			}
		}

		/*ofstream myfile;
		string f = "test" + to_string(t) + ".txt";
		myfile.open(f);
		for (int x = 0; x < nov; x++)
			myfile << to_string(distance[x]) << " ";
		myfile << endl;
		myfile.close();*/

		delete[] distance;

		cout << "Threads: " << t << "\tLevel: " << level << "\tDelta: " << nov - traversed << "\tTime: " << total << endl;
	}
	//cout << "Threads: " << t << "\tTime: " << total / N << endl;
	}

	return 0;
}
