extern "C"
{
	#include "graphio.h"
	#include "graph.h"
}
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <random>
#include <vector>

#define N 1

char gfile[2048];
char gfile_inv[2048];

using namespace std;

void wbfs(unsigned int *row_ptr, int *col_ind, int *distance, int nov, int *d_distance, unsigned int *d_row_ptr, int *d_col_ind);
void qbfs(unsigned int *row_ptr, int *col_ind, int *distance, int nov, int source);
void hybrid(unsigned int *row_ptr, unsigned int *row_ptr_inv, int *col_ind, int *col_ind_inv, int *distance, int nov, int source, double alpha);

int main(int argc, char *argv[]) {
	//GRAPH READ
	etype *row_ptr;
	vtype *col_ind;
	ewtype *ewghts;
	vwtype *vwghts;
	vtype nov, source;
	etype *row_ptr_inv;
	vtype *col_ind_inv;
	ewtype *ewghts_inv;
	vwtype *vwghts_inv;
	vtype nov_inv, source_inv;
	double start, end, total;

	const char *fname = argv[1]; // matrix file name
	strcpy(gfile, fname);
	int zerobased = atoi(argv[2]);
	int symmetric = 0;
	int *symptr = &symmetric;

	if (read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0, zerobased, symptr) == -1) {
		printf("error in graph read\n");
		exit(1);
	}

	/*CHECK IF SYMMETRIC*/
	if (!is_symmetric(gfile)) {
		string filename(gfile);
		string name = filename.substr(0, filename.find(".")) + "_inverse.mtx";
		ifstream f(name.c_str());
		const char *fname_inv = name.c_str();
		strcpy(gfile_inv, fname_inv);

		if (!f.good()) {
		fstream input;
		input.open(filename.c_str());
		if (input.fail()) {
			cout << "Error in file opening." << endl;
		}

		string oname = filename.substr(0, filename.find(".")) + "_inverse.mtx";
		ofstream out;
		out.open(oname.c_str());
		string line;
		while (getline(input, line)) {
			if (line[0] == '%') {
			out << line << endl;
			}
			else {
			break;
			}
		}
		out << line << endl;
		int v1, v2;
		while (!input.eof()) {
			input >> v1 >> v2;
			out << v2 << " " << v1 << endl;
		}
		out.close();

		const char *fname_inv = oname.c_str();
		strcpy(gfile_inv, fname_inv);
		}

		if (read_graph(gfile_inv, &row_ptr_inv, &col_ind_inv, &ewghts_inv, &vwghts_inv, &nov, 0, zerobased, symptr) == -1) {
		printf("error in graph read\n");
		exit(1);
		}
	}
	else {
		row_ptr_inv = row_ptr;
		col_ind_inv = col_ind;
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

  	/*cout << "Sources: ";
	for(int i = 0; i < N; i++) {
		cout << sources[i] << " ";
	}
	cout << endl;*/

	for (int n = 0; n < N; n++) {
		int *distance = new int[nov];
		int *globalQueue = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance[i] = -1;
			globalQueue[i] = -1;
		}
		source = 0;
		distance[source] = 0;
		globalQueue[0] = source;

		int level = 0;
		bool improvement = true;
		int s = 0;
		int e = 1;

		int max = 0;
		start = omp_get_wtime();
		while (s < e) {
			int u = globalQueue[s++];
			level = distance[u];
			for (int i = row_ptr[u]; i < row_ptr[u + 1]; i++) {
				int v = col_ind[i];		
				if (distance[v] < 0) {
					distance[v] = level + 1;
					globalQueue[e++] = v;
				}
			}
			if (e - s > max)
				max = e - s;
		}
		end = omp_get_wtime();

		cout << "Max frontier size: " << max << endl;
		total += end - start;
		int traversed = 0;
		for (int i = 0; i < nov; i++) {
			if (distance[i] != - 1) {
				traversed++;
			}
		}
		//ofstream myfile;
		//string f = "test_cpu.txt";
		//myfile.open(f);
		//for (int x = 0; x < nov; x++)
		//	myfile << to_string(distance[x]) << " ";
		//myfile.close();

		cout << "CPU time: " << total  << " s" << endl;

		int * distance_for_cuda = new int[nov];
		//GPU WBFS
		/*for (int i = 0; i < nov; i++) {
			distance_for_cuda[i] = -1;
		}
		distance_for_cuda[source] = 0;

		for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) { // preprocessing
			if (distance_for_cuda[col_ind[i]] < 0)
				distance_for_cuda[col_ind[i]] = 1;
		}

		int *d_distance;
		unsigned int *d_row_ptr;
		int *d_col_ind;

		wbfs(row_ptr, col_ind, distance_for_cuda, nov,  d_distance,  d_row_ptr,  d_col_ind);

		for (int i = 0; i < nov; i++) {
			if (distance_for_cuda[i] != distance[i]) {
				cout << "wbfs error " << i << " " << distance_for_cuda[i] << " " << distance[i] << endl;
			}
		}

		//GPU QBFS
		for (int i = 0; i < nov; i++) {
			distance_for_cuda[i] = -1;
		}

		distance_for_cuda[source] = 0;
		for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) { // preprocessing
			if (distance_for_cuda[col_ind[i]] < 0)
				distance_for_cuda[col_ind[i]] = 1;
		}
		
		qbfs(row_ptr, col_ind, distance_for_cuda, nov, source);

		for (int i = 0; i < nov; i++) {
			if (distance_for_cuda[i] != distance[i]) {
				cout << "qbfs error " << i << " " << distance_for_cuda[i] << " " << distance[i] << endl;
			}
		}*/

		//GPU HYBRID
		for (int i = 0; i < nov; i++) {
			distance_for_cuda[i] = row_ptr[i] - row_ptr[i + 1]; // = -outdegree(i)
			if (distance[i] < 0) {
				distance[i] = row_ptr[i] - row_ptr[i + 1];
			}
		}

		distance_for_cuda[source] = 0;
		for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) { // preprocessing
			if (distance_for_cuda[col_ind[i]] < 0)
				distance_for_cuda[col_ind[i]] = 1;
		}		

		double alpha = 8.0;
		hybrid(row_ptr, row_ptr_inv, col_ind, col_ind_inv, distance_for_cuda, nov, source, alpha);

		for (int i = 0; i < nov; i++) {
			if (distance_for_cuda[i] != distance[i]) { //(distance_for_cuda[i] >= 0 && distance[i] >= 0) && 
				cout << "hybrid error " << i << " " << distance_for_cuda[i] << " " << distance[i] << endl;
			}
		}

		delete[] distance;
		delete[] distance_for_cuda;
		delete[] globalQueue;
	}

	/*for (int n = 0; n < N; n++) {
		int *distance = new int[nov];
		int *globalQueue = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance[i] = -1;
			globalQueue[i] = -1;
		}
		source = 0;
		distance[source] = 0;
		globalQueue[0] = source;

		int level = 0;
		bool improvement = true;

		//GPU WBFS
		int * distance_for_cuda = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance_for_cuda[i] = -1;
		}
		distance_for_cuda[source] = 0;

		int *d_distance;
		unsigned int *d_row_ptr;
		int *d_col_ind;

		wbfs(row_ptr, col_ind, distance_for_cuda, nov,  d_distance,  d_row_ptr,  d_col_ind);

		delete[] distance;
		delete[] distance_for_cuda;
		delete[] globalQueue;
	}

	for (int n = 0; n < N; n++) {
		int *distance = new int[nov];
		int *globalQueue = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance[i] = -1;
			globalQueue[i] = -1;
		}
		source = 0;
		distance[source] = 0;
		globalQueue[0] = source;

		int level = 0;
		bool improvement = true;

		//GPU QBFS
		int * distance_for_cuda = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance_for_cuda[i] = -1;
		}
		distance_for_cuda[source] = 0;

		qbfs(row_ptr, col_ind, distance_for_cuda, nov, source);

		delete[] distance;
		delete[] distance_for_cuda;
		delete[] globalQueue;
	}

	for (int n = 0; n < N; n++) {
		int *distance = new int[nov];
		int *globalQueue = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance[i] = -1;
			globalQueue[i] = -1;
		}
		source = 0;
		distance[source] = 0;
		globalQueue[0] = source;

		int level = 0;
		bool improvement = true;
	
		//GPU HYBRID
		int * distance_for_cuda = new int[nov];
		for (int i = 0; i < nov; i++) {
			distance_for_cuda[i] = row_ptr[i] - row_ptr[i + 1]; // = -outdegree(i)
			if (distance[i] < 0) {
				distance[i] = row_ptr[i] - row_ptr[i + 1];
			}
		}

		distance_for_cuda[source] = 0;
		double alpha = 8.0;
		double beta = 15.0;

		hybrid(row_ptr, row_ptr_inv, col_ind, col_ind_inv, distance_for_cuda, nov, source, alpha, beta);

		delete[] distance;
		delete[] distance_for_cuda;
		delete[] globalQueue;
	}*/
	return 0;
}
