// rmater -> 8
// coPapersDBLP -> 14
// wiki -> 257
// europe -> 17345

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
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>

#define N 1

char gfile[2048];
char gfile_inv[2048];
double mf, mu, nf, old_nf;
//mu -> total unvisited nof edges
//mf -> total number of edges in froniter
//nf -> number of vertices in frontier

using namespace std;

void printArray(int *x, int len) {
	for (int i = 0; i < len; i++) {
		cout << x[i] << " ";
	}
	cout << endl;
}

bool topDown(etype *row, vtype *col, int *distance, int &level, int nov) {
	bool improvement = false;

#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for reduction(-:mf) reduction(||:improvement) schedule(guided, 32)
		for (int i = 0; i < nov; i++) {
			if (distance[i] == level) {
				for (int j = row[i]; j < row[i + 1]; j++) {
					int u = col[j];
					if (distance[u] < 0) {
						mf -= distance[u];
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

bool bottomUp(etype *row_inv, vtype *col_inv, int *distance, int &level, int nov, int *unvisited, int uvSize) {
	bool improvement = false;
	nf = 0;

#pragma omp parallel for reduction(+:nf) reduction(||:improvement) schedule(guided, 32)
	for (int i = 0; i < uvSize; i++) {
		int v = unvisited[i];
		if (distance[v] < 0) {
			for (int j = row_inv[v]; j < row_inv[v + 1]; j++) {
				int u = col_inv[j];
				if (distance[u] == level) {
					distance[v] = level + 1;
					nf++;
					improvement = true;
					break;
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
	etype *row_ptr_inv;
	vtype *col_ind_inv;
	ewtype *ewghts_inv;
	vwtype *vwghts_inv;
	vtype nov_inv, source_inv;
	double start, end, total;

	const char* fname = argv[1]; // matrix file name
	strcpy(gfile, fname);
	int zerobased = atoi(argv[2]);
	int symmetric = 0;
	int *symptr = &symmetric;

	if (read_graph(gfile, &row_ptr, &col_ind, &ewghts, &vwghts, &nov, 0, zerobased, symptr) == -1)
	{
		printf("error in graph read\n");
		exit(1);
	}

	/*CHECK IF SYMMETRIC*/
	if (!is_symmetric(gfile))
	{
		string filename(gfile);
		string name = filename.substr(0, filename.find(".")) + "_inverse.mtx";
		ifstream f(name.c_str());
		const char *fname_inv = name.c_str();
		strcpy(gfile_inv, fname_inv);

		if (!f.good())
		{
			//cout << "The graph is not symmetric. Creating its inverse." << endl;
			//string filename(gfile);
			fstream input;
			input.open(filename.c_str());
			if (input.fail())
			{
				cout << "Error in file opening." << endl;
			}

			string oname = filename.substr(0, filename.find(".")) + "_inverse.mtx";
			ofstream out;
			out.open(oname.c_str());
			string line;
			while (getline(input, line))
			{
				if (line[0] == '%')
				{
					out << line << endl;
				}
				else
				{
					break;
				}
			}
			out << line << endl;
			int v1, v2;
			while (!input.eof())
			{
				input >> v1 >> v2;
				out << v2 << " " << v1 << endl;
			}
			out.close();

			const char *fname_inv = oname.c_str();
			strcpy(gfile_inv, fname_inv);
		}

		if (read_graph(gfile_inv, &row_ptr_inv, &col_ind_inv, &ewghts_inv, &vwghts_inv, &nov, 0, zerobased, symptr) == -1)
		{
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
	int count = 0;
	for (int i = 0; i < nov; i++) {
	  d = row_ptr[i + 1] - row_ptr[i];
	  if (maxDegree < d)
		maxDegree = d;
	  if (minDegree > d)
		minDegree = d;
	  if (d == 0)
		count++;
	}
	cout << "Min & max degrees: " << minDegree << " & " << maxDegree << endl;
	cout << "Floating vertices: " << count << endl;*/

	vector<int> sources;
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<std::mt19937::result_type> rand(0, nov - 1);
	for (int i = 0; i < N; i++) {
		sources.push_back(rand(rng));
	}

	cout << "Sources: ";
	for (int i = 0; i < N; i++) {
		cout << sources[i] << " ";
	}
	cout << endl;


	for (int t = 1; t <= 16; t = t * 2) {
		sleep(1);
		total = 0;
		for (int n = 0; n < N; n++) {
			//omp_set_dynamic(0);
			omp_set_num_threads(t);

			int *distance = new int[nov];
			for (int i = 0; i < nov; i++) {
				distance[i] = row_ptr[i] - row_ptr[i + 1];	// = -outdegree(i)
			}

			source = sources[n];
			mf = -distance[source]; //number of edges on frontier
			mu = row_ptr[nov];  // total number of edges
			distance[source] = 0;

			// (a,b) = (15,18) in implementation, (14,24) in paper.
			double alpha = 6.0;
			double beta = 24.0;

			int level = 0;
			bool improvement = true;

			int *unvisited = new int[nov];
			int uvSize = 0;

			start = omp_get_wtime();
			while (improvement) {
				if (mf > mu / alpha) {
					uvSize = 0;
					for (int i = 0; i < nov; i++) {
						if (distance[i] < 0) {
							unvisited[uvSize++] = i;
						}
						else if (distance[i] == level) {
							nf++;
						}
					}

					do {
						old_nf = nf;
						improvement = bottomUp(row_ptr_inv, col_ind_inv, distance, level, nov, unvisited, uvSize);
					} while (improvement && ((mf > mu / alpha) || (nf >= old_nf || nf > double(nov) / beta)));
				}

				else {
					mu -= mf;
					improvement = topDown(row_ptr, col_ind, distance, level, nov);
				}
			}
			end = omp_get_wtime();
			total += end - start;

			int traversed = 0;
			for (int i = 0; i < nov; i++) {
				if (distance[i] > 0) {
					traversed++;
				}
			}
			traversed++; //for source

			/*ofstream myfile;
			string f = "test" + to_string(t) + ".txt";
			myfile.open(f);
			for (int x = 0; x < nov; x++)
			myfile << to_string(distance[x]) << " ";
			myfile << endl;
			myfile.close();*/

			delete[] unvisited;
			delete[] distance;

			cout << "Threads: " << t << "\tLevel: " << level << "\tDelta: " << nov - traversed << "\tTime: " << total << endl;
		}
		//cout << "Threads: " << t << "\tTime: " << total / N << endl;
	}

	return 0;
}
