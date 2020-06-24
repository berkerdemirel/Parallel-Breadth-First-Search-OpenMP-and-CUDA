#include <string.h>
#include <stdlib.h>
#include "graph.h"

#ifndef GRAPHIO_H
#define GRAPHIO_H

int read_graph(char* gfile, etype **xadj, vtype **adj, ewtype **ewghts, vwtype **vwghts, vtype* nov, int loop, int zerobased, int *sym);
int is_symmetric(char* gfile);
#endif
