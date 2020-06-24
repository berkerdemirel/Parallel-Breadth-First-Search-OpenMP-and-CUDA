#include <cuda.h>
#include <iostream>
#include <stdio.h>

using namespace std;

#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("Fatal error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
  }

__global__ void cudawbfs(int *distance, unsigned int *row_ptr, int *col_ind, int nov, int *improvement, int level)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int localImprovement = 0;

  if(tid < nov && distance[tid] == level) {
    for(int e = row_ptr[tid]; e < row_ptr[tid + 1]; e++){
      int adj = col_ind[e];
      if(distance[adj] < 0){
        distance[adj] = level + 1;
        localImprovement = 1;
      }
    }
  }
  
  if(localImprovement) {
    (*improvement) = localImprovement;
  }
}

void wbfs(unsigned int * row_ptr, int * col_ind, int * distance, int nov, int * d_distance, unsigned int * d_row_ptr, int * d_col_ind){
  //initializations
  int size_of_rowptr = (nov + 1) * sizeof(int);
  int size_of_colind = row_ptr[nov] * sizeof(int);
  int *d_improvement, *d_nov, *d_level;

  //memory allocations
  cudaCheck(cudaMalloc((void**) &d_improvement, sizeof(int)));
  cudaCheck(cudaMalloc((void**) &d_nov, sizeof(int)));
  cudaCheck(cudaMalloc((void**) &d_level, sizeof(int)));
  cudaCheck(cudaMalloc((void**) &d_row_ptr, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_distance, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_col_ind, size_of_colind));

  //memory copies
  cudaCheck(cudaMemcpy(d_distance, distance, size_of_rowptr, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, size_of_rowptr, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_col_ind, col_ind, size_of_colind, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_nov, &nov, sizeof(int), cudaMemcpyHostToDevice));


  //start time
  cudaEvent_t start;
  cudaEvent_t stop;

  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  cudaCheck(cudaEventRecord(start, 0));

  int *improvement = new int;
  int level = 1;
  do{
     (*improvement) = 0;
     cudaCheck(cudaMemcpy(d_improvement, improvement, sizeof(int), cudaMemcpyHostToDevice));
     cudawbfs<<<(nov + 1023) / 1024, 1024>>>(d_distance, d_row_ptr, d_col_ind, nov, d_improvement, level);
     cudaCheck(cudaMemcpy(improvement, d_improvement, sizeof(int), cudaMemcpyDeviceToHost));
     level++;
  } while((*improvement) == 1);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);

  //take value again
  cudaCheck(cudaMemcpy(distance, d_distance, size_of_rowptr, cudaMemcpyDeviceToHost));

  //deallocations
  cudaCheck(cudaFree(d_row_ptr));
  cudaCheck(cudaFree(d_distance));
  cudaCheck(cudaFree(d_col_ind));

  printf("GPU WBFS time: %f s\n", elapsed / 1000);
}

__global__ void cudaqbfs(int *distance, unsigned int *row_ptr, int *col_ind, int *queue, int *nextQueue, int size, int *nextSize, int level) {
  int index, u, v, tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < size) {
    u = queue[tid];
    for(int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
      v = col_ind[e];
      if (distance[v] == -1) {
        distance[v] = level + 1;
        index = atomicAdd(nextSize, 1);
        nextQueue[index] = v;
      }
    }
  }
}

void qbfs(unsigned int *row_ptr, int *col_ind, int *distance, int nov, int source) {
  int srcNeigh = row_ptr[source + 1] - row_ptr[source];
  int *srcArr = new int[srcNeigh];
  int index = 0;
  for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) {
    if (distance[col_ind[i]] == 1) {
      srcArr[index++] = col_ind[i];
    }
  }
  
  int size_of_rowptr = (nov + 1) * sizeof(int);
  int size_of_colind = row_ptr[nov] * sizeof(int);
  unsigned int *d_row_ptr;
  int *d_col_ind, *d_distance, *d_queue, *d_nextQueue, *d_nextSize;

  cudaCheck(cudaMalloc((void**) &d_row_ptr, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_col_ind, size_of_colind));
  cudaCheck(cudaMalloc((void**) &d_distance, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_queue, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_nextQueue, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_nextSize, sizeof(int)));

  cudaCheck(cudaMemcpy(d_distance, distance, size_of_rowptr, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, size_of_rowptr, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_col_ind, col_ind, size_of_colind, cudaMemcpyHostToDevice));
  //cudaCheck(cudaMemcpy(d_queue, &source, sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_queue, srcArr, srcNeigh * sizeof(int), cudaMemcpyHostToDevice));

	cudaEvent_t start;
 	cudaEvent_t stop;
 	cudaCheck(cudaEventCreate(&start));
 	cudaCheck(cudaEventCreate(&stop));
 	cudaCheck(cudaEventRecord(start, 0));

  int size = srcNeigh;
  int *nextSize = new int;
  *nextSize = 0;
  int level = 1;
  do {
    cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));
    cudaqbfs<<<(size + 1023) / 1024, 1024>>>(d_distance, d_row_ptr, d_col_ind, d_queue, d_nextQueue, size, d_nextSize, level);
    cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));
    level++;
    size = *nextSize;
    *nextSize = 0;
    swap(d_queue, d_nextQueue);
  } while(size > 0);

	cudaEventRecord(stop, 0);
 	cudaEventSynchronize(stop);
 	float elapsed;
 	cudaEventElapsedTime(&elapsed, start, stop);

  cudaCheck(cudaMemcpy(distance, d_distance, size_of_rowptr, cudaMemcpyDeviceToHost));

  cudaCheck(cudaFree(d_row_ptr));
  cudaCheck(cudaFree(d_col_ind));
  cudaCheck(cudaFree(d_distance));
  cudaCheck(cudaFree(d_queue));
  cudaCheck(cudaFree(d_nextQueue));
  cudaCheck(cudaFree(d_nextSize));

  printf("GPU QBFS time: %f s\n", elapsed / 1000);
}


__global__ void cudatdwbfs(int *distance, unsigned int *row_ptr, int *col_ind, int nov, int level, int *mf) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < nov && distance[tid] == level) {     
     for(int e = row_ptr[tid]; e < row_ptr[tid + 1]; e++) {
       int adj = col_ind[e];
       if(distance[adj] < 0) {
         atomicAdd(mf, -distance[adj]);
         distance[adj] = level + 1;
       }
     }
   }
}

__global__ void cudatdqbfs(int *distance, unsigned int *row_ptr, int *col_ind, int *queue, int *nextQueue, int size, int *nextSize, int level, int *mf) {
  int index, u, v, tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < size) {
    u = queue[tid];
    for(int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
      v = col_ind[e];
      if (distance[v] < 0) {
        index = atomicAdd(nextSize, 1);
        atomicAdd(mf, -distance[v]);
        distance[v] = level + 1;
        nextQueue[index] = v;
      }
    }
  }
}

__global__ void cudabuwbfs(int *distance, unsigned int *row_ptr_inv, int *col_ind_inv, int nov, int level, int *mf) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < nov && distance[tid] < 0) {
    for(int e = row_ptr_inv[tid]; e < row_ptr_inv[tid + 1]; e++) {
      int adj = col_ind_inv[e];
      if(distance[adj] == level) {
        atomicAdd(mf, -distance[tid]);
        distance[tid] = level + 1;
        break;
      }
    }
  }
}

__global__ void cudabuqbfs(int *distance, unsigned int *row_ptr_inv, int *col_ind_inv, int nov, int level, int *nextSize, int *mf) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < nov && distance[tid] < 0) {
    for(int e = row_ptr_inv[tid]; e < row_ptr_inv[tid + 1]; e++) {
      int adj = col_ind_inv[e];
      if(distance[adj] == level) {
        atomicAdd(mf, -distance[tid]);
        atomicAdd(nextSize, 1);
        distance[tid] = level + 1;
        break;
      }
    }
  }
}

void hybrid(unsigned int *row_ptr, unsigned int *row_ptr_inv, int *col_ind, int *col_ind_inv, int *distance, int nov, int source, double alpha) { // int init_mf?
	int size_of_rowptr = (nov + 1) * sizeof(int);
  int size_of_colind = row_ptr[nov] * sizeof(int);
  int *improvement = new int;
  unsigned int *d_row_ptr, *d_row_ptr_inv;
  int *d_col_ind, *d_col_ind_inv, *d_distance, *d_mf;

  cudaCheck(cudaMalloc((void**) &d_row_ptr, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_row_ptr_inv, size_of_rowptr));
	cudaCheck(cudaMalloc((void**) &d_col_ind, size_of_colind));
	cudaCheck(cudaMalloc((void**) &d_col_ind_inv, size_of_colind));
  cudaCheck(cudaMalloc((void**) &d_distance, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_mf, sizeof(int)));

  cudaCheck(cudaMemcpy(d_distance, distance, size_of_rowptr, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, size_of_rowptr, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_row_ptr_inv, row_ptr_inv, size_of_rowptr, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_col_ind, col_ind, size_of_colind, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_col_ind_inv, col_ind_inv, size_of_colind, cudaMemcpyHostToDevice));


  int srcNeigh = row_ptr[source + 1] - row_ptr[source];
  int *srcArr = new int[srcNeigh];
  int index = 0;
  for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) {
    if (distance[col_ind[i]] == 1) {
      srcArr[index++] = col_ind[i];
    }
  }

  int *d_queue, *d_nextQueue, *d_nextSize;
  cudaCheck(cudaMalloc((void**) &d_queue, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_nextQueue, size_of_rowptr));
  cudaCheck(cudaMalloc((void**) &d_nextSize, sizeof(int)));

  //cudaCheck(cudaMemcpy(d_queue, &source, sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_queue, srcArr, srcNeigh * sizeof(int), cudaMemcpyHostToDevice));

  int mf = row_ptr[source + 1] - row_ptr[source]; // number of traversed edges
  int mu = row_ptr[nov];                          // total number of edges
  int prev_mf = -1;
  int level = 1;

  int size = srcNeigh;
  int *nextSize = new int;
  *nextSize = 0;

	cudaEvent_t start;
 	cudaEvent_t stop;
 	cudaCheck(cudaEventCreate(&start));
 	cudaCheck(cudaEventCreate(&stop));
  cudaCheck(cudaEventRecord(start, 0));

  while (mf != prev_mf) {
    prev_mf = mf;
    if (mf > mu / alpha) {
      cudaCheck(cudaMemcpy(d_mf, &mf, sizeof(int), cudaMemcpyHostToDevice));
      cudabuwbfs<<<(nov + 1023) / 1024, 1024>>>(d_distance, d_row_ptr_inv, d_col_ind_inv, nov, level, d_mf);
      cudaCheck(cudaMemcpy(&mf, d_mf, sizeof(int), cudaMemcpyDeviceToHost));
    }
    else {
      cudaCheck(cudaMemcpy(d_mf, &mf, sizeof(int), cudaMemcpyHostToDevice));
      cudatdwbfs<<<(nov + 1023) / 1024, 1024>>>(d_distance, d_row_ptr, d_col_ind, nov, level, d_mf);      
      cudaCheck(cudaMemcpy(&mf, d_mf, sizeof(int), cudaMemcpyDeviceToHost));
    }
    level++;
  }

  /*while (mf != prev_mf) {
    prev_mf = mf;
    if (mf > mu / alpha) {
      cudaCheck(cudaMemcpy(d_mf, &mf, sizeof(int), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));
      cudabuqbfs<<<(nov + 1023) / 1024, 1024>>>(d_distance, d_row_ptr_inv, d_col_ind_inv, nov, level, d_nextSize, d_mf);
      cudaCheck(cudaMemcpy(&mf, d_mf, sizeof(int), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));
    }
    else {
      cudaCheck(cudaMemcpy(d_mf, &mf, sizeof(int), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));
      cudatdqbfs<<<(size + 1023) / 1024, 1024>>>(d_distance, d_row_ptr, d_col_ind, d_queue, d_nextQueue, size, d_nextSize, level, d_mf);      
      cudaCheck(cudaMemcpy(&mf, d_mf, sizeof(int), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));
    }
    level++;     
    size = *nextSize;
    *nextSize = 0;
    swap(d_queue, d_nextQueue);
  }*/

	cudaEventRecord(stop, 0);
 	cudaEventSynchronize(stop);
 	float elapsed;
 	cudaEventElapsedTime(&elapsed, start, stop);

  cudaCheck(cudaMemcpy(distance, d_distance, size_of_rowptr, cudaMemcpyDeviceToHost));

  cudaCheck(cudaFree(d_row_ptr));
  cudaCheck(cudaFree(d_row_ptr_inv));
  cudaCheck(cudaFree(d_col_ind));
  cudaCheck(cudaFree(d_col_ind_inv));
  cudaCheck(cudaFree(d_distance));
  cudaCheck(cudaFree(d_mf));
  cudaCheck(cudaFree(d_queue));
  cudaCheck(cudaFree(d_nextQueue));
  cudaCheck(cudaFree(d_nextSize));

  printf("GPU Hybrid time: %f s\n", elapsed / 1000);

}