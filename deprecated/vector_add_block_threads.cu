// Several blocks with one thread each 
// One block with several threads
// Now, use both blocks and threads
// Indexing an array with one element per threads (8 threads/block)
// threadIdx.x 0 1 2 3 4 5 6 7 
//           blockIdx.x=0
// use the built-in variable blockDim.x for threads per block
#include "stdio.h"

__global__ void add(int *a, int *b, int *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = rand();
}


#define N (2048*2048) 
#define THREADS_PER_BLOCK 512
int main(void) {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof(int);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);

  // Alloc space for host copies of a, b, c and setup input values 
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size); 

  // Copy inputs to device 
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with Nthreads 
  add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

  // Copy results back to host 
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}



// Typical problems are not friendly multiples of blockDim.x
// Avoid accessing beyond the end of the arrays
__global__ voidadd(int *a, int *b, int *c, int n) {
  int index = threadIdx.x+ blockIdx.x* blockDim.x;
  if (index < n)
    c[index] = a[index] + b[index];
}
// Update the kernel launch
add<<<(N + M - 1) / M, M>>>(d_a, d_b, d_c, N);

// Importance of THREADS
// - Communicate 
// - Synchronize

// Implementaing within a Block
// Sharing data between threads 
// - declaration using __shared___
// - not visible to threads in other blocks

