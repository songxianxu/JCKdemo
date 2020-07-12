#include "stdio.h"

// Global means add will execute on the device 
// add() will be called from the host
// Pointer for the varaibles because a b runs on device, a, b, c must point to device memory
__global__ void add(int *a, int *b, int *c) {
  *c = *a + *b;
}
// Host and device memory are separate entities
// - Device pointers point to GPU memory
// - Host pointers point to CPU memory
// - cudaMalloc(), cudaFree(), cudaMemcpy()
// - malloc(), free(), memcpy()

int main(void) {
  int a, b, c;          // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = sizeof(int);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  a = 2;
  b = 7;

  cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

  add<<<1,1>>>(d_a, d_b, d_c);

  cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

