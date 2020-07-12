// A block can be split into parallel threads
// use parallel threads instead of parallel blocks
__global__ void add(int *a, int *b, int *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = rand();
}


#define N 512
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
  add<<<1,N>>>(d_a, d_b, d_c);
  printf(a)
  printf(b)
  printf(c)

  // Copy results back to host 
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}



