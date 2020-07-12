
// add<<<1, 1>>>();
// add<<<N ,1>>>(); N times in parallel

// Each parallel invocation of add() is referred to as a block
// The set of blocks is referred to as a grid
// Each invocation can refer to its block index using blockIdx.x
// By using blockIdx.x to index into the array, each block handles a different element of the array
// On the device, each block can execute in parallel 
// Block 0 c[0]= = a[0] + b[0]
// Block 1 c[1]= = a[1] + b[1]

__global__ void add(int *a, int *b, int *c) {
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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

  // Launch add() kernel on GPU with N blocks
  add<<<N,1>>>(d_a, d_b, d_c);

  // Copy results back to host 
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}



