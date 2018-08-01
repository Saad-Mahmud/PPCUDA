#include <iostream>
#include <math.h>
#include <time.h> 
#include <stdio.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{

  int N = 2e7;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 512>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  clock_t t=clock();
  for (int i = 0; i < N; i++) y[i] = x[i] + y[i];
  t = clock() - t;
  printf ("It took CPU %f ms.\n",(((float)t)/CLOCKS_PER_SEC)*1000);
  
  // Free memory
  cudaFree(x);
  cudaFree(y); 
  return 0;
}


