#include <bits/stdc++.h>
using namespace std;

// Kernel function to add the elements of two arrays
__global__
void GPUmatmul(int N, double *x, double *y, double *ans)
{
    int t= (blockDim.x*blockDim.y)*threadIdx.z+(threadIdx.y*blockDim.x)+(threadIdx.x);
    int b= (gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);
    int T= blockDim.x*blockDim.y*blockDim.z;
    int B= gridDim.x*gridDim.y*gridDim.z;
    
    for (int i=b;i<N;i+=B)
    {
		for(int j=t;j<N;j+=T)
		{
			for(int k=0;k<N;k++)
			{
				ans[i*N+j]+=(x[i*N+k]*y[k*N+j]);
			}
		}
	}
}

void CPUmatmul(int N,double *x, double *y, double *ans)
{
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				ans[i*N+j]+=(x[i*N+k]*y[k*N+j]);
			}
		}
	}
}

bool sanity_check(int N,double *ans)
{
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			if(ans[i*N+j]!=20.0)return false;
		}
	}
	return true;
}

void printtt(int N,double *ans)
{
	for(int i=0;i<16;i++)
	{
		for(int j=0;j<16;j++)
		{
			cout<<ans[i*N+j]<<" ";
		}
		cout<<endl;
	}
}
int main(void)
{

	int N = 1<<13;
	double *x, *y, *ans;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N*N*sizeof(double));
	cudaMallocManaged(&y, N*N*sizeof(double));
	cudaMallocManaged(&ans, N*N*sizeof(double));

	// initialize x,y and ans arrays on the host
	for (int i = 0; i < N; i++) 
	{
		for(int j=0;j<N;j++)
		{
			x[i*N+j]=5;
			y[i*N+j]=(i==j?1:0);
			ans[i*N+j]=(double)0.000000000000;
		}
	}

  
  clock_t t;
  double avg=0;
  // Run kernel on 1M elements on the GPU
  for(int i=0;i<=3;i++)
  {
	  t=clock();
	  //CPUmatmul(N, x, y,ans);
	  t = clock() - t;
	  if(i)avg+=t;
	  printf ("It took CPU-%d %f ms.\n",i,(((double)t)/CLOCKS_PER_SEC)*1000);
  }
  avg/=3;
  avg/=CLOCKS_PER_SEC;
  avg*=1000;
  printf ("It took %lf ms on avg.\n",avg);
  if(sanity_check(N,ans))cout<<"RUN OK."<<endl;
  else cout<<"RUN NOT OK."<<endl;
 
	// initialize x,y and ans arrays on the host
	for (int i = 0; i < N; i++) 
	{
		for(int j=0;j<N;j++)
		{
			x[i*N+j]=5;
			y[i*N+j]=(i==j?1:0);
			ans[i*N+j]=(double)0.000000000000;
		}
	}
   avg=0;
  // Run kernel on 1M elements on the GPU
  for(int i=0;i<=3;i++)
  {
	  t=clock();
	  GPUmatmul<<<dim3(16,16,16), dim3(16,8,8)>>>(N, x, y,ans);
	  cudaDeviceSynchronize();
	  t = clock() - t;
	  if(i)avg+=t;
	  printf ("It took GPU-%d %f ms.\n",i,(((double)t)/CLOCKS_PER_SEC)*1000);
  }
  avg/=3;
  avg/=CLOCKS_PER_SEC;
  avg*=1000;
  printf ("It took %lf ms on avg.\n",avg);
  if(sanity_check(N,ans))cout<<"RUN OK."<<endl;
  else cout<<"RUN NOT OK."<<endl;

  
  // Free memory
  cudaFree(x);
  cudaFree(y); 
  return 0;
}



