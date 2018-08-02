#include <bits/stdc++.h>
using namespace std;

double data[1024+1];
// to save data in file 
void csv()
{
    char filename[11]="data.csv";
    FILE *fp;

    fp=fopen(filename,"w+");

    fprintf(fp,"Number of Threads, Average Time");

    for(int i=32;i<=1024;i+=32)
    {
        fprintf(fp,"\n%d",max(i,1));
        fprintf(fp,",%lf ",data[i]);
    }

    fclose(fp);

    printf("\n%sfile created",filename);

}


// Kernel function to add the elements of two arrays
__global__
void add(int N, float *X, float *Y)
{
    int t= blockIdx.x * blockDim.x + threadIdx.x;
    int T = blockDim.x * gridDim.x;
    for (int i = t; i < N; i += T)
        Y[i] = X[i] + Y[i];
}

int main(void)
{

	int N = 1<<27;//1.34217728 *10^8 elements. 512 MB
	float *X, *Y;

	//Allocates Memory so that both GPU and CPU can access (512*2=1GB). 
	cudaMallocManaged(&X, N*sizeof(float));
	cudaMallocManaged(&Y, N*sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++)
	{
		X[i] = 1.0f;
		Y[i] = 2.0f;
	}
	clock_t t;
	
	
	// Run add 10 times with different number of threads. and save the average time on a table.
	//it is good practice to keep thread number multiple of 32. 
	for(int i=32;i<=1024;i+=32)
	{
		int T=i;// we will need atleast 1 thread. 
		double avg=0;
		for(int j=0;j<=10;j++)
		{
			t=clock();
			add<<<dim3(8,1,1), dim3(T,1,1)>>>(N, X, Y);
			cudaDeviceSynchronize();
			t = clock() - t;
			printf("T = %d, Run = %d Time = %lf\n",T,j,(((double)t)/CLOCKS_PER_SEC)*1000);
			if(j)avg+=((((double)t)/CLOCKS_PER_SEC)*1000);// skips the first run. 
		}
		avg=avg/10;
		data[i]=avg;
		printf ("It took GPU %lf ms with %d threads.\n",avg,T);
	}
	
	csv();// save data in output file
	
	// Free memory
	cudaFree(X);
	cudaFree(Y); 
	return 0;
}


