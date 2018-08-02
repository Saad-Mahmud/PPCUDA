#include <bits/stdc++.h>
using namespace std;

void add(int N, float *X, float *Y)
{
	for (int i=0; i<N; i++)
	{  
		Y[i] = X[i] + Y[i];
	}
}

int main(void)
{
	int N = 1<<27;//1.34217728 *10^8 elements. 512 MB
	
	//Allocate Memory (512*2=1GB). 
	//Malloc allocates memory block and returns the pointer
	float *X = new float[N];
    float *Y = new float[N];
	printf("here");
	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) 
	{
		X[i] = 1.0f;
		Y[i] = 2.0f;
	}
	
	double avg=0;
	clock_t t;
	
	// Runs add 10 times on CPU
	for(int i=0;i<10;i++)
	{
		t=clock();//start time
		add(N, X, Y);
		t = clock() - t;//total time = end time - start time
		printf ("CPU RUN-%d time = %f ms.\n",i,(((float)t)/CLOCKS_PER_SEC)*1000);
		avg+=((((float)t)/CLOCKS_PER_SEC)*1000);//time is calculated in terms of clockcycle. Converted in millisecond
	}
	
	avg=avg/10;// average of 10 elements
	printf ("CPU Avg time = %lf ms.\n",avg);

	delete [] X;
	delete [] Y; 
	return 0;
}


