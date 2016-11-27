#include <iostream>

#define N 8

using namespace std;

/*CUDA kernel*/
__global__ void square(float *d_out, float *d_in)
{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f*f;
}

int main(int argc, char*argv[] )
{

	/*set up memory on the host*/
	float h_in[N];
	float h_out[N];
	for(int i=0; i<N; i++)
		h_in[i] = (float)i+10;

	/*set up memory on the device*/
	float *d_in, *d_out;
	cudaMalloc((void**) &d_in, N*sizeof(float));
	cudaMalloc((void**) &d_out, N*sizeof(float));

	/*transfer to device*/
	cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);

	/*launch kernel*/
	square<<<1, N>>>(d_out, d_in);
	cudaDeviceSynchronize();

	/*transfer to host*/
	cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    
	cudaFree(d_in);
	cudaFree(d_out);
    
	/*print results*/
	for(int i=0; i<N; i++)
		cout << "h_in[" << i << "]^2 = " << h_in[i] << "^2 = " << h_out[i] << endl;
	
	return EXIT_SUCCESS;
}
