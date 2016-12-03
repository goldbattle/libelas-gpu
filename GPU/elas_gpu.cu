
/*CUDA kernel*/
__global__ void square(float *d_out, float *d_in)
{
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f*f;
}