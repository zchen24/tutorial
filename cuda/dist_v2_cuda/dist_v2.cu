// From CUDA for Engineering
// dist_v2/kernel.cu

#include "dist_v2.h"
#include <stdio.h>
#include <cuda_runtime.h>
#define TPB 32

__device__ 
float distance(float x1, float x2)
{
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__
void distanceKernel(float *d_out, float *d_in, float ref)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = d_in[i];
    d_out[i] = distance(x, ref);
}

void distanceArray(float *out, float *in, float ref, int len)
{
    // alloc cuda memory
    float *d_in = 0;
    float *d_out = 0;
    cudaMalloc(&d_in, len * sizeof(float));
    cudaMalloc(&d_out, len * sizeof(float));

    // memcpy to device
    cudaMemcpy(d_in, in, len * sizeof(float), cudaMemcpyHostToDevice);

    // call wrapper
    distanceKernel<<<len/TPB, TPB>>>(d_out, d_in, ref);

    // memcpy from device
    cudaMemcpy(out, d_out, len * sizeof(float), cudaMemcpyDeviceToHost);

    // free cuda memory
    cudaFree(d_in);
    cudaFree(d_out);
}