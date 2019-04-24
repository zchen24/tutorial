// From CUDA for Engineering
// dist_v1/kernel.cu

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_utils.h"

#define N   64   // array length
#define TPB 32   // threads per block

// Convert i from 0, ..., n-1 to 0-1
__device__ float scale(int i, int n)
{
    return ((float)i) / (n - 1);
}

// compute the distance between 2 points
__device__ float distance(float x1, float x2)
{
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__ void distanceKernel(float *d_out, float ref, int len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = scale(i, len);
    d_out[i] = distance(x, ref);
    printf("%2d: dist from %f to %f is %f. \n", i, ref, x, d_out[i]);
}


int main(int argc, char** argv)
{
    const float ref = 0.5f;
    float *out = (float*)calloc(N, sizeof(float));
    float *d_out = 0;
    cudaMalloc(&d_out, N*sizeof(float));

    distanceKernel<<<N/TPB, TPB>>> (d_out, ref, N);

    cudaMemcpy(out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "out: ";
    for (int i = 0; i < N; i++) {
        std::cout << " " << out[i];
    }
    std::cout << "\n";

    cudaFree(d_out);
    free(out);
    return 0;
}