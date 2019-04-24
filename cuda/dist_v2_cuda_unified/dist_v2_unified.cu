// From CUDA for Engineering
// dist_v2_cuda_unified/kernel.cu

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define N   1000000000
#define TPB 32
#define DEBUG 0


float scale(int i, int n) {
    return ((float)i) / (n - 1);
}

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


int main()
{
    const float ref = 0.5f;
    float *in = 0;
    float *out = 0;

    // Allocate managed memory for in/out arrays
    cudaMallocManaged(&in, N* sizeof(float));
    cudaMallocManaged(&out, N* sizeof(float));

    for (int i = 0; i < N; i++) { in[i] = scale(i, N); }

    // launch kernel
    distanceKernel<<<N/TPB, TPB>>>(out, in, ref);
    cudaDeviceSynchronize();

#if DEBUG
    std::cout << "dist_v2_unified: cuda unified memory\n";
    std::cout << "out: ";
    for (int i = 0; i < N; i++) {
        std::cout << " " << out[i];
    }
    std::cout << "\n";
#endif

    cudaFree(in);
    cudaFree(out);
}
