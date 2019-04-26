// From CUDA for Engineering
// dist_v2/kernel.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define TPB 32
#define N 256000
#define M 5   // number of times to do cudaMemcpy

#define DEBUG 0

__device__ 
float distance(float x1, float x2)
{
    return sqrt((x2 - x1) * (x2 - x1));
}

__global__
void distanceKernel(float *d_out, float *d_in, float ref, int len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) { return; }
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
    struct timespec t0 = {0,0};
    struct timespec t1 = {0,0};
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int i = 0; i < M; i++) {
        cudaMemcpy(d_in, in, len * sizeof(float), cudaMemcpyHostToDevice);
    }
    clock_gettime(CLOCK_REALTIME, &t1);
    std::cout << "Data transfer time (ms) = " << (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6  << "\n";


    // call wrapper
    clock_gettime(CLOCK_REALTIME, &t0);
    distanceKernel<<<(len+TPB-1)/TPB, TPB>>>(d_out, d_in, ref, len);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &t1);
    std::cout << "Kernel time (ms) = " << (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6  << "\n";

    // memcpy from device
    cudaMemcpy(out, d_out, len * sizeof(float), cudaMemcpyDeviceToHost);

    // free cuda memory
    cudaFree(d_in);
    cudaFree(d_out);
}

float scale(int i, int n) {
    return ((float)i) / (n - 1);
}

int main()
{
    std::cout << "dist_v2_cuda\n";

    const float ref = 0.5f;
    float *in = (float*)calloc(N, sizeof(float));
    float *out = (float*)calloc(N, sizeof(float));

    for (int i = 0; i < N; i++) {
        in[i] = scale(i, N);
    }
    distanceArray(out, in, ref, N);

#if DEBUG
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < N; i++) {
        std::cout << "i = " << i << "\tin: " << in[i] << "\tout: " << out[i] << "\n";
    }
#endif

    free(in);
    free(out);
    return 0;
}