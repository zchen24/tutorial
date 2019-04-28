// From CUDA for Engineers
// Listing 6.1: parallel_dot/kernel.cu


#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define TPB 64
#define ATOMIC 1   // 0 for non-atomic addition
#define N 1024


__global__
void dotKernel(int *d_res, const int *d_a, const int *d_b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int s_idx = threadIdx.x;

    // shared & sync
    __shared__ int s_prod[TPB];
    s_prod[s_idx] = d_a[i] * d_b[i];
    __syncthreads();

    // only pick one thread to do the sum & addition
    // we happen to pick the 1st thread
    if (s_idx == 0) {
        int blockSum = 0;
        for (int j = 0; j < blockDim.x; j++) {
            blockSum += s_prod[j];
        }
        printf("Block_%d, blockSum = %d\n", blockIdx.x, blockSum);

        if (ATOMIC) {
            atomicAdd(d_res, blockSum);
        } else {
            *d_res += blockSum;
        }
    }
}


void dotLauncher(int *res, const int *a, const int *b, int n)
{
    int *d_res;
    int *d_a = 0;
    int *d_b = 0;
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(int));
    cudaMalloc(&d_res, sizeof(int));

    cudaMemset(d_res, 0, sizeof(n));
    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

    dotKernel<<<(n+TPB-1)/TPB, TPB>>>(d_res, d_a, d_b, n);

    cudaMemcpy(res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
}


int main()
{
    std::cout << "parallel_dot: reduction\n";
    int cpu_res = 0;
    int gpu_res = 0;
    int *a = (int*)malloc(N*sizeof(int));
    int *b = (int*)malloc(N*sizeof(int));
    // initialize
    for (int i = 0; i < N; i++) {
        a[i] = 1;
        b[i] = 1;
    }

    // cpu compute
    for (int i = 0; i < N; i++) {
        cpu_res += a[i]*b[i];
    }
    std::cout << "cpu result = " << cpu_res << "\n";

    // gpu compute
    dotLauncher(&gpu_res, a, b, N);
    std::cout << "gpu result = " << gpu_res << "\n";

    free(a);
    free(b);
    return 0;
}