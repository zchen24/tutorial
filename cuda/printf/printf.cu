// From CUDA for Engineering
// norm/kernel.cu
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_kernel() 
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char** argv)
{
    print_kernel << <5, 5 >> > ();
    cudaDeviceSynchronize();
    return 0;
}