// From CUDA for Engineers
// Listing 5.5: dd_1d_shared/kernel.cu

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

#define TPB 64
#define RAD 1


__global__
void ddKernel(float *d_out, const float *d_in, int size, float h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > size) return;
    if (i == 0 || i == (size-1)) {d_out = 0; return;}

    const int s_idx = threadIdx.x + RAD;
    extern __shared__ float s_in[];

    s_in[s_idx] = d_in[i];

    // halo cells
    if (threadIdx.x < RAD) {
        s_in[s_idx - RAD] = d_in[i - RAD];
        s_in[s_idx + blockDim.x] = d_in[i + blockDim.x];
    }


    // sync & out
    __syncthreads();
    d_out[i] = (s_in[s_idx+1] + s_in[s_idx-1] - 2.0f*s_in[s_idx]) / (h*h);
}

void ddParallel(float *out, const float *in, int n, float h)
{
    float *d_out = 0;
    float *d_in = 0;
    cudaMalloc(&d_out, n*sizeof(float));
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMemcpy(d_in, in, n*sizeof(float), cudaMemcpyHostToDevice);

    // set shared memory size in bytes
    const size_t smemsize = (TPB + 2*RAD)*sizeof(float);
    ddKernel<<<(n+TPB-1)/TPB, TPB, smemsize>>>(d_out, d_in, n, h);

    cudaMemcpy(out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);
}


int main(){
    std::cout << "dd_1d_shared\n";

    const float PI = 3.1415916;
    const int N = 150;
    const float h = 2*PI/N;

    float x[N] = {0.0f};
    float u[N] = {0.0f};
    float result_parallel[N] = {0.0f};
    for (int i = 0; i < N; i++) {
        x[i] = i * (2*PI/N);
        u[i] = sinf(x[i]);
    }

    ddParallel(result_parallel, u, N, h);

    std::ofstream outfile;
    outfile.open("results.csv");
    // x[i]   u[i]   d2u/d2x[i]          u[i] + d2u/d2x[i]
    // u = sin(x)    d2u/d2x = -sin(x)   u + d2u/d2x = 0.0
    for (int i = 0; i < N; i++) {
        outfile << x[i] << ", " << u[i] << ", " <<
                result_parallel[i] << ", " << result_parallel[i] + u[i] << "\n";
    }
    outfile.close();
}
