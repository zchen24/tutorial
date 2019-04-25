// Listing 5.1: dd_1d_global/main.cpp

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define TPB 64   // thread per block


__global__
void ddKernel(float *d_out, const float *d_in, int size, float h)
{
    // on device, and hence do not have access to CPU memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > size) return;
    if (i == 0 || i == size-1) {
        d_out[i] = 0;
        return;
    }
    d_out[i] = (d_in[i+1] + d_in[i-1] - 2.0f*d_in[i])/(h*h);
}

void ddParallel(float *out, const float *in, int n, float h)
{
    // create device memory
    float *d_out = 0;
    float *d_in = 0;
    cudaMalloc(&d_out, n*sizeof(float));
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMemcpy(d_in, in, n*sizeof(float), cudaMemcpyHostToDevice);
    // call ddKernel
    ddKernel<<<(n+TPB-1)/TPB, TPB>>>(d_out, d_in, n, h);

    cudaMemcpy(out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
}


int main()
{
    const float PI = 3.1415926;
    const int N = 150;
    const float h = 2*PI/N;  //

    float x[N] = {0.0f};
    float u[N] = {0.0f};
    float result_parallel[N] = {0.0f};

    // initialize x & u
    for (int i = 0; i < N; i++) {
        x[i] = i * (2 * PI / N);
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
    std::cout << "dd_1d_global\n";
}
