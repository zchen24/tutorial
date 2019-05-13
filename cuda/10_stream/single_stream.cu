// CUDA by Example
// Ch10.4: using a single CUDA stream

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)


static void HandleError(cudaError_t err,
    const char *file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void kernel(int* d_a, int* d_b, int* d_c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (d_a[idx] + d_a[idx1] + d_a[idx2]) / 3.0f;
        float bs = (d_b[idx] + d_b[idx1] + d_b[idx2]) / 3.0f;
        d_c[idx] = (as + bs) / 2;
    }
}


int main()
{
    cudaDeviceProp prop;
    int device_id = 0;
    HANDLE_ERROR(cudaGetDevice(&device_id));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_id));
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, exiting...");
        return -1;
    }

    cudaEvent_t start, stop;
    float elapsedTime;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    cudaStream_t stream;
    HANDLE_ERROR(cudaStreamCreate(&stream));
    HANDLE_ERROR(cudaEventRecord(start));

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));
    cudaMallocHost(&h_a, FULL_DATA_SIZE * sizeof(int));
    cudaMallocHost(&h_b, FULL_DATA_SIZE * sizeof(int));
    cudaMallocHost(&h_c, FULL_DATA_SIZE * sizeof(int));

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        h_a[i] = rand();
        h_b[i] = rand();
    }

    // loop over full data in chunks
    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        HANDLE_ERROR(cudaMemcpyAsync(d_a, h_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream));
        HANDLE_ERROR(cudaMemcpyAsync(d_b, h_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
        kernel <<<N/256, 256, 0, stream>>> (d_a, d_b, d_c);
        HANDLE_ERROR(cudaMemcpyAsync(h_c+i, d_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    }
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time taken: %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaStreamDestroy(stream));
    HANDLE_ERROR(cudaFree(d_a));
    HANDLE_ERROR(cudaFree(d_b));
    HANDLE_ERROR(cudaFree(d_c));
    HANDLE_ERROR(cudaFreeHost(h_a));
    HANDLE_ERROR(cudaFreeHost(h_b));
    HANDLE_ERROR(cudaFreeHost(h_c));

    return 0;
}
