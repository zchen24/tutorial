// CUDA by Example
// Ch10: page-locked (pinned) host memory

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

#define SIZE (10*1024*1024)

float cuda_malloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *a, *d_a;
    float d_t;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int*)malloc(size*sizeof(*a));
    cudaMalloc(&d_a, size * sizeof(*a));

    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) {
        if (up) {
            cudaMemcpy(d_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
        }
        else {
            cudaMemcpy(a, d_a, size * sizeof(*a), cudaMemcpyDeviceToHost);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d_t, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(a);
    cudaFree(d_a);

    return d_t;
}

float cuda_host_alloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *a, *d_a;
    float d_t;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaHostAlloc(&a, size*sizeof(*a), cudaHostAllocDefault);
    cudaMalloc(&d_a, size*sizeof(*a));

    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) {
        if (up) {
            cudaMemcpy(d_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
        }
        else {
            cudaMemcpy(a, d_a, size * sizeof(*a), cudaMemcpyDeviceToHost);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d_t, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFreeHost(a);
    cudaFree(d_a);
    return d_t;
}



int main()
{
    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
    
    elapsedTime = cuda_malloc_test(SIZE, true);
    printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
    printf("\t MB/s during copy up: %3.1f\n", MB/(elapsedTime/1000));

    elapsedTime = cuda_malloc_test(SIZE, false);
    printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
    printf("\t MB/s during copy down: %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, true);
    printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
    printf("\t MB/s during copy up: %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, false);
    printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
    printf("\t MB/s during copy down: %3.1f\n", MB / (elapsedTime / 1000));


    return 0;
}
