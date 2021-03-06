// From CUDA for Engineers
// Listing 4.3

#include <cuda_runtime.h>
#include <iostream>


#define W   500
#define H   500
#define TX  32      // thread per block along x
#define TY  32      // thread per block along y


__global__
void distanceKernel(float *d_out, int w, int h, float2 pos)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = r * w + c;
	if ( (c >= w) || (r >= h) ) return;

	// compute the distance
	d_out[i] = sqrtf((c - pos.x) * (c - pos.x) + (r - pos.y) * (r - pos.y));
}


int main()
{
    float *out = (float*)calloc(W*H, sizeof(float));
    float *d_out = 0;
    cudaMalloc(&d_out, W*H*sizeof(float));

    const float2 pos = {0.0f, 0.0f};
    const dim3 blockSize(TX, TY);
    const int bx = (W + TX - 1) / TX;
    const int by = (W + TY - 1) / TY;
    const dim3 gridSize = dim3(bx, by);

    distanceKernel<<<gridSize, blockSize>>>(d_out, W, H, pos);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, W*H*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "bx = " << bx << "  by = " << by << "\n";
    std::cout << "out[0] = " << out[0] << "   out[end] = " << out[W*H-1] << '\n';

    free(out);
    cudaFree(d_out);

    return 0;
}



