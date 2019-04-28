// From CUDA for Engineering
// norm/kernel.cu
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <math.h>
#include <stdio.h>

#define N (128*128)

int main(int argc, char** argv)
{
    thrust::device_vector<float> dvec_x(N, 1.0f);
    float norm = sqrt(thrust::inner_product(dvec_x.begin(), dvec_x.end(), dvec_x.begin(), 0.0f));
    printf("norm = %.0f\n", norm);
    return 0;
}
