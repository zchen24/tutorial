#include <iostream>
#include "distv2kernel.h"
#define N 64

float scale(int i, int n) {
    return ((float)i) / (n - 1);
}

int main(int argc, char** argv)
{
	std::cout << "Hello CUDA\n";

    const float ref = 0.5f;
    float *in = (float*)calloc(N, sizeof(float));
    float *out = (float*)calloc(N, sizeof(float));

    for (int i = 0; i < N; i++) {
        in[i] = scale(i, N);
    }
    distanceArray(out, in, ref, N);

    std::cout << "out: ";
    for (int i = 0; i < N; i++) {
        std::cout << " " << out[i]; g
    }
    std::cout << "\n";

    free(in);
    free(out);
}