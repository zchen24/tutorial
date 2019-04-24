#include <iostream>
#include "dist_v2.h"
#define N 20000000

#define DEBUG 0

float scale(int i, int n) {
    return ((float)i) / (n - 1);
}

int main(int argc, char** argv)
{
    std::cout << "Hello CUDA\n";

    const float ref = 0.5f;
    auto in = (float*)calloc(N, sizeof(float));
    auto out = (float*)calloc(N, sizeof(float));

    for (int i = 0; i < N; i++) {
        in[i] = scale(i, N);
    }
    distanceArray(out, in, ref, N);

#if DEBUG
    std::cout << "out: ";
    for (int i = 0; i < N; i++) {
        std::cout << " " << out[i];
    }
    std::cout << "\n";
#endif

    free(in);
    free(out);
}