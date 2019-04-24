// From CUDA for Engineers
// Listing 8.3: dist_1d_thrust

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <iostream>

#define N 64

using namespace thrust::placeholders;  // _1

struct SqrtOf{
    __host__ __device__
    float operator()(float x) {
        return sqrt(x);
    }
};

int main()
{
    const float ref = 0.5f;
    thrust::device_vector<float> d_x(N);
    thrust::device_vector<float> d_dist(N);

    thrust::sequence(d_x.begin(), d_x.end());
    thrust::transform(d_x.begin(), d_x.end(), d_x.begin(), _1/(N-1));
    thrust::transform(d_x.begin(), d_x.end(), d_dist.begin(), (_1 - ref)*(_1 - ref));
    thrust::transform(d_dist.begin(), d_dist.end(), d_dist.begin(), SqrtOf());
    thrust::host_vector<float> h_x = d_x;
    thrust::host_vector<float> h_dist = d_dist;
    for (int i = 0; i < N; i++) {
        printf("x=%3.3f, dist=%3.3f\n", h_x[i], h_dist[i]);
    }

    std::cout << "dist_1d_thrust\n";
    return 0;
}
