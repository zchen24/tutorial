// CUDA For Engineers
// Listing 8.10: thrustpi/kernel.cu

#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <iostream>

#define N (1 << 20)

using namespace thrust;

int main()
{
    std::cout << "curand_pi\n";
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42ULL);

    // d_vec for x & y
    thrust::device_vector<float> d_x(N);
    thrust::device_vector<float> d_y(N);
    curandGenerateUniform(gen, thrust::raw_pointer_cast(d_x.data()), N);
    curandGenerateUniform(gen, thrust::raw_pointer_cast(d_y.data()), N);
    curandDestroyGenerator(gen);

    int insideCount =
      thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(
        d_x.begin(), d_y.begin())), thrust::make_zip_iterator(
        thrust::make_tuple(d_x.end(), d_y.end())),
        []__device__(const thrust::tuple<float, float> &el) {
          return (pow(thrust::get<0>(el)/RAND_MAX, 2) +
                  pow(thrust::get<1>(el)/RAND_MAX, 2)) < 1.f; });

//    int insideCount = count_if(make_zip_iterator(make_tuple(d_x.begin(), d_y.begin())),
//            make_zip_iterator(make_tuple(d_x.end(), d_y.end())),
//            []__device__(const tuple<float, float> &el){
//                return(pow(get<0>(el),2 + pow(get<1>(el),2)) < 1.0f);
//    });

    std::cout << "pi = " << (insideCount * 4.0f / N);
}
