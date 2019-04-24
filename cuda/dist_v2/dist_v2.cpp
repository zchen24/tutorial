// From CUDA for Engineering
// dist_v2/main.cpp

#include <iostream>
#include <cmath>

#define N 1000000000    // a large array size
#define DEBUG 0


// Convert i from 0, ..., n-1 to 0-1
float scale(int i, int n)
{
	return static_cast<float>(i) / (n - 1);
}

// compute the distance between 2 points
float distance(float x1, float x2)
{
	return static_cast<float>(sqrt((x2 - x1) * (x2 - x1)));
}


void distanceArray(float *out, float *in, float ref, int n)
{
	for (int i = 0; i < n; i++) {
		out[i] = distance(in[i], ref);
	}
}


int main(int argc, char** argv)
{
	auto in = (float*)calloc(N, sizeof(float));
	auto out = (float*)calloc(N, sizeof(float));
	float ref = 0.5f;

	for (int i = 0; i < N; i++) {
		in[i] = scale(i, N);
	}

	distanceArray(out, in, ref, N);

#if DEBUG
	std::cout << "dist_v2\n";
	std::cout << "out: ";
	for (int i = 0; i < N; i++) { std::cout << " " << out[i]; }
	std::cout << "\n";
#endif

	// free
	free(in);
	free(out);
}