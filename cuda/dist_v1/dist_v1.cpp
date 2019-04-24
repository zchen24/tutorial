// From CUDA for Engineering
// dist_v1/main.cpp

#include <iostream>
#include <math.h>

#define N 64   // array length

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


int main(int argc, char** argv)
{
	std::cout << "dist_v1: sequential CPU implementation\n";
    float out[N] = { 0.0f };
    const float ref = 0.5f;

    for (int i = 0; i < N; i++) {
        float x = scale(i, N);
        out[i] = distance(x, ref);
    }

    std::cout << "out: ";
    for (auto o : out) {
        std::cout << " " << o;
    }
    std::cout << "\n";

    return 0;
}