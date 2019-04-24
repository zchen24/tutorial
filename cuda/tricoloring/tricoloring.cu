// From CUDA for Engineers
// Chapter 08: tri-color rings

#define cimg_display 0

#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include "CImg.h"


int main()
{
    std::cout << "tricolor ring\n";
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1) {
        std::cerr << "Failed to find CUDA device. Exiting ... \n";
        return -1;
    }

    // load image
    cimg_library::CImg<unsigned char> image("Tricoloring.png");
    const int w = image.width();
    const int h = image.height();
    const int kNumCh = 3;
    Npp8u *hn_img = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));
    Npp8u *hn_sharp = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));  // host, npp
    Npp8u *hn_out = (Npp8u*)malloc(kNumCh*w*h*sizeof(Npp8u));  // host, npp

    // copy to hn_img
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; c++) {
            for (int ch = 0; ch < kNumCh; ch++) {
                hn_img[kNumCh*(r*w+c) + ch] = image(c, r, ch);
            }
        }
    }

    Npp8u *d_in = 0;
    Npp8u *d_sharp = 0;
    Npp8u *d_swap = 0;
    Npp8u *d_out = 0;
    cudaMalloc(&d_in, kNumCh*w*h*sizeof(Npp8u));
    cudaMalloc(&d_sharp, kNumCh*w*h*sizeof(Npp8u));
    cudaMalloc(&d_swap, kNumCh*w*h*sizeof(Npp8u));
    cudaMalloc(&d_out, kNumCh*w*h*sizeof(Npp8u));
    cudaMemcpy(d_in, hn_img, kNumCh*w*h*sizeof(Npp8u), cudaMemcpyHostToDevice);

    // sharpen
    const NppiSize oSizeROI = {w, h};

    // swap color channels
    const int aDstOrder[3] = {1, 2, 0};
    NppStatus ret  = nppiSwapChannels_8u_C3R(d_in, kNumCh*w*sizeof(Npp8u), d_swap, kNumCh*w*sizeof(Npp8u), oSizeROI, aDstOrder);

    // bitwise add
    const int nScaleFactor = 1;
    ret = nppiAdd_8u_C3RSfs(d_in, kNumCh*w*sizeof(Npp8u),   // src1
            d_swap, kNumCh*w*sizeof(Npp8u),                 // src2
            d_out, kNumCh*w*sizeof(Npp8u),                  // out
            oSizeROI, nScaleFactor);


    // save to host
    cudaMemcpy(hn_img, d_swap, kNumCh*w*h*sizeof(Npp8u), cudaMemcpyDeviceToHost);
    cudaMemcpy(hn_out, d_out, kNumCh*w*h*sizeof(Npp8u), cudaMemcpyDeviceToHost);

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; c++) {
            for (int ch = 0; ch < kNumCh; ch++) {
                image(c, r, ch) = hn_img[kNumCh*(r*w+c) + ch];
            }
        }
    }
    image.save_png("out_swap.png");

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; c++) {
            for (int ch = 0; ch < kNumCh; ch++) {
                image(c, r, ch) = hn_out[kNumCh*(r*w+c) + ch];
            }
        }
    }
    image.save_png("out.png");
    return 0;
}
