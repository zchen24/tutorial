// Basic CUDA example
//
// 2018-12-21
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/core/cuda.hpp>

int main(int argc, char** argv)
{
    std::cout << "cuda101\n";
    std::cout << "coutn = " << cv::cuda::getCudaEnabledDeviceCount() << "\n";
    std::cout << "device = " << cv::cuda::getDevice() << "\n";

    cv::Mat img = cv::imread("./imgs/lena.jpg");
    cv::cuda::GpuMat g_img(img);
    cv::cuda::GpuMat g_gray;
    cv::cuda::cvtColor(g_img, g_gray, cv::COLOR_RGB2GRAY);
    cv::Mat img_gray(g_gray);

    cv::imshow("raw", img);
    cv::imshow("gray", img_gray);

    cv::waitKey(0);
    cv::destroyAllWindows();
}