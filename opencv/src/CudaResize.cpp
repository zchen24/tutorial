//
// Image Resize: cv::resize and cv::cuda::resize
//
// Zihan Chen
// 2018-06-19
// License: BSD
//


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudawarping.hpp>


int main(int argc, char** argv)
{
    cv::Mat img, dst;
    img = cv::imread("../imgs/lena.jpg");
    cv::imshow("original", img);
}

