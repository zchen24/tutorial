#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int main(int argc, char *argv[])
{
  /// Create window
  cv::Mat src, dst;
  src = cv::imread("../imgs/lena.jpg");
  dst = src;

  cv::Mat img_grey; 
  cv::Mat img_yuv;

  cv::cvtColor(src, img_grey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(src, img_yuv, cv::COLOR_BGR2YUV);

  cv::imshow("original", dst);
  cv::imshow("grey", img_grey);
  cv::imshow("yuv", img_yuv);

  /// Wait until user exit program by pressing a key
  cv::waitKey(0);
  return 0;
}
