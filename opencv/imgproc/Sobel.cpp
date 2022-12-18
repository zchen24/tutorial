#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main(int argc, char** argv)
{
  cv::Mat src, src_gray;
  std::string window_name = "Sobel Demo";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  src = cv::imread("./imgs/lena.jpg");
  if (!src.data) {
    std::cerr << "Error: can not read file" << std::endl;
    return -1;
  }

  cv::GaussianBlur(src, src, cv::Size(3, 3), 0);
  cv::cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);

  /// Generate grad_x and grad_y
  cv::Mat grad;
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
//  cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
//  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  cv::imshow(window_name, abs_grad_x);

  cv::waitKey(0);
  return 0;
}
