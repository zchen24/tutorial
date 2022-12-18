#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Press 'c' to set the border to a random constant value
// Press 'r' to set the border to be replicated
// Press 'q' to exit the program

int main(int argc, char *argv[])
{
  /// Create window
  std::string window_name = "copyMakeBoarder Demo";
  cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );

  cv::Mat src, dst;
  src = cv::imread("./imgs/lena.jpg");
  dst = src;

  // compute size argument
  int top, bottom, left, right;
  top = (int) (0.05 * src.rows); bottom = (int) (0.05 * src.rows);
  left = (int) (0.05 * src.cols); right = (int) (0.05 * src.cols);

  int boarderType = cv::BORDER_CONSTANT;
  cv::RNG rng;

  while (true)
  {
    int c = cv::waitKey(500);

    if ((char)c == 'q') {
      break;
    } else if ((char)c == 'c') {
      boarderType = cv::BORDER_CONSTANT;
    } else if ((char)c == 'r') {
      boarderType = cv::BORDER_REPLICATE;
    }

    cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
                                  rng.uniform(0, 255),
                                  rng.uniform(0, 255));
    cv::copyMakeBorder(src, dst, top, bottom, left, right,
                       boarderType, color);

    cv::imshow(window_name, dst);
  }

  return 0;
}
