#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// Contrast & Brightness
//   - loop through each pixel
//   - saturate_cast usage
//
// Reference: OpenCV tutorial
//   - Changing the contrast and brightness of an image!


int main(int argc, char *argv[])
{
  double alpha;   // simple contrast control
  double beta;    // simple brightness control

  cv::Mat image = cv::imread("./imgs/WindowsLogo.jpg");
  cv::Mat imageNew = cv::Mat::zeros(image.size(), image.type());

  alpha = 2;    // Enter alpha value [1.0-3.0]
  beta = 10;      // Enter beta value [0-100]

  // imageNew(i,j) = alpha * image(i,j) + beta
  CV_Assert(image.channels() == 3);
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      for (int c = 0; c < 3; c++) {
        imageNew.at<cv::Vec3b>(i, j)[c] =
            cv::saturate_cast<uchar>(alpha * image.at<cv::Vec3b>(i,j)[c] + beta);
      }
    }
  }

  // Create Windows
  cv::namedWindow("Original Image", 1);
  cv::namedWindow("New Image", 1);

  // Show stuff
  cv::imshow("Original Image", image);
  cv::imshow("New Image", imageNew);

  // Wait until user press some key
  cv::waitKey();

  return 0;
}
