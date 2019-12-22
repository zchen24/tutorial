//  
//  cvCamera: open camera and show video stream 
//
//  Zihan Chen 
//  2018-03-22

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

int main()
{
  std::cout << "hello world" << std::endl;

  cv::VideoCapture cap(0);   // open default camera
  if (!cap.isOpened()) {
      cerr << "Failed to open camera\n";
      return -1;
  }

  cv::Mat frame;

  while (true) {
      cap >> frame;
      cv::imshow("raw", frame);
      if (cv::waitKey(30) >= 0) break;
  }

  return 0;
}
