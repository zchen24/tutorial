//
// Created by zihan on 4/23/18.
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>


int main()
{
    std::cout << "cv callback pattern" << std::endl;

    // CV_WINDOW_NORMAL: all user change window size
    // CV_GUI_NORMAL: no toolbar
    cv::namedWindow("win", CV_WINDOW_NORMAL | CV_GUI_NORMAL);

    // Set property to fullscreen
    cv::setWindowProperty("win", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    cv::Mat img = cv::imread("./imgs/lena.jpg");
    cv::imshow("win", img);
    cv::waitKey(0);
}