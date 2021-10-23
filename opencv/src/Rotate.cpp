//  
//  cvRotate: rotate an image
//
//  Zihan Chen 
//  2021-09-30

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <chrono>

int main() {
    std::cout << "Basic Image Operations" << std::endl;

    using namespace std::chrono;
    high_resolution_clock::time_point t0, t1;
    cv::Mat img, dst;
    img = cv::imread("../imgs/left_01.png");
    cv::imshow("original", img);
    t0 = high_resolution_clock::now();
    for (size_t i = 0; i < 1000; i++) {
        cv::rotate(img, dst, cv::ROTATE_180);
    }
    t1 = high_resolution_clock::now();
    duration<double> dt = duration_cast<duration<double>>(t1 - t0);
    std::cout << "dt = " << dt.count()/1000*1000 <<  "ms \n";
    cv::imshow("180", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
