//  
//  cvImage: basice image operation
//     1) crop (ROI)
//     2) create mat from existing memory
//     3) flip an image
//
//  Zihan Chen 
//  2018-03-22

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

int main()
{
    std::cout << "Basic Image Operations" << std::endl;

    cv::Mat img, dst;
    img = cv::imread("../imgs/lena.jpg");
    cv::imshow("original", img);

    // 1) crop image (non-copy)
    int x_start = 50;
    int y_start = 50;
    int width = 350;
    int height = 350;
    cv::Rect roi(x_start, y_start, width, height);
    cv::Mat cropRef(img, roi);
    cv::imshow("cropref", cropRef);


    // 2) Create image from existing memory
    unsigned char* img_memory;
    int img_memory_size = img.cols * img.rows * 3;   // 3 channel BGR image
    img_memory = (unsigned char*)malloc(img_memory_size);
    cv::Mat img_from_memory(img.rows, img.cols, CV_8UC3, img_memory);

    // set a rectangle to blue, via cv::Mat
    for (int i = 100; i < 200; i++) {
        for (int j = 100; j < 200; j++) {
            img_from_memory.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
        }
    }

    // now using img_memory to access
    // i is horizontal 
    // j is vertical
    for (int i = 200; i < 300; i++) {
        for (int j = 200; j < 300; j++) {
            img_memory[(j * img.cols + i) * 3] = 0;
            img_memory[(j * img.cols + i) * 3 + 1] = 0;
            img_memory[(j * img.cols + i) * 3 + 2] = 255;
        }
    }
    cv::imshow("img from memory", img_from_memory);


    // 3) flip
    cv::Mat img_hflip, img_vflip, img_flip_both;

    // horizontal flip
    cv::flip(img, img_hflip, 1);
    cv::flip(img, img_vflip, 0);
    cv::flip(img, img_flip_both, -1);

    cv::imshow("hflip", img_hflip);
    cv::imshow("vflip", img_vflip);
    cv::imshow("flip both", img_flip_both);


  cv::waitKey(0);

  return 0;
}
