//  
//  cvVideoOut: open camera and save video to a file
//
//  Zihan Chen 
//  2022-12-10

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
    cap >> frame;
    cv::VideoWriter out("tmp.mp4",
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        30,
                        cv::Size(frame.size().width, frame.size().height));

    while (true) {
        cap >> frame;
        cv::imshow("raw", frame);
        out.write(frame);
        if (cv::waitKey(5) == 'q') break;
    }

    cap.release();
    out.release();
    cv::destroyAllWindows();

    return 0;
}