#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>

using namespace std;

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);   // open default camera
    if (!cap.isOpened()) {
        cerr << "Failed to open camera\n";
        return -1;
    }

    cv::Mat frame;
    cv::Mat frame2;
    cv::Mat diff;
    cv::Mat diffGray;
    cv::Mat display;

    while (true) {
        cap >> frame;
        usleep(30000);
        cap >> frame2;
        diff = frame2 - frame;
        cv::cvtColor(diff, diffGray, cv::COLOR_BGR2GRAY);
        double sum = cv::sum(diffGray)[0];

        // Motion is detected when threshold is higher than a threshold e.g. 1.0%
        std::cout << "width = " << diffGray.cols
                  << "  height = " << diffGray.rows
                  << "  ele = " << int(diffGray.at<uchar >(10, 10))
                  << "  sum = " << sum
                  << "  per = " << sum / (diff.rows * diff.cols * 255) * 100 << "\n";

        cv::hconcat(frame, diff, display);
        cv::imshow("raw", frame);
        cv::imshow("diff", diffGray);
        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}
