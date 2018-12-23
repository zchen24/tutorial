// OpenCV InputArray and OutputArray
//
// Zihan Chen
// 2018-12-22
//
// Reference:
// https://blog.csdn.net/yang_xian521/article/details/7755101

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


void do_fancy_processing(cv::InputArray in, cv::OutputArray out)
{
    cv::Mat m_in = in.getMat();
    out.create(m_in.rows, m_in.cols, m_in.type());
    cv::Mat m_out = out.getMat();

    int rows = m_in.rows;
    int cols = m_in.cols;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            m_out.at<cv::Vec3b>(r, c) = m_in.at<cv::Vec3b>(r, c)/2.0;
        }
    }
}


int main(int argc, char** argv)
{
    std::cout << "Hello CV" << "\n";

    cv::Mat img = cv::imread("./imgs/lena.jpg");
    cv::Mat out;
    do_fancy_processing(img, out);

    cv::imshow("img", img);
    cv::imshow("out", out);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
