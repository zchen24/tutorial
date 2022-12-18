#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates circle finding with the Hough transform.\n"
            "Usage:\n"
            "./houghcircles <image_name>, Default is pic1.png\n" << endl;
}

int main(int argc, char** argv)
{
    const char* filename = argc >= 2 ? argv[1] : "./imgs/pot.jpg";

    Mat img = imread(filename, 0);
    if(img.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }

    Mat cimg;
    medianBlur(img, img, 5);
    cvtColor(img, cimg, COLOR_GRAY2BGR);

//    imshow("detected circles", cimg);

    vector<Vec3f> circles;
    HoughCircles(img, circles, cv::HOUGH_GRADIENT, 1, 10,
                 100, 30, 20, 200 // change the last two parameters
                                // (min_radius & max_radius) to detect larger circles
                 );
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        circle( cimg, Point(c[0], c[1]), c[2], Scalar(0,0,255), 3);
        circle( cimg, Point(c[0], c[1]), 2, Scalar(0,255,0), 3);
    }

    imshow("detected circles", cimg);
    waitKey();

    return 0;
}
