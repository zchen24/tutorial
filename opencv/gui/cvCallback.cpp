//
// Created by zihan on 4/23/18.
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>


class MyClass {
public:
    explicit MyClass(){
        cv::namedWindow("win");
        cv::setMouseCallback("win", &MyClass::mouseCallback, this);
        cv::waitKey(0);
    }
    virtual ~MyClass() = default;

    static void mouseCallback(int event, int x, int y, int flags, void* param)
    {
        auto mc = static_cast<MyClass*>(param);
        mc->realMouseCallback(event, x, y, flags);
    }

private:
    void realMouseCallback(int event, int x, int y, int flags) {
        if (event == CV_EVENT_RBUTTONDOWN)
        {
            // convert to CVStereoView
            std::cout << "Right mouse button pressed\n";
        }
    }
};


int main()
{
    std::cout << "cv callback pattern" << std::endl;
    MyClass mc;
}