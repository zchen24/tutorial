/*
 * FLIR and OpenCV
 *
 * */

#include "Spinnaker.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;

int main(int /*argc*/, char** /*argv*/)
{
    std::cout << "FLIR: Acquisition\n";
    auto system = System::GetInstance();
    auto cam_list = system->GetCameras();

    if (cam_list.GetSize() == 0) {
        std::cerr << "No cameras found\n";
        cam_list.Clear();
        system->ReleaseInstance();
        return -1;
    }

    auto cam = cam_list.GetByIndex(0);
    cam->Init();
    std::cout << "Fancy processing\n";
    printf("*** IMAGE ACQUISITION ***\n");
    cam->AcquisitionMode.SetValue(AcquisitionMode_Continuous);
    cam->ExposureAuto.SetValue(ExposureAuto_Continuous);
    cam->GainAuto.SetValue(GainAuto_Continuous);
    cam->BalanceWhiteAuto.SetValue(BalanceWhiteAuto_Continuous);
    printf("Acquisition mode set to continuous...\n");

    cam->BeginAcquisition();

    while (true) {
        auto img = cam->GetNextImage(50);
        if (img->IsIncomplete()) {
            fprintf(stderr, "Image incomplete with image statue %d ...", img->GetImageStatus());
            continue;
        }
        int width = img->GetWidth();
        int height = img->GetHeight();
        cv::Mat cImg;
        cImg.create(height, width, CV_8UC1);
        memcpy(cImg.data, img->GetData(), width * height);
        cv::resize(cImg, cImg, cv::Size(), 0.7, 0.7);
        cv::imshow("Preview", cImg);
        int key = cv::waitKey(5) & 0xFF;
        if (key == 'q') {
            printf("Quitting preview");
            img->Release();
            break;
        }
        img->Release();
    }
    cam->EndAcquisition();

    // Release reference to the camera
    cam->DeInit();
    cam = nullptr;
    cam_list.Clear();
    system->ReleaseInstance();
    return 0;
}