/*
 * Shows how to acquire an image.
 *
 * */


#include "Spinnaker.h"
#include <iostream>

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
    cam->DeInit();

    printf("*** IMAGE ACQUISITION ***\n");
    cam->AcquisitionMode.SetValue(AcquisitionMode_Continuous);
    printf("Acquisition mode set to continuous...");

    cam->BeginAcquisition();
    auto cam_serial_number = cam->DeviceSerialNumber.ToString();
    printf("SN = %s\n", cam_serial_number.c_str());
    const int NUM_IMAGES = 10;
    for (int i = 0; i < NUM_IMAGES; i++) {
        auto img = cam->GetNextImage();
        if (img->IsIncomplete()) {
            fprintf(stderr, "Image incomplete with image statue %d ...", img->GetImageStatus());
        }
        else {
            auto width = img->GetWidth();
            auto height = img->GetHeight();
            printf("Grabbed Image %d, width = %lu, height = %lu", i, width, height);
            auto img_converted = img->Convert(PixelFormat_Mono8, HQ_LINEAR);
            char filename[100];
            sprintf(filename, "Acquisition-{%s}-{%02d}.jpg", cam_serial_number.c_str(), i);
            img_converted->Save(filename);
            img->Release();
        }
    }
    cam->EndAcquisition();

    // Release reference to the camera
    cam = nullptr;
    cam_list.Clear();
    system->ReleaseInstance();
    return 0;
}

