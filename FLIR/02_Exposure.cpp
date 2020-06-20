/*
 * Shows how to manually change camera's exposure time.
 *
 * */


#include "Spinnaker.h"
#include <iostream>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;

void set_exposure(CameraPtr& cam, int exposure_ms)
{
    cam->ExposureAuto.SetValue(ExposureAuto_Off);
    auto exp_max = cam->ExposureTime.GetMax();
    if (exposure_ms > exp_max) {
        exposure_ms = exp_max;
        std::cerr << "Exposure set value exceeds limit, setting to max " << exp_max << "\n";
    }
    else if (exposure_ms <= 0) {
        exposure_ms = 1;
        std::cerr << "Exposure time must be a positive int value\n";
    }
    cam->ExposureTime.SetValue(exposure_ms);
    std::cout << "Set exposure to " << exposure_ms << " ms\n";
}


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
    std::vector<int> exposure_all{1000, 2000, 3000, 4000, 5000};
    for (const auto exposure : exposure_all) {
        set_exposure(cam, exposure);
        usleep(50000);
        auto img = cam->GetNextImage();
        if (!img->IsIncomplete()) {
            char file_name[100];
            sprintf(file_name, "Exposure_%d.jpg", exposure);
            img->Save(file_name);
        }
        img->Release();
    }

    cam->EndAcquisition();
    cam->ExposureAuto.SetValue(ExposureAuto_Continuous);

    // Release reference to the camera
    cam = nullptr;
    cam_list.Clear();
    system->ReleaseInstance();
    return 0;
}

