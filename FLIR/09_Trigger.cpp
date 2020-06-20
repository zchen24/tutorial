/*
 * Shows how to use trigger feature.
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

    // turn off trigger mode
    cam->TriggerMode.SetValue(TriggerMode_Off);
    cam->TriggerSource.SetValue(TriggerSource_Software);
    cam->TriggerMode.SetValue(TriggerMode_On);
    cam->AcquisitionMode.SetValue(AcquisitionMode_SingleFrame);

    cam->BeginAcquisition();

    cam->TriggerSoftware.Execute();
    auto img = cam->GetNextImage(1000);
    uint64_t timestamp = img->GetTimeStamp();
    img->Release();
    cam->EndAcquisition();

    // reset trigger mode to off
    cam->TriggerMode.SetValue(TriggerMode_Off);

    // Release reference to the camera
    cam = nullptr;
    cam_list.Clear();
    system->ReleaseInstance();
    return 0;
}
