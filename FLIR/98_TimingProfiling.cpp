/*
 * Get some timestamp data
 *
 * */

#include "Spinnaker.h"
#include <iostream>
#include <sys/time.h>

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

    // turn off trigger mode
    cam->TriggerMode.SetValue(TriggerMode_Off);
    cam->TriggerSource.SetValue(TriggerSource_Software);
    cam->TriggerMode.SetValue(TriggerMode_On);
    cam->AcquisitionMode.SetValue(AcquisitionMode_SingleFrame);

    cam->BeginAcquisition();

    struct timeval tv0, tv1, tv2;
    gettimeofday(&tv0, nullptr);
    cam->TriggerSoftware.Execute();
    gettimeofday(&tv1, nullptr);
    auto img = cam->GetNextImage(1000);
    gettimeofday(&tv2, nullptr);
    uint64_t timestamp = img->GetTimeStamp();
    img->Release();
    cam->EndAcquisition();

    std::cout << "dt01 = " << (tv1.tv_sec - tv0.tv_sec) * 1000 + (tv1.tv_usec - tv0.tv_usec) / 1e3 << " ms\n";
    std::cout << "dt02 = " << (tv2.tv_sec - tv0.tv_sec) * 1000 + (tv2.tv_usec - tv0.tv_usec) / 1e3 << " ms\n";
    // reset trigger mode to off
    cam->TriggerMode.SetValue(TriggerMode_Off);

    // Release reference to the camera
    cam->DeInit();
    cam = nullptr;
    cam_list.Clear();
    system->ReleaseInstance();
    return 0;
}
