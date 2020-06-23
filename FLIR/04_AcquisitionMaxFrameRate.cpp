/*
 * Shows how to stream video max speed (over 200 FPS)
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

    cam->AcquisitionMode.SetValue(AcquisitionMode_Continuous);
    printf("Acquisition mode set to continuous...\n");

    // Turn off all auto algorithms
    cam->ExposureAuto.SetValue(ExposureAuto_Off);
    cam->ExposureTime.SetValue(2000);
    cam->GainAuto.SetValue(GainAuto_Off);
    cam->Gain.SetValue(1.0);
    cam->BalanceWhiteAuto.SetValue(BalanceWhiteAuto_Off);

    // Set Acquisition frame rate
    cam->AcquisitionFrameRateEnable.SetValue(true);
    cam->AcquisitionFrameRate.SetValue(226);
    printf("Result frame rate = %f fps\n", cam->AcquisitionResultingFrameRate.GetValue());

    // Start acquisition
    cam->BeginAcquisition();
    const int NUM_IMAGES = 10;
    uint64_t t_start = 0;
    for (int i = 0; i < NUM_IMAGES; i++) {
        auto img = cam->GetNextImage();
        if (!img->IsIncomplete()) {
            if (t_start == 0) {
                t_start = img->GetTimeStamp();
            }
            printf("FrameID = %lu, Timestamp = %f ms\n",
                   img->GetFrameID(),
                   (img->GetTimeStamp() - t_start) / 1000000.0);
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

