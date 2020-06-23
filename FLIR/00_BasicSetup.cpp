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
    auto version = system->GetLibraryVersion();
    auto cam_list = system->GetCameras();

    std::cout << "Spinnaker Version: " << version.major << "."
              << version.minor << "." << version.type << "\n";

    if (cam_list.GetSize() == 0) {
        std::cerr << "No cameras found\n";
        cam_list.Clear();
        system->ReleaseInstance();
        return -1;
    } else {
        std::cout << cam_list.GetSize() << " Cameras Found\n";
    }
    auto cam = cam_list.GetByIndex(0);
    cam->Init();
    std::cout << "Fancy processing\n";
    cam->DeInit();

    // Release reference to the camera
    cam = nullptr;
    cam_list.Clear();
    system->ReleaseInstance();
    return 0;
}

