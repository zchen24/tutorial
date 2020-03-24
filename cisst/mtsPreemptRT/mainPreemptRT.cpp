// Brief:
// This example shows how to configure a cisst mtsTaskPeriodic component
// to get the best timing performance on a PREEMPT_RT patched Linux kernel.
//
// Author: Zihan Chen
// Date: 2020-03-24

#include <cisstMultiTask.h>   // mtsManagerLocal
#include <cisstCommon.h>
#include "sys/mman.h"
#include "mtsPreemptRT.h"

int main( int, const char** )
{
    // Lock memory to prevent paging
    mlockall(MCL_CURRENT | MCL_FUTURE);

    mtsManagerLocal* componentManager = mtsManagerLocal::GetInstance();

    double periodInSec = 1.0 * cmn_ms;
    bool isHardRealtime = true;

    mtsPreemptRT rt("Test", periodInSec, isHardRealtime);
    rt.Configure("");
    componentManager->AddComponent(&rt);

    componentManager->CreateAll();
    componentManager->StartAll();

    int key = ' ';
    std::cout << "Press 'q' to quit\n";
    while (key != 'q') {
        key = cmnGetChar();
    }

    componentManager->KillAll();
    componentManager->Cleanup();

    // stop all logs
    cmnLogger::Kill();

    return 0;
}


