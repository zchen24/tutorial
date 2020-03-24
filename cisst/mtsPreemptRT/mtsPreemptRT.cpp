// Periodic Component for PREEMPT_RT Linux
// Author: Zihan Chen
// Date: 2020-03-24

#include "mtsPreemptRT.h"
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstOSAbstraction/osaCPUAffinity.h>


CMN_IMPLEMENT_SERVICES_DERIVED(mtsPreemptRT, mtsComponent);


// ====================== Constructors and Destructors ========================
// ----------------------------------------------------------------------------
mtsPreemptRT::mtsPreemptRT(const std::string & componentName, double periodInSecs, bool isHardRealTime):
    mtsTaskPeriodic(componentName, periodInSecs, isHardRealTime),
    mCounter(0)
{
}

// ----------------------------------------------------------------------------

// =========================== Essential Functions ============================
void mtsPreemptRT::Configure(const std::string& filename)
{
    printf("Configured: %s\n", filename.c_str());
}

// ----------------------------------------------------------------------------
void mtsPreemptRT::Startup(void)
{
    printf("Startup called\n");

    // This function is called after thread Create() is called
    // and right before Run() starts. Set CPU Affinity, Priority
    // and Scheduling Policy here
    osaCPUSetAffinity(OSA_CPU2);

    // Note: for Linux, these two calls use pthread_setschedparam()
    // hence needs to happen after thread create (see man page)
    Thread.SetSchedulingPolicy(SCHED_FIFO);
    Thread.SetPriority(99);
}

// ----------------------------------------------------------------------------
void mtsPreemptRT::Cleanup(void)
{

}

void mtsPreemptRT::Run()
{
    mCounter++;
    if (mCounter % 100 == 0) {
        printf("Stats: min: %f, max: %f, mean: %f, std: %f\n",
               StateTable.PeriodStats.PeriodMin(),
               StateTable.PeriodStats.PeriodMax(),
               StateTable.PeriodStats.PeriodAvg(),
               StateTable.PeriodStats.PeriodStdDev()
        );
    }
}