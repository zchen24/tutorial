// Periodic Component for PREEMPT_RT Linux
// Author: Zihan Chen
// Date: 2020-03-24

#ifndef _mtsPreemptRT_h
#define _mtsPreemptRT_h

#include <cisstMultiTask/mtsTaskPeriodic.h>

class CISST_EXPORT mtsPreemptRT : public mtsTaskPeriodic
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_ALLOW_DEFAULT);
public:

    mtsPreemptRT(const std::string & componentName, double periodInSecs, bool isHardRealTime);
    ~mtsPreemptRT() = default;

    void Configure(const std::string & filename = "") override;
    void Startup() override ;
    void Cleanup(void);
    void Run() override;

protected:
    int mCounter;
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsPreemptRT);

#endif
