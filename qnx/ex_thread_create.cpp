//
// QNX Thread Create Example
// Date: 2019-10-25
//

#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>


void* new_thread(void*)
{
    std::cout << "Hey, this is new_thread\n";
    sleep(5);
    return nullptr;
}


int main()
{
    std::cout << "pthread create\n";

    pthread_t t1;
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);
    // WARNING! must set priority, or else qnx default to 0 (idle)
    sched_param param{.sched_priority = 63};
    pthread_attr_setschedparam(&attr, &param);

    pthread_create(&t1, &attr, new_thread, nullptr);
}