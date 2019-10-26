//
// QNX Kernel Timeout Example
// Date: 2019-10-23
//

#include <iostream>
#include <time.h>
#include <signal.h>
#include <sys/siginfo.h>
#include <sys/neutrino.h>

void sig_alarm_handler( int sig_number )
{
    static int sig_count = 0;
    if (sig_number == SIGALRM) {
        std::cout << "Received SIGALARM " << sig_count++ << "\n";

        if (sig_count == 10) {
            exit(0);
        }
    }
}

void* thread_func(void*)
{
    static int thread_count = 0;
    std::cout << "Starting thread " << thread_count++ << "\n";
}


int pthread_join_non_blocking(int tid, void **rval)
{
    TimerTimeout(CLOCK_REALTIME, _NTO_TIMEOUT_JOIN,
                 nullptr, nullptr, nullptr);
    return (pthread_join(tid, rval));
}


int main(int argc, char** argv)
{
    std::cout << "QNX Kernel Timeout Examples\n";

    // -------------------------------
    // with pthread_join
    // -------------------------------
    pthread_t thread_id;
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_create(&thread_id, &thread_attr, thread_func, nullptr);


    uint64_t timeout;
    struct sigevent event;
    int rval;

    //unblock
    SIGEV_UNBLOCK_INIT(&event);

    timeout = 100LL * 1000000000LL;
    TimerTimeout(CLOCK_REALTIME,
                 _NTO_TIMEOUT_JOIN,
                 &event,
                 &timeout,
                 nullptr);
    rval = pthread_join(thread_id, nullptr);
    if (rval == ETIMEDOUT) {
        std::cout << "Thread " << thread_id << " still running after 10 seconds!\n";
    }
}
