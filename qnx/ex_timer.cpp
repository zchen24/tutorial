//
// QNX Timer Example
// Date: 2019-10-22
//

#include <iostream>
#include <time.h>
#include <signal.h>
#include <sys/siginfo.h>

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

void thread_func(sigval param)
{
    static int thread_count = 0;
    std::cout << "Starting thread " << thread_count++ << "  param = " << param.sival_int << "\n";
}


int main(int argc, char** argv)
{
    std::cout << "QNX Timer Examples\n";


    // -------------------------------
    // Example 1: Timer with signal
    // -------------------------------

    signal( SIGALRM, sig_alarm_handler );

    struct sigevent se;
    timer_t timer_id;

    SIGEV_SIGNAL_INIT(&se, SIGALRM);
    if (timer_create(CLOCK_REALTIME, &se, &timer_id) < 0) {
        std::cerr << "can't create timer. errno = " << errno << "\n";
    }

    struct itimerspec value;
    value.it_value.tv_sec = 1;
    value.it_value.tv_nsec = 0;
    value.it_interval.tv_sec = 1;
    value.it_interval.tv_nsec = 0;

    timer_settime(timer_id,
                  0,   // relative
                  &value,
                  nullptr);

    // -------------------------------
    // Example 2: Timer with thread
    // -------------------------------
    struct sigevent se_thread;
    SIGEV_THREAD_INIT(&se_thread, thread_func, 199, NULL);
    timer_t  timer_thread;
    if (timer_create(CLOCK_REALTIME, &se_thread, &timer_thread) < 0) {
        std::cerr << "can't create timer thread, errno = " << errno << "\n";
    }
    value.it_value.tv_sec = 2;
    value.it_value.tv_nsec = 0;
    value.it_interval.tv_sec = 0;
    value.it_interval.tv_nsec = 0;

    timer_settime(timer_thread,
                  0,
                  &value,
                  nullptr);
    while (true) {
    }
}
