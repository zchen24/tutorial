//
// QNX Clock Example
// Date: 2019-10-22
//

#include <iostream>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/neutrino.h>


int main(int argc, char** argv)
{
    std::cout << "QNX Clock Examples\n";

    // -------------------------------
    // Posix Calls
    // -------------------------------
    struct timespec time_now;
    int ret = clock_gettime(CLOCK_REALTIME, &time_now);
    std::cout << "time now = " << time_now.tv_sec + time_now.tv_nsec/1000000000.0 << "\n";
    sleep(1);

    // ------------------------------------
    // ClockAdjust: current time of day
    // ------------------------------------


    // ------------------------------------
    // ClockPeriod: base timing resolution
    // ------------------------------------
    struct _clockperiod base_period{0, 0};

    // read
    ret = ClockPeriod(CLOCK_REALTIME,
                      nullptr,            // new
                      &base_period,       // old
                      0);
    std::cout << "initial base_period = " << base_period.nsec << "\n";

    // write
    struct _clockperiod new_base_period{
        .nsec = 10000,
        .fract = 0
    };
    if (ClockPeriod(CLOCK_REALTIME, &new_base_period, nullptr, 0) < 0) {
        std::cerr << "Failed to set base period\n";
    } else {
        std::cout << "setting new base_period, ret = " << ret << "\n";
        ret = ClockPeriod(CLOCK_REALTIME,
                          nullptr,            // new
                          &base_period,       // old
                          0);
        std::cout << "new base_period = " << base_period.nsec << "\n";
    }

    // ------------------------------------
    // ClockCycles: CPU high-res cycles
    // ------------------------------------
    uint64_t cycles_0 = ClockCycles();
    for (auto i = 0; i < 10; i++) {}
    uint64_t cycles_1 = ClockCycles();
    std::cout << "cycles_0 = " << cycles_0 << "\n"
              << "d_cycles = " << cycles_1 - cycles_0 << "\n";
}
