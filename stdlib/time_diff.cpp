//! @brief demo how to get time difference
//! @author Zihan Chen
//! @date 2018-12-22

// Reference:
// https://www.techiedelight.com/find-execution-time-c-program/

#include <ctime>
#include <chrono>
#include <iostream>
#include <thread>
#include <sys/time.h>


int main(int argc, char** argv)
{
    // -----------------------
    // cpp: high_resolution
    // -----------------------
    using namespace std::chrono;
    high_resolution_clock::time_point t0, t1;

    t0 = high_resolution_clock::now();
    std::this_thread::sleep_for(milliseconds(5));
    t1 = high_resolution_clock::now();
    duration<double> dt = duration_cast<duration<double>>(t1 - t0);
    std::cout << "dt = " << dt.count() <<  "\n";

    // -----------------------
    // c: time_t resolution second
    // -----------------------
    time_t tt0, tt1;
    time(&tt0);
    std::this_thread::sleep_for(seconds(2));
    time(&tt1);
    std::cout << "dt = " <<  difftime(tt1, tt0) << "\n";

    // ---------------------------------------
    // c: clock_t
    // NOTE: not accurate
    //      or I don't know how to use it ...
    // ---------------------------------------
    clock_t c0, c1;
    c0 = clock();
    std::this_thread::sleep_for(seconds(1));
    c1 = clock();

    std::cout << "c0 = " << c0 << "  c1 = " << c1 << "\n";
    std::cout << "dt = " <<  ((double) (c1 - c0))/CLOCKS_PER_SEC << " s \n";


    // ---------------------------------------
    // c: gettimeofday (UNIX ONLY)
    //    <sys/time.h>
    // ---------------------------------------
    struct timeval tv0{0,0}, tv1{0,0};
    gettimeofday(&tv0, nullptr);
    std::this_thread::sleep_for(milliseconds(6));
    gettimeofday(&tv1, nullptr);
    std::cout << "dt = " << (tv1.tv_sec - tv0.tv_sec) * 1000 + (tv1.tv_usec - tv0.tv_usec) / 1e3
              << " ms\n";


    // -----------------------
    // c: clock_gettime
    // -----------------------
    struct timespec ts0 = {0,0};
    struct timespec ts1 = {0,0};
    clock_gettime(CLOCK_REALTIME, &ts0);
    std::this_thread::sleep_for(milliseconds(4));
    clock_gettime(CLOCK_REALTIME, &ts1);
    std::cout << "dt = " << (ts1.tv_sec - ts0.tv_sec) * 1000 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6
              << " ms\n";

    return 0;
}
