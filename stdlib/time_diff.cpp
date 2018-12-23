//! @brief demo how to get time difference
//! @author Zihan Chen
//! @date 2018-12-22

#include <ctime>
#include <chrono>
#include <iostream>
#include <thread>


int main(int argc, char** argv)
{
    using namespace std::chrono;
    high_resolution_clock::time_point t0, t1;

    t0 = high_resolution_clock::now();
    std::this_thread::sleep_for(milliseconds(5));
    t1 = high_resolution_clock::now();
    duration<double> dt = duration_cast<duration<double>>(t1 - t0);
    std::cout << "dt = " << dt.count() <<  "\n";

    time_t tt0, tt1;
    time(&tt0);
    std::this_thread::sleep_for(seconds(2));
    time(&tt1);
    std::cout << "dt = " <<  difftime(tt1, tt0) << "\n";
    return 0;
}
