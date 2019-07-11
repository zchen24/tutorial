/**
 * Demo how to interrupt boost thread
 * 
 * Author: Zihan Chen
 * Date: 2019-07-04
 *
 * BSD License
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#define N (20)


void thread_handler()
{
    for (size_t i = 0; i < N; ++i) {
        std::cout << "Running thread handler: " << i << "\n";
        // this is an interruption point
        // there are other interruption points e.g. sleep_for()
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        boost::this_thread::interruption_point();
    }
}



int main(int argc, char *argv[])
{
    std::cout << "boost thread interrupt demo\n";
    boost::thread t{boost::bind(thread_handler)};
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    t.interrupt();
    t.join();
    std::cout << "interrupted thread t\n";
    return 0;
}
