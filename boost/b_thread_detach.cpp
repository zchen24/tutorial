/**
 * Demo how to detach a thread. This demo is identical to the
 * b_thread_interrupt except that the t.detach() is called, hence
 * t.interrupt() after it has no effect.
 *
 * "When a thread is detached, the boost::thread object ceases to represent
 * the now detached thread, and instead represents Not-a-Thread."
 *
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
    t.detach();
    // now t does NOT represent the thread
    // hence, calling interrupt does not interrupt the thread
    t.interrupt();
    t.join();
    std::cout << "interrupted thread t\n";
    return 0;
}
