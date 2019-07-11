/**
 * Shows how to use std::chrono library
 *
 * Author: Zihan Chen
 * Date: 2019-07-10
 *
 * BSD License
 */

#include <iostream>
#include <chrono>
#include <thread>


int main(int argc, char** argv)
{
    auto d10 = std::chrono::milliseconds(10);
    auto d20 = std::chrono::milliseconds(20);
    auto diff = d20 - d10;
    std::cout << "20ms - 10ms = " << diff.count() << "ms" << "\n";

    
    // get time 0
    auto t0 = std::chrono::high_resolution_clock::now();
    // sleep for 20 ms
    std::this_thread::sleep_for(d20);
    // get time 1
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "d_time after 20ms sleep = "
              << std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() << " s\n";
}