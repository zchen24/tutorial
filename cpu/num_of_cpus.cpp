/**
 * Get number of CPUs
 *
 * Author: Zihan Chen
 * Date: 2021-11-11
 *
 * BSD License
 */

#include <iostream>
#include <thread>


int main(int, char**)
{
    const auto numCpus = std::thread::hardware_concurrency();
    std::cout << "Num of cpus = " << numCpus << '\n';
}
