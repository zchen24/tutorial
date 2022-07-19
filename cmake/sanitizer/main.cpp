/**
 * Author: Zihan Chen
 * Date: 2022-07-18
 */

#include <iostream>
#include <thread>

int main()
{
    // For ASan - address sanitizer
//    char mem[100];
//    std::cout << "mem-101: " << mem[101] << "\n";

    // For TSan - thread sanitizer
    int X = 40;
    std::thread  t0([&] {X++;});
    std::thread  t1([&] {X++;});
    std::thread  t2([&] {X++;});
    std::thread  t3([&] {X++;});

    X = 43;
    t0.join();
    t1.join();
    t2.join();
    t3.join();
    std::cout << "X = " << X << "\n";
}
