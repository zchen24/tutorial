//! @brief demo how to use std::thread
//! @author Zihan Chen
//! @date 2018-12-22


#include <iostream>
#include <thread>


void func() {
    std::cout << "Hello from func\n";
}

void func_args(int a0, int a1) {
    std::cout << "Hello from func_args: "
              << a0 << "  " << a1 << "\n";
}

int main(int argc, char** argv)
{
    std::cout << "Hello std::thread" << "\n";

    std::thread t0(func);
    std::thread t1(func_args, 2, 4);

    // Without these, the main thread can finish before t0, t1
    // and will exit (killing the unfinished t0, t1).
    // -- Error: terminate called without an active exception
    t0.join();
    t1.join();
    return 0;
}
