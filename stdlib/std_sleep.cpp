//! @brief demo how to use std::this_thread::sleep_for
//! @author Zihan Chen
//! @date 2018-12-22


#include <iostream>
#include <thread>

int main(int, char**)
{
    std::cout << "sleep_for example" << "\n";
    for (size_t i = 0; i < 10; i++) {
      std::cout << "sleep 500 ms " << i << "/" << 10 << "\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
