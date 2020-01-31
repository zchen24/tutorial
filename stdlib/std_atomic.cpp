/**
 * Shows how to use std::atomic
 *    1) without atomic, the counter value is < 50000 (race condition)
 *    2) with atomic, the counter value is 50000
 *
 * Author: Zihan Chen
 * Date: 2020-01-30
 *
 * BSD License
 */

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

int counter = 0;
std::atomic<int> a_counter(0);

void increment() {
    for (int i = 0; i < 500; i++) {
        counter++;
    }
}

void a_increment() {
    for (int i = 0; i < 500; i++) {
        a_counter++;
    }
}


int main(int, char**)
{
    constexpr size_t num_threads = 100;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(increment));
    }
    // sync
    for (auto& t:threads) { t.join();}
    std::cout << "Counter = " << counter << "\n";

    threads.clear();
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(a_increment));
    }
    for (auto& t:threads) { t.join();}
    std::cout << "Atomic Counter = " << a_counter << "\n";

    return 0;
}
