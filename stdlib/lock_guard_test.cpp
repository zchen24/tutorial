//! @brief std::lock_guard example, this example has 3 parts.
//! @author Zihan Chen
//! @date 2018-10-13
//! See: https://en.cppreference.com/w/cpp/thread/lock_guard

// ----------------------------------
// Part I: no mutex
//   the expected value of i = 20 * num_threads = 20000
//   Without mutex protection, i is 19983 in an example run, which is < 20000.
//   This is a race condition.

// Part II: use mutex.lock/unlock
//   manually lock/unlock mutex

// Part III: use std::lock_guard
//   use std::lock_guard to lock and auto unlock the mutex


#include <thread>
#include <mutex>
#include <iostream>
#include <vector>


int g_i = 0;
std::mutex g_i_mutex;


void increment()
{
    for (size_t i = 0; i < 20; i++) {
        ++g_i;
        std::cout << std::this_thread::get_id() << ": " << g_i << "\n";
    }
}


void safe_increment()
{
    for (size_t i = 0; i < 20; i++) {
        g_i_mutex.lock();
        ++g_i;
        std::cout << std::this_thread::get_id() << ": " << g_i << "\n";
        g_i_mutex.unlock();
    }
}

void safe_increment_lock_guard()
{
    for (size_t i = 0; i < 20; i++) {
        std::lock_guard<std::mutex> lock(g_i_mutex);
        ++g_i;
        std::cout << std::this_thread::get_id() << ": " << g_i << "\n";
        // g_i_mutex is auto-released when lock is out of scope
    }
}

int main()
{
    std::cout << "main: " << g_i << "\n";

    const int num_thread = 1000;
    std::vector<std::thread> all_threads;
    for (int i = 0; i < num_thread; i++) {
        // use this line for part I
        all_threads.push_back(std::thread(increment));

        // use this line for part II
        // all_threads.push_back(std::thread(safe_increment));

        // use this line for part III
        // all_threads.push_back(std::thread(safe_increment_lock_guard));
    }

    for (auto& t : all_threads) {
        t.join();
    }

    std::cout << "main: " << g_i << "\n";
}