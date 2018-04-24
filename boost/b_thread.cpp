#include <iostream>
#include <boost/bind.hpp>
#include <boost/thread.hpp>


void thread_handler()
{
    for (size_t i = 0; i < 10; i++) {
        std::cout << "Running thread handler: " << i << "\n";
    }
}


int main(int argc, char** argv)
{
    std::cout << "hello boost thread\n";
    boost::thread t{boost::bind(thread_handler)};
    t.join();
}