// Author: Zihan Chen
// Date: 2023-08-13

// Design an experiment
//  - thread 1 - priority 10, runs continuously, runs for 500ms, then sleeps for 500ms
//  - thread 2 - priority 20, runs periodically, runs for 5ms, then sleeps for 10ms

#include <iostream>
#include <pthread.h>
#include <ctime>
#include <unistd.h>
#include <cstring>


#define NUM_SECONDS 30

void busy_wait_delay(int delay_ms) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (true) {
        clock_gettime(CLOCK_MONOTONIC, &end);
        if ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6 > delay_ms) {
            break;
        }
    }
}

void* thread_func_p20(void* arg) {
    cpu_set_t cpuset;

    // Clear the CPU set
    CPU_ZERO(&cpuset);
    // Add the desired core to the CPU set
    CPU_SET(5, &cpuset);

    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        perror("pthread_setaffinity_np");
        return NULL;
    }

    int count = 0;
    while(count++ < (NUM_SECONDS * 10)) {
        std::cout << "p20 busy " << count << "\n";
        // Busy for 5 ms
        busy_wait_delay(50);
        std::cout << "p20 sleep " << count << "\n";
        // Sleep for 5 ms
        usleep(50000);
    }
    return nullptr;
}

void* thread_func_p10(void* arg) {
    cpu_set_t cpuset;

    // Clear the CPU set
    CPU_ZERO(&cpuset);
    // Add the desired core to the CPU set
    CPU_SET(5, &cpuset);

    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        perror("pthread_setaffinity_np");
        return NULL;
    }

    int count = 0;
    while(count++ < NUM_SECONDS) {
        std::cout << "p10 busy\n";
        // Busy for 5 ms
        busy_wait_delay(500);

        std::cout << "p10 sleep\n";
        // Sleep for 5 ms
        usleep(500000);
    }
    return nullptr;
}

int main() {
    pthread_t thread_20, thread_10;
    pthread_attr_t attr;
    struct sched_param param;

    // Initialize thread_20 attributes
    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);

    // Set the priority
    param.sched_priority = 20;
    pthread_attr_setschedparam(&attr, &param);

    // Create the thread_20
    int ret = pthread_create(&thread_20, &attr, thread_func_p20, nullptr);
    if (ret != 0) {
        std::cerr << "Error creating the thread_20! ret = " << ret << "  " << strerror(ret) << std::endl;
        return 1;
    }

    param.sched_priority = 10;
    pthread_attr_setschedparam(&attr, &param);
    ret = pthread_create(&thread_10, &attr, thread_func_p10, nullptr);
    if (ret != 0) {
        std::cerr << "Error creating the thread_10! ret = " << ret << "  " << strerror(ret) << std::endl;
        return 1;
    }

    // Wait for the thread_20 to finish
    pthread_join(thread_20, nullptr);
    pthread_join(thread_10, nullptr);

    return 0;
}