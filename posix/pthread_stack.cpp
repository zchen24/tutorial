// Posix thread how to
// 2020-11-19
// Reference:
// https://computing.llnl.gov/tutorials/pthreads/


#include <iostream>
#include <thread>

#define N_THREADS 5

void* func(void*) {
    std::cout << "Hello from func\n";
    return nullptr;
}


int main(int argc, char** argv)
{
    std::cout << "Hello std::thread" << "\n";

    pthread_t threads[N_THREADS];
    size_t stacksize;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_getstacksize(&attr, &stacksize);
    // Linux 5.4.0 x64: 0x800000
    printf("Default stack size = 0x%lx\n", stacksize);

    pthread_attr_setstacksize(&attr, 0x400000);
    pthread_attr_getstacksize(&attr, &stacksize);
    printf("Setting stack size = 0x%lx\n", stacksize);
    for (unsigned long & thread : threads) {
        auto rc = pthread_create(&thread, &attr, func, nullptr);
        if (rc){
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    printf("Created %d threads.\n", N_THREADS);
    pthread_exit(nullptr);
}
