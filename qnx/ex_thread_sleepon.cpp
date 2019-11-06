//
// QNX Thread Barrier Example
// from QNX Neutrino 2 Book
// Date: 2019-10-25
//

#include <stdint.h>
#include <pthread.h>
#include <malloc.h>
#include <iostream>
#include <unistd.h>

pthread_barrier_t barrier;

void* read_thread1(void *)
{
    time_t now;
    time(&now);
    printf("read_thread1 starting at %s", ctime(&now));
    sleep(5);
    pthread_barrier_wait(&barrier);
    time(&now);
    printf("barrier in read_thread1 done at %s", ctime(&now));
    return nullptr;
}

void* thread2(void*)
{
    time_t now;
    time(&now);
    printf("thread2 starting at %s", ctime(&now));
    sleep(10);
    pthread_barrier_wait(&barrier);
    time(&now);
    printf("barrier in thread2 done at %s", ctime(&now));
    return nullptr;
}


int main(int argc, char** argv)
{
    time_t now;
    pthread_t t1, t2;

    // create a battier object
    pthread_barrier_init(&barrier, nullptr, 3);

    // start up two threads
    pthread_create(&t1, nullptr, read_thread1, nullptr);
    pthread_create(&t2, nullptr, thread2, nullptr);

    // t1, t2 are running
    time(&now);
    printf("main() waiting for barrier at %s", ctime(&now));
    pthread_barrier_wait(&barrier);

    // after barrier, all threads are done
    time(&now);
    printf("barrier in main() done at %s", ctime(&now));
}
