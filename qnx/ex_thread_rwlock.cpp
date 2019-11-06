//
// QNX Thread Rwlock Example
// from QNX Neutrino 2 Book
// Date: 2019-10-26
//

#include <stdint.h>
#include <pthread.h>
#include <iostream>
#include <unistd.h>


int value;
pthread_rwlock_t lock;


void* read_thread1(void *)
{
    for (auto i = 0; i < 10; i++) {
        pthread_rwlock_rdlock(&lock);
        printf("read thread1 value: %d\n", value);
        pthread_rwlock_unlock(&lock);
    }
    return nullptr;
}

void* read_thread2(void*)
{
    for (auto i = 0; i < 10; i++) {
        pthread_rwlock_rdlock(&lock);
        printf("read thread2 value: %d\n", value);
        pthread_rwlock_unlock(&lock);
    }
    return nullptr;
}


int main(int argc, char** argv)
{
    value = 0;
    pthread_t t1, t2;
    pthread_rwlock_init(&lock, nullptr);

    pthread_create(&t1, nullptr, read_thread1, nullptr);
//    pthread_create(&t2, nullptr, read_thread2, nullptr);

    printf("main thread value: %d\n", value);
    for (auto i = 0; i < 10; i++) {
        pthread_rwlock_wrlock(&lock);
        value++;
        pthread_rwlock_unlock(&lock);
    }
    printf("main thread value: %d\n", value);

    pthread_join(t1, nullptr);
//    pthread_join(t2, nullptr);
    pthread_rwlock_destroy(&lock);
}
