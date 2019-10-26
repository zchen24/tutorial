//
// QNX Thread Join Example
// Date: 2019-10-25
//

#include <sys/syspage.h>
#include <stdint.h>
#include <pthread.h>
#include <malloc.h>
#include <iostream>


uint16_t num_cpus;


void* do_one_batch(void *c)
{
    std::cout << "doing one batch for cpu #" << (int)c << "\n";
}


int main(int argc, char** argv)
{
    int cpu;
    pthread_t *thread_ids;

    num_cpus = _syspage_ptr->num_cpu;
    thread_ids = (pthread_t*)malloc(sizeof(pthread_t)  * num_cpus);

    for (cpu = 0; cpu < num_cpus; cpu++) {
        pthread_create(&thread_ids[cpu],        // thread_id
                       nullptr,        // thread_attr
                       do_one_batch,
                       (void*) cpu);
    }

    // display results
    for (cpu = 0; cpu < num_cpus; cpu++) {
        pthread_join(thread_ids[cpu], nullptr);
    }
}
