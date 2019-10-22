#include <iostream>
#include <sched.h>

int main(int argc, char const *argv[])
{
    /* code */

    int pri_min = sched_get_priority_min(SCHED_FIFO);
    int pri_max = sched_get_priority_max(SCHED_FIFO);

    std::cout << "QNX Priority Info\n";
    std::cout << "FIFO min = " << pri_min << "  max = " << pri_max << "\n";
    return 0;
}
