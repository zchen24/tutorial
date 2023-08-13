// From ChatGPT

// For SCHED_FIFO in the Linux kernel, 99 is the highest priority and
// 1 is the lowest priority. The higher the numerical value, the higher
// the priority of the task. So, a task with priority 99 will be scheduled
// before a task with priority 1.

//  SCHED_FIFO can be used only with static priorities higher than 0,
//       which means that when a SCHED_FIFO thread becomes runnable, it
//       will always immediately preempt any currently running
//       SCHED_OTHER, SCHED_BATCH, or SCHED_IDLE thread.

// Note: SCHED_IDLE has a priority of 0


#include <cstdio>
#include <sched.h>

int main() {
    int min_priority = sched_get_priority_min(SCHED_FIFO);
    int max_priority = sched_get_priority_max(SCHED_FIFO);

    printf("SCHED_FIFO priority range: %d to %d\n", min_priority, max_priority);
    return 0;
}