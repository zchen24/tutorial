//
// QNX Fork Example
// Date: 2019-10-22
//

#include <stdio.h>
#include <process.h>


int main(int argc, char const *argv[])
{
    pid_t retval;

    printf("This is most definitely the parent process\n");
    fflush(stdout);
    retval = fork();
    printf("Which process printed this? %d\n", retval);

    return 0;
}
