#include <fcntl.h>
#include <unistd.h>
#include <iostream>

int main(int argc, char const *argv[])
{
    /* code */
    int fd;
    fd = open("./tmpfile", O_RDWR | O_CREAT);
    if (fd < 0) {
        std::cerr << "Failed to open file\n";
        return -1;
    } else {
        write(fd, "This is message passing\n", 24);
        close(fd);
    }
    return 0;
}
