#include <iostream>
#include <string.h>
#include <poll.h>
#include <unistd.h>
#include <stdio.h>

//! @brief Use poll & fgets to read keyboard input non-blockingly
//! @author Zihan Chen
//! @date 2018-10-13


int main(int argc, char** argv)
{
    char buffer[1024];
    struct pollfd fds{
        .fd = STDIN_FILENO,
        .events = POLLIN,
    };
    int timeout_ms = 10;
    // set timeout_ms to -1 for blocking call

    while (true) {
        int ret = poll(&fds, 1, timeout_ms);
        if (ret < 0) {
            std::cerr << "Failed to poll stdin\n";
        }
        else if (ret == 0) {
            // no data found
            continue;
        }
        else {
            if (fgets(buffer, 1024, stdin) != nullptr) {
                std::cout << "1st key = " << buffer[0] << "\n";
                if (buffer[0] == 'q') {
                    break;
                }
            } else {
                std::cerr << "fgets returns NULL\n";
            }
        }
    }
}