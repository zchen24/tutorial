//
// QNX Msg Server Example
// Date: 2019-10-23
//

#include <stdio.h>
#include <iostream>
#include <process.h>
#include <string.h>
#include <sys/neutrino.h>


int main(int argc, char const *argv[])
{
    int rcv_id;     // client id
    int ch_id;      // channel id
    char message[512];

    // create a channel
    ch_id = ChannelCreate(0);
    int msg_count = 0;

    while (true) {
        // get message and print
        rcv_id = MsgReceive(ch_id, message, sizeof(message), nullptr);
        std::cout << "Msg " << msg_count++ << " from " << rcv_id << ": " << message << "\n";

        // reply
        strcpy(message, "This is the reply");
        MsgReply(rcv_id, EOK, message, sizeof(message));

        if (msg_count > 10) {break;}
    }

    ChannelDestroy(ch_id);
    return 0;
}
