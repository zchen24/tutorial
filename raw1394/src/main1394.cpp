
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <byteswap.h>
#include <libraw1394/raw1394.h>
#include <bitset>

// Declare handle here
raw1394handle_t handle;


/* signal handler cleans up and exits the program */
void signal_handler(int sig) {
    signal(SIGINT, SIG_DFL);
    raw1394_destroy_handle(handle);
    exit(0);
}


/* bus reset handler updates the bus generation */
int reset_handler(raw1394handle_t hdl, unsigned int gen) {
    int id = raw1394_get_local_id(hdl);
    printf("Bus reset to gen %d, local id %x\n", gen, id);

    std::cout << "hello in reset handler" << std::endl;
    raw1394_update_generation(hdl, gen);
    return 0;
}

// asynchronous read/write handler
int arm_handler(raw1394handle_t handle, unsigned long arm_tag,
                byte_t request_type, unsigned int requested_length,
                void *data)
{

    std::cout << "called " << std::endl;
}

int main(int argc, char** argv)
{
    // ----- Set linux signal handler -----
    signal(SIGINT, signal_handler);

    // ----- Get handle and set port for the handle -------

    // create handle
    handle = raw1394_new_handle();
    if (handle == NULL) {
        std::cerr << "**** Error: could not open 1394 handle" << std::endl;
        return -1;
    }

    // set the bus reset handler
    raw1394_set_bus_reset_handler(handle, reset_handler);
    // set the bus arm_tag handler
    raw1394_set_arm_tag_handler(handle, arm_handler);


    // get port info & save to portinfo
    const int maxports = 4;
    raw1394_portinfo portinfo[maxports];
    int numPorts = raw1394_get_port_info(handle, portinfo, maxports);

    // display port info
    for (size_t i = 0; i < numPorts; i++) {
        std::cout << "port " << portinfo[i].name
                  << "  has " << portinfo[i].nodes << " nodes" << std::endl;
    }

    // set port to handle, port should be 0 to numPorts-1
    int port = 0;
    raw1394_set_port(handle, port);


    // ------ Get number of node --------
    int numNodes = raw1394_get_nodecount(handle);
    std::cout << numNodes << " nodes" << std::endl;

    /**
      * nodeid_t is a 16 bits value,
      *   16 = 10 + 6
      *   higher 10 bits are bus ID
      *   lower 6 bits are local node number (physical ID)
      *   physical ID are cynamic and determined during bus reset
      */
    nodeid_t localID = raw1394_get_local_id(handle);
    std::cout << "Local ID = " << std::hex << localID << std::endl;

    // get nodeid_t for the node (here node physical id = 0)
    int node = 0;
    nodeid_t targetNodeID = (localID & 0xFFC0) + node;

    // -------- get config rom info ---------
    int rc;    // return code
    size_t romBufferSize = 100;
    quadlet_t romBuffer[romBufferSize];
    size_t rom_size;
    unsigned char rom_version[100];
    rc = raw1394_get_config_rom(handle,
                                romBuffer,
                                romBufferSize,
                                &rom_size,
                                rom_version);

    std::cout << "rom_size = " << rom_size
              << "  rom_version = " << rom_version << std::endl;
    for (size_t i = 0; i < rom_size; i++) {
//        std::cout << std::hex << bswap_32(romBuffer[i]) << std::endl;
    }


    // -------  Asynchronous read -----------

    // we are ready for read
    int size;  // read/write size
    const int maxBufferSize = 100;  // max 100 quadlets
    quadlet_t readBuffer[maxBufferSize];

    // quadlet read
    size = 1;
    rc = raw1394_read(handle, targetNodeID, 0, size*4, readBuffer);
    for (size_t i = 0; i < size; i++) {
        std::cout << "quadlet read = " << std::hex << bswap_32(readBuffer[i]) << std::endl;
    }

    // block read





//    // ------ Test bswap_32 ---------
//    quadlet_t value = 0x40880f00;
//    std::cout << std::hex << "value = " << value << std::endl
//              << "bswap(value) = " << bswap_32(value) << std::endl;

    while (1) {
        raw1394_loop_iterate(handle);
    }

    return 0;
}












