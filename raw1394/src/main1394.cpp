
// system
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <byteswap.h>
#include <bitset>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h> // htonl

// libraw1394
#include <libraw1394/raw1394.h>
#include <libraw1394/csr.h>

// Declare handle here
raw1394handle_t handle;
const int maxBufferSize = 100;  // max 100 quadlets
quadlet_t readBuffer[maxBufferSize];


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

// tag handler
int my_tag_handler(raw1394handle_t handle, unsigned long tag,
                raw1394_errcode_t errcode)
{
    int err = raw1394_errcode_to_errno(errcode);
    if (err) {
        std::cerr << "failed with error: " << strerror(err) << std::endl;
    } else {
        std::cout << "completed customized tag handler = " << bswap_32(readBuffer[0]) << std::endl;
    }
}


// asynchronous read/write handler
int arm_handler(raw1394handle_t handle, unsigned long arm_tag,
                byte_t request_type, unsigned int requested_length,
                void *data)
{
    std::cout << "arm_handler called " << std::endl
              << "req len = " << requested_length << " type = " << request_type << std::endl;
}



// arm tag callback
int my_arm_req_callback(raw1394handle_t handle,
                        struct raw1394_arm_request_response *arm_req_resp,
                        unsigned int requested_length,
                        void *pcontext, byte_t request_type)
{
    std::cout << "arm_req_callback, type = " << std::dec << (int)request_type << std::endl;
    // RAW1394_ARM_READ = 1,  WRITE = 2, LOCK = 4
    raw1394_arm_request *req = arm_req_resp->request;
    std::cout << "tcode: " << (int)req->tcode
              << "  tlabel: " << (int)req->tlabel
              << "  destid: 0x" << std::hex << req->destination_nodeid
              << "  sourid: 0x" << std::hex << req->source_nodeid
              << "  ext_tcode: " << std::dec << (int)req->extended_transaction_code << std::endl;

    quadlet_t value = 0x5432;

    quadlet_t *resp_quad_read = new quadlet_t[4];
    resp_quad_read[0] =
            ((req->source_nodeid & 0xffff) << 16) +
            ((req->tlabel & 0x3f) << 10) +
            (6 << 4);  // tcode = 6: read response
    resp_quad_read[1] =
            ((req->destination_nodeid & 0xffff) << 16) +
            ((0 & 0xf) << 12);  // rcode = 0 = reap_complete
//    resp_quad_read[2] reserved
    resp_quad_read[3] = htonl(value);

    raw1394_arm_get_buf(handle, 0, 4, readBuffer);
    std::cout << "Address 0x0 has been written value = " << std::hex << bswap_32(readBuffer[0]) << std::endl;

    raw1394_arm_get_buf(handle, CSR_REGISTER_BASE + 0x4000, 4, readBuffer);
    std::cout << "Address broadcast has been written value = " << std::hex << bswap_32(readBuffer[0]) << std::endl << std::endl;

    delete resp_quad_read;
    return 0;
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
    // this will override the arm_tag_handler
//    raw1394_set_arm_tag_handler(handle, arm_handler);


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

    // --------------------------------------------
    // ------------- ARM Handling -----------------
    const int maxArmBufferSize = 300;
    uint8_t armBuffer[maxArmBufferSize];
    size_t armSize = 10;
    int mode = RAW1394_ARM_WRITE | RAW1394_ARM_READ;
    byte_t configROM[4] = {0x01, 0x04, 0x02, 0x09};
    raw1394_arm_reqhandle reqHandle;

    char my_arm_callback_context[] = "my_arm_callback_context";


    reqHandle.arm_callback = my_arm_req_callback;
    reqHandle.pcontext = my_arm_callback_context;

    // regitster arm to handle request for address 0x00
    rc = raw1394_arm_register(handle, 0, 4, configROM,
                              (unsigned long) &reqHandle, mode, mode, 0);
    if (rc) {
        std::cerr << "addr = 0x00 arm register: " << strerror(errno) << std::endl;
    }

    // register arm handler for broadcast
    rc = raw1394_arm_register(handle, CSR_REGISTER_BASE + 0x4000, 4, configROM,
                              (unsigned long) &reqHandle, mode, mode, 0);
    if (rc) {
        std::cerr << "addr = csr arm register: " << strerror(errno) << std::endl;
    }


    // get memory
    rc = raw1394_arm_get_buf(handle, 0, 4, armBuffer);
    if (rc) {
        std::cerr << "arm get buf: " << strerror(errno) << std::endl;
    } else {
        for (size_t i = 0; i < 4; i++) {
            std::cout << "arm buf " << std::dec << i << " = "
                      << std::hex << bswap_32(armBuffer[i]) << std::endl;
        }
    }


    // loop iterate to pull events
    while (1) {
        raw1394_loop_iterate(handle);
    }

    return 0;
}











