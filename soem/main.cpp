/*
 * Example code for simple ECAT arm communication
 * */

#include <iostream>
#include "ethercat.h"

#define k_LEDVendorId 0x34E  // Infineon
#define k_LEDProductId 0xAA

typedef struct LEDOutput
{
    u_char LED1:1;  // 1-bit LED1
    u_char LED2:1;
    u_char LED3:1;
    u_char LED4:1;
    u_char LED5:1;
    u_char LED6:1;
    u_char LED7:1;
    u_char LED8:1;
} LEDOutput_t;

typedef struct LEDInput
{
    char BTN1:1;
    char BTN2:1;
} LEDInput_t;


boolean inOP;
char IOmap[4096];

int main(int argc, const char** argv)
{
	std::cout << "Simple SOEM communication\n";
    std::cout << "Usage: soem_ecat [eth0]\n";
    const char* ifname;
    int ret{-1};

    if (argc < 2) {
        std::cerr << "Missing interface name\n";
        return -1;
    } else {
        ifname = argv[1];
        std::cout << "Using interface " << ifname << "\n";
    }

    int i;
//    int oloop, iloop;
    int expectedWKC;
    volatile int wkc;
    LEDInput_t *in_LED;
    LEDOutput_t *out_LED;

    ret = ec_init(ifname);
    if (ret <= 0) {
        printf("No socket connection on %s\n Execute as root\n", ifname);
        return -1;
    }

    if (ec_config_init(FALSE) <= 0) {
        printf("No slaves found! Exiting\n");
        ec_close();
        return -1;
    } else {
        printf("%d slaves found and configured. \n", ec_slavecount);
        for (i = 1; i <= ec_slavecount; i++) {
            printf("Slave %d: %s connected, VendorID: 0x%x\n ProductID: 0x%x", i,
                ec_slave[i].name, ec_slave[i].eep_man, ec_slave[i].eep_id);
        }
    }

    int iomap_size =  ec_config_map(&IOmap);
    printf("IO map size = %d\n", iomap_size);
    // Check SAFE_OP state

    ec_configdc();
    uint16 state = ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);
    if (state != EC_STATE_SAFE_OP) {
        std::cerr << "Failed to switch to SAFE_OP state. Exiting\n";
        ec_close();
        return -1;
    }

    expectedWKC = ec_group[0].outputsWKC * 2 + ec_group[0].inputsWKC;
    printf("Calculated working counter: %d\n", expectedWKC);

    ec_slave[0].state = EC_STATE_OPERATIONAL;
    ec_send_processdata();
    ec_receive_processdata(EC_TIMEOUTRET);
    // request OP state for all slaves
    // write slave state, if slave = 0, to all slaves.
    ec_writestate(0);
    // wait for all slaves to reach OP state
    int checks = 200;
    do {
        ec_send_processdata();
        ec_receive_processdata(EC_TIMEOUTRET);
        ec_statecheck(0, EC_STATE_OPERATIONAL, 500000);
    }
    while (checks-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));

    if (ec_slave[0].state == EC_STATE_OPERATIONAL) {
        printf("Operational state reached for all slaves.\n");
        // cyclic loop
        inOP = TRUE;
        for (i = 0; i < 10000; i++) {
            ec_send_processdata();
            wkc = ec_receive_processdata(EC_TIMEOUTRET);
            if (expectedWKC < wkc) {
                std::cerr << "Expected WKC = %d, actual WKC = %d, aborting"
                          << expectedWKC << wkc << "\n";
                break;
            }
            printf("Processdata cycle %4d, WKC %d , O:", i, wkc);

            for (auto j = 1; j <= ec_slavecount; j++) {
                if ((ec_slave[j].eep_man == k_LEDVendorId) && (ec_slave[j].eep_id == k_LEDProductId))
                {
                    in_LED = (LEDInput_t*)ec_slave[j].inputs;
                    out_LED = (LEDOutput_t*)ec_slave[j].outputs;

                    if (i % 50 == 0) {
                        out_LED->LED3 ^= 0b1u;
                        out_LED->LED4 ^= 0b1u;
                        out_LED->LED5 ^= 0b1u;
                        out_LED->LED6 ^= 0b1u;
                        out_LED->LED7 ^= 0b1u;
                        out_LED->LED8 ^= 0b1u;
                    }
                    out_LED->LED1 = in_LED->BTN1;
                    out_LED->LED2 = in_LED->BTN2;
                }
            }
            osal_usleep(10000); // 100 hz
        }
        inOP = FALSE;
    }
    else {
        printf("Not all slaves reached operational state.\n");
        ec_readstate();
        for (i = 1; i <= ec_slavecount; i++) {
            if (ec_slave[i].state != EC_STATE_OPERATIONAL) {
                printf("Slave %d State=0x%2.2x StatusCode=0x%4.4x : %s\n", i,
                       ec_slave[i].state,
                       ec_slave[i].ALstatuscode,
                       ec_ALstatuscode2string(ec_slave[i].ALstatuscode));
            }
        }
    }

    // request INIT state for all slaves
    ec_slave[0].state = EC_STATE_INIT;
    ec_writestate(0);

    printf("End simple test, close socket\n");
    ec_close();
    std::cout << "End program\n";
	return 0;
}