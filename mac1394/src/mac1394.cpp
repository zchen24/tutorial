#include <iostream>
//#include <


#ifdef __APPLE__
#include <IOKit/firewire/IOFireWireLib.h>
#define bswap_32 OSSwapInt32
#endif

void ReadCallback(void*, IOReturn)
{
    std::cout << "read callback" << std::endl;
}

int main(int argc, char** argv)
{
//    std::cout << "mac firewire port sample code" << std::endl;

    kern_return_t result;
    mach_port_t masterPort;
    result = IOMasterPort(MACH_PORT_NULL, &masterPort);
    std::cout << "IOMasterPort return = " << result << std::endl;

//    CFMutableDictionaryRef matchingDic = IOServiceMatching("IOFireWireLocalNode");
    CFMutableDictionaryRef matchingDic = IOServiceMatching("IOFireWireDevice");

    io_iterator_t iterator;
    result = IOServiceGetMatchingServices(masterPort,
                                          matchingDic,
                                          &iterator);

    io_object_t aDevice;
    int count = 0;
    while ( (aDevice = IOIteratorNext(iterator)) != 0 )
    {
        count++;

        // get a device interface for the device
        IOCFPlugInInterface** cfPlugInInterface = 0;
        IOReturn result;
        SInt32 theScore;

        result = IOCreatePlugInInterfaceForService(aDevice,
                                                   kIOFireWireLibTypeID,
                                                   kIOCFPlugInInterfaceID,
                                                   &cfPlugInInterface,
                                                   &theScore);

        IOFireWireLibDeviceRef  fwDeviceInterface = 0;

        (*cfPlugInInterface)->QueryInterface( cfPlugInInterface,
                                              CFUUIDGetUUIDBytes(kIOFireWireDeviceInterfaceID_v9 ),
                                              (void**) &fwDeviceInterface );

        std::cout << "FireWireDevice version = " << (*fwDeviceInterface)->version << std::endl;

//        (*fwDeviceInterface)->CreateReadQuadletCommand()
        FWAddress addr(0x0000, 0x00000000);
        UInt32 dataValue = 0;
        UInt32 generation;

        // Get bus generation
        (*fwDeviceInterface)->GetBusGeneration(fwDeviceInterface,
                                               &generation);
//        std::cout << "generation = " << generation << std::endl;

        // Get dvice nodeid (16 bits)
        (*fwDeviceInterface)->GetRemoteNodeID(fwDeviceInterface,
                                              generation,
                                              &addr.nodeID);
//        addr.nodeID += 1;
        std::cout << "nodeid = " << std::hex << addr.nodeID << std::endl;

        // Get device
        io_object_t device;
        device = (*fwDeviceInterface)->GetDevice(fwDeviceInterface);


        // Create  ReadCommandInterface



        // Now let's read quadlet
        (*fwDeviceInterface)->Open(fwDeviceInterface);
        (*fwDeviceInterface)->ReadQuadlet(fwDeviceInterface,
                                          device,
                                          &addr,
                                          &dataValue,
                                          true,
                                          generation);
        std::cout << "address node = " << addr.nodeID
                  << "  hi = " << addr.addressHi
                  << "  low = " << addr.addressLo
                  << std::dec
                  << "  gen = " << generation << std::endl;
        std::cout << "quadlet data = " << std::hex << bswap_32(dataValue) << std::endl;

        // Read 0x01
        addr.addressLo = 0x04;
        (*fwDeviceInterface)->ReadQuadlet(fwDeviceInterface,
                                          device,
                                          &addr,
                                          &dataValue,
                                          true,
                                          generation);

        std::cout << "address node = " << addr.nodeID
                  << "  hi = " << addr.addressHi
                  << "  low = " << addr.addressLo
                  << std::dec
                  << "  gen = " << generation << std::endl;
        std::cout << "quadlet data = " << std::hex << bswap_32(dataValue) << std::endl;



        // block read
        const int maxReadBufferSize = 100;
        UInt8 readBuffer[maxReadBufferSize];
        UInt32 readQuadletdSize = 20;
        UInt32 readSize = readQuadletdSize * 4;
        for(int i = 0; i < maxReadBufferSize; i++) {
            readBuffer[i] = 0;
        }
        (*fwDeviceInterface)->Read(fwDeviceInterface,
                                   device,
                                   &addr,
                                   readBuffer,
                                   &(readSize),
                                   true,
                                   generation);

        // convert UInt8 to UInt32
        UInt32 readQuadletBuffer[readQuadletdSize];
        for (int i = 0; i < readQuadletdSize; i++) {
            readQuadletBuffer[i] = 0;
            for (int j = 0; j < 4; j++) {
                readQuadletBuffer[i] += (readBuffer[i*4+j] << ((3-j)*8));
            }
            std::cout << "block data = " << std::hex << (readQuadletBuffer[i]) << std::endl;
        }

        (*fwDeviceInterface)->Close(fwDeviceInterface);
        (*fwDeviceInterface)->Release(fwDeviceInterface);

    }

    std::cout << "total number of count = " << count << std::endl;
}
