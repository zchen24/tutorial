// From CUDA for Engineer
// Listing D.5

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    cout << "Number of devices = " << numDevices << "\n";
    for (int i = 0; i < numDevices; i++) {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "Device Number: " << i << "\n"
             << "Device name: " << prop.name << "\n"
             << "Compute capability: " << prop.major << "." << prop.minor << "\n"
             << "MultiProcessor count: " << prop.multiProcessorCount << "\n"
             << "Maximum threads/block: " << prop.maxThreadsPerBlock << "\n"
             << "Shared memory/block: " << prop.sharedMemPerBlock/1024.0 << " KBytes\n"
             << "Total global memory: " << prop.totalGlobalMem/1e9 << " Gbs\n"
             << "Total constant memory: " << prop.totalConstMem/1024.0 << " KBytes\n";        
    }

    return 0;
}