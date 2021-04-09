// AN485: Intel CPUID Instruction Note
// g++ cpuid.cpp -o cpuid
// 

#include <iostream>

static inline void native_cpuid(unsigned int *eax, unsigned int *ebx,
                                unsigned int *ecx, unsigned int *edx)
{
        /* ecx is often an input as well as an output. */
        asm volatile("cpuid"
            : "=a" (*eax),
              "=b" (*ebx),
              "=c" (*ecx),
              "=d" (*edx)
            : "0" (*eax), "2" (*ecx));
}

int main(int, char**)
{
  std::cout << "cpuid\n";
  unsigned int eax, ebx, ecx, edx;

  eax = 0;
  native_cpuid(&eax, &ebx, &ecx, &edx);
  printf("max function #: %d\n", eax);

  eax = 1;
  native_cpuid(&eax, &ebx, &ecx, &edx);
  printf("stepping %d\n", eax & 0xF);
  printf("model %d\n", (eax >> 4) & 0xF);
  printf("family %d\n", (eax >> 8) & 0xF);
  printf("processor type %d\n", (eax >> 12) & 0x3);
  printf("extended model %d\n", (eax >> 16) & 0xF);
  printf("extended family %d\n", (eax >> 20) & 0xFF);

  eax = 3;
  native_cpuid(&eax, &ebx, &ecx, &edx);
  printf("SN 0x%08x%08x\n", edx, ecx);

  // AN485: Table 5-12
  eax = 6;
  native_cpuid(&eax, &ebx, &ecx, &edx);
  printf("==============\n");
  printf("DTS\t\t: %d\n", eax & 0x1);
  printf("Turbo\t\t: %d\n", (eax >> 1) & 0x1);
  printf("ARAT\t\t: %d\n", (eax >> 2) & 0x1);
  printf("PLN\t\t: %d\n", (eax >> 4) & 0x1);
  printf("ECMD\t\t: %d\n", (eax >> 5) & 0x1);
  printf("PTM\t\t: %d\n", (eax >> 6) & 0x1);
  printf("#Interrupt\t: %d\n", ebx & 0xF);
  printf("#HW Coordinate\t: %d\n", ecx & 0x1);
  printf("#ACNT2\t\t: %d\n", (ecx >> 1) & 0x1);
  printf("#PERF_BIAS\t: %d\n", (ecx >> 3) & 0x1);  
}
