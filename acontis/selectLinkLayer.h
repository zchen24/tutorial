/*-----------------------------------------------------------------------------
 * selectLinkLayer.h
 * Copyright                acontis technologies GmbH, Weingarten, Germany
 * Response                 Paul Bussmann
 * Description              EC-Master link layer selection
 *---------------------------------------------------------------------------*/

#ifndef INC_SELECTLINKAYER
#define INC_SELECTLINKAYER 1

/*-INCLUDES------------------------------------------------------------------*/

#ifndef INC_ECOS
#include "EcOs.h"
#endif
#include "stdio.h"
#include "stdlib.h"
#ifndef INC_ECLINK
#include "EcLink.h"
#endif

#if (defined INCLUDE_DUMMY)
#include "EcLinkDummy.h"
#endif

#ifdef __INTIME__
#include <stdio.h>
#elif _MSC_VER
#include "warn_dis.h"
#include <windows.h>
#include <tchar.h>
#define TCHAR_DEFINED
#include <stdio.h>
#include "warn_ena.h"
#endif

/*-DEFINES-------------------------------------------------------------------*/
#if (defined EC_VERSION_ECOS)
 #define INCLUDE_EMLL_STATIC_LIBRARY
 #define INCLUDE_EMLLANTAIOS
#elif (defined EC_VERSION_FREERTOS)
 #define INCLUDE_EMLL_SOC_XILINX
#elif (defined EC_VERSION_GO32)
 #define INCLUDE_EMLLR6040
#elif (defined EC_VERSION_INTEGRITY)
 #define INCLUDE_EMLL_STATIC_LIBRARY
 #define INCLUDE_EMLL_PCI_ALL
#elif (defined EC_VERSION_INTIME)
 #define INCLUDE_EMLL_PCI_ALL
#elif (defined EC_VERSION_JSLWARE)
 #define INCLUDE_EMLL_SOC_TI
#elif (defined EC_VERSION_LINUX)
 #define INCLUDE_EMLL_PCI_ALL
 #define INCLUDE_EMLL_SOC_ALL
 #define INCLUDE_EMLLALTERATSE
 #define INCLUDE_EMLLSOCKRAW
#elif (defined EC_VERSION_QNX)
 #define INCLUDE_EMLL_PCI_ALL
 #define INCLUDE_EMLL_SOC_ALL
#elif (defined EC_VERSION_RIN32M3)
 #define INCLUDE_EMLLRIN32M3
#elif (defined EC_VERSION_RTEMS)
 #define INCLUDE_EMLL_STATIC_LIBRARY
 #define INCLUDE_EMLL_PCI_ALL
#elif (defined EC_VERSION_RTOS32)
 #if !(ATECAT_DLL)
  #define INCLUDE_EMLL_STATIC_LIBRARY
 #endif
 #define INCLUDE_EMLL_PCI_ALL
#elif (defined EC_VERSION_RTX)
 #define INCLUDE_EMLL_PCI_ALL
#elif (defined EC_VERSION_RTXC)
 #define INCLUDE_EMLL_STATIC_LIBRARY
 #define INCLUDE_EMLL_SOC_SYNOPSYS
#elif (defined EC_VERSION_RZGNOOS)
 #define INCLUDE_EMLLRZT1
#elif (defined EC_VERSION_RZT1)
 #define INCLUDE_EMLLRZT1
#elif (defined EC_VERSION_STARTERWARE)
 #define INCLUDE_EMLL_SOC_TI
#elif (defined EC_VERSION_SYLIXOS)
 #define INCLUDE_EMLL_PCI_INTEL
 #define INCLUDE_EMLLCPSW
#elif (defined EC_VERSION_SYSBIOS)
 #define INCLUDE_EMLL_STATIC_LIBRARY
 /* #define INCLUDE_EMLL_SOC_TI */ /* currently set in specific demo project file */
#elif (defined EC_VERSION_TKERNEL) || (defined EC_VERSION_ETKERNEL)
 #define INCLUDE_EMLL_STATIC_LIBRARY
 #define INCLUDE_EMLL_PCI_ALL
 #define INCLUDE_EMLLL9218I 
#elif (defined EC_VERSION_UC3)
 #define INCLUDE_EMLL_SOC_SYNOPSYS
#elif (defined EC_VERSION_UCOS)
 #define INCLUDE_EMLL_STATIC_LIBRARY
 #define INCLUDE_EMLL_SOC_NXP
#elif (defined EC_VERSION_VXWORKS)
 #define INCLUDE_EMLL_PCI_ALL
 #define INCLUDE_EMLL_SOC_ALL
 #define INCLUDE_EMLLSNARF
#elif (defined EC_VERSION_WINCE)
 #define INCLUDE_EMLL_PCI_ALL
 #define INCLUDE_EMLLNDISUIO
#elif (defined EC_VERSION_WINDOWS)
 #define INCLUDE_EMLL_PCI_ALL
 #define INCLUDE_EMLLUDP
 #define INCLUDE_EMLLWINPCAP
#elif (defined EC_VERSION_XENOMAI)
 #define INCLUDE_EMLL_PCI_ALL
 #define INCLUDE_EMLL_SOC_ALL
#elif (defined EC_VERSION_XILINX_STANDALONE)
 #define INCLUDE_EMLL_SOC_XILINX
#endif

#if (defined INCLUDE_EMLL_PCI_ALL)
 #define INCLUDE_EMLL_PCI_BECKHOFF
 #define INCLUDE_EMLL_PCI_INTEL
 #define INCLUDE_EMLL_PCI_REALTEK
#endif
#if (defined INCLUDE_EMLL_PCI_BECKHOFF)
 #define INCLUDE_EMLLCCAT
#endif
#if (defined INCLUDE_EMLL_PCI_INTEL)
 #define INCLUDE_EMLLEG20T
 #define INCLUDE_EMLLI8254X
 #define INCLUDE_EMLLI8255X
#endif
#if (defined INCLUDE_EMLL_PCI_REALTEK)
 #define INCLUDE_EMLLRTL8169
 #define INCLUDE_EMLLRTL8139
#endif

#if (defined INCLUDE_EMLL_SOC_ALL)
 #define INCLUDE_EMLL_SOC_NXP
 #define INCLUDE_EMLL_SOC_SYNOPSYS
 #define INCLUDE_EMLL_SOC_TI
 #define INCLUDE_EMLL_SOC_XILINX
#endif

#if (EC_ARCH == EC_ARCH_ARM)
 #if (defined INCLUDE_EMLL_SOC_NXP)
  #define INCLUDE_EMLLETSEC
  #define INCLUDE_EMLLFSLFEC
 #endif
 #if (defined INCLUDE_EMLL_SOC_SYNOPSYS)
  #define INCLUDE_EMLLDW3504
 #endif
 #if (defined INCLUDE_EMLL_SOC_RENESAS)
  #define INCLUDE_EMLLSHETH
 #endif
 #if (defined INCLUDE_EMLL_SOC_TI)
  #define INCLUDE_EMLLCPSW
  #define INCLUDE_EMLLICSS
 #endif
 #if (defined INCLUDE_EMLL_SOC_XILINX)
  #define INCLUDE_EMLLEMAC
  #define INCLUDE_EMLLGEM
 #endif
#endif

#if (EC_ARCH == EC_ARCH_ARM64)
 #if (defined INCLUDE_EMLL_SOC_XILINX)
  #define INCLUDE_EMLLGEM
 #endif
#endif

#if (EC_ARCH == EC_ARCH_PPC)
 #if (defined INCLUDE_EMLL_SOC_NXP)
  #define INCLUDE_EMLLETSEC
  #define INCLUDE_EMLLFSLFEC
 #endif
#endif

/*-TYPEDEFS------------------------------------------------------------------*/
#ifndef TCHAR_DEFINED
typedef char TCHAR;
#endif

/*-FUNCTION DECLARATION------------------------------------------------------*/
EC_T_CHAR* GetNextWord(EC_T_CHAR **ppCmdLine, EC_T_CHAR *pStorage);

EC_T_DWORD CreateLinkParmsFromCmdLine(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord, EC_T_LINK_PARMS** ppLinkParms
#if defined(INCLUDE_TTS)
                                      , EC_T_DWORD* pdwTtsBusCycleUsec /* [out] TTS Bus Cycle overrides original one when TTS is used */
                                      , EC_T_VOID** pvvTtsEvent        /* [out] TTS Cycle event. Should override original one when TTS is used */
#endif
                                    );
EC_T_BOOL ParseIpAddress(EC_T_CHAR* ptcWord, EC_T_BYTE* pbyIpAddress);

EC_T_VOID ShowLinkLayerSyntax1(EC_T_VOID);
EC_T_VOID ShowLinkLayerSyntax2(EC_T_VOID);

#if (defined INCLUDE_EMLL_STATIC_LIBRARY)
EC_PF_LLREGISTER DemoGetLinkLayerRegFunc(EC_T_CHAR* szDriverIdent);
#endif
#endif /* INC_SELECTLINKAYER */
 
/*-END OF SOURCE FILE--------------------------------------------------------*/
