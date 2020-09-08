/*
 * Example code for EtherCAT master using Acontis
 * Slave Device: XMC4800 Relax Kit with LED example
 * */

// To debug gdb:
// set args -i eth0 -f xmc48_led.xml
// run

#include <iostream>
#include <AtEthercat.h>
#include <cisstCommon.h>
#include "EcOs.h"
#include "Logging.h"
#include "ecatNotification.h"
#include "EcatDefsLED.h"

#define LOG_VERBOSE 0
#define LOG_THREAD_PRIO                 ((EC_T_DWORD)29)   /* EtherCAT message logging thread priority (tAtEmLog) */
#define JOBS_THREAD_PRIO                ((EC_T_DWORD)98)   /* EtherCAT master job thread priority (tEcJobTask) */
#define JOBS_THREAD_STACKSIZE           0x4000
#define ETHERCAT_STATE_CHANGE_TIMEOUT   15000   /* master state change timeout in ms */
#define ETHERCAT_SCANBUS_TIMEOUT        10000   /* scanbus timeout in ms, see also EC_SB_DEFAULTTIMEOUT */

bool ecJobTaskStopEvent = false;
EC_T_CFG_SLAVE_INFO gCfgSlaveInfo;


static EC_T_VOID tEcJobTask(EC_T_VOID*)
{
    EC_T_DWORD dwRes{EC_E_ERROR};
    EC_T_USER_JOB_PARMS  oJobParms;
    OsMemset(&oJobParms, 0, sizeof(EC_T_USER_JOB_PARMS));

    do {
        dwRes = ecatExecJob(eUsrJob_ProcessAllRxFrames, &oJobParms);
        if (EC_E_NOERROR != dwRes && EC_E_INVALIDSTATE != dwRes && EC_E_LINK_DISCONNECTED != dwRes) {
            fprintf(stderr, "ERROR: ecatExecJob( eUsrJob_ProcessAllRxFrames): %s (0x%lx)\n", ecatGetText(dwRes), dwRes);
        } else {
            if (LOG_VERBOSE) {
                printf("1: SendAllCycFrames\n");
            }
        }

        // Demo code
        {
            EC_T_BYTE* abyPdIn = ecatGetProcessImageInputPtr();
            EC_T_BYTE *abyPdOut = ecatGetProcessImageOutputPtr();
            EC_T_STATE eMasterState = ecatGetMasterState();
            if ((eEcatState_SAFEOP == eMasterState) || (eEcatState_OP == eMasterState)) {
                // Add process data processing
            }

            if (eMasterState == eEcatState_OP) {
                static int counter = 0;
                counter++;
                LEDInput_t inputLED;
                LEDOutput_t outputLED;

                OsMemcpy(&inputLED, abyPdIn + gCfgSlaveInfo.dwPdOffsIn/8, sizeof(inputLED));
                outputLED.LED1 = !inputLED.BTN1;
                outputLED.LED2 = !inputLED.BTN2;

                auto value = counter%1000 > 500;
                outputLED.LED3 = value;
                outputLED.LED4 = value;
                outputLED.LED5 = value;
                outputLED.LED6 = value;
                outputLED.LED7 = value;
                outputLED.LED8 = value;
                OsMemcpy(abyPdOut + gCfgSlaveInfo.dwPdOffsOut/8, &outputLED, sizeof(outputLED));

//                // Print to terminal
//                if (counter % 100 == 0) {
//                    std::cout << "Buttons = " << (bool)inputLED.BTN1 << "." << (bool)inputLED.BTN2 << "\n";
//                }
            }
        }

        dwRes = ecatExecJob(eUsrJob_SendAllCycFrames, &oJobParms);
        if (EC_E_NOERROR != dwRes && EC_E_INVALIDSTATE != dwRes && EC_E_LINK_DISCONNECTED != dwRes) {
            fprintf(stderr, "ERROR: ecatExecJob(eUsrJob_SendAllCycFrames): %s (0x%lx)\n", ecatGetText(dwRes), dwRes);
        } else {
            if (LOG_VERBOSE) {
                printf("2: SendAllCycFrames\n");
            }
        }

        dwRes = ecatExecJob(eUsrJob_MasterTimer, EC_NULL);
        if (EC_E_NOERROR != dwRes && EC_E_INVALIDSTATE != dwRes) {
            fprintf(stderr, "ERROR: ecatExecJob(eUsrJob_MasterTimer): %s (0x%lx)\n", ecatGetText(dwRes), dwRes);
        } else {
            if (LOG_VERBOSE) {
                printf("3: MasterTimer\n");
            }
        }

        dwRes = ecatExecJob(eUsrJob_SendAcycFrames, EC_NULL);
        if (EC_E_NOERROR != dwRes && EC_E_INVALIDSTATE != dwRes && EC_E_LINK_DISCONNECTED != dwRes) {
            fprintf(stderr, "ERROR: ecatExecJob(eUsrJob_SendAcycFrames): %s (0x%lx)\n", ecatGetText(dwRes), dwRes);
        } else {
            if (LOG_VERBOSE) {
                printf("4: SendAsyncFrames\n");
            }
        }
        OsSleep(1); // sleep 1 ms
    } while (!ecJobTaskStopEvent);
}


static EC_T_DWORD ecatNotifyCallback(
    EC_T_DWORD         dwCode,  /**< [in]   Notification code */
    EC_T_NOTIFYPARMS*  pParms   /**< [in]   Notification parameters */
)
{
    EC_T_DWORD dwRetVal = EC_E_NOERROR;

    if ((EC_NULL == pParms) || (EC_NULL == pParms->pCallerData))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* notification for application ? */
    if ((dwCode >= EC_NOTIFY_APP) && (dwCode <= EC_NOTIFY_APP+EC_NOTIFY_APP_MAX_CODE))
    {
        /*****************************************************/
        /* Demo code: Remove/change this in your application */
        /* to get here the API ecatNotifyApp(dwCode, pParms) has to be called */
        /*****************************************************/
//        dwRetVal = myAppNotify(dwCode - EC_NOTIFY_APP, pParms);
        printf("AppNotify dwCode = %d\n", dwCode);
    }
    else
    {
        /* call the default handler */
//        dwRetVal = pDemoThreadParam->pNotInst->ecatNotify(dwCode, pParms);
        printf("DefaultNotify dwCode = %d\n", dwCode);
    }
    printf("ecatNotifyCallback: normal return\n");
    return dwRetVal;

Exit:
    printf("ecatNotifyCallback: Exit\n");
    return dwRetVal;
}


EC_T_DWORD EnableRealtimeEnvironment()
{
    printf("Enable Realtime Environment\n");
    return EC_E_NOERROR;
}

static EC_T_VOID LinkParmsInit(EC_T_LINK_PARMS* pLinkParms,
                               const EC_T_DWORD dwSignature, const EC_T_DWORD dwSize, const char* szDriverIdent,
                               const EC_T_DWORD dwInstance, const EC_T_LINKMODE eLinkMode, const EC_T_DWORD dwIstPriority = 0)
{
    OsMemset(pLinkParms, 0, sizeof(EC_T_LINK_PARMS));
    pLinkParms->dwSignature = dwSignature;
    pLinkParms->dwSize = dwSize;
    OsStrncpy(pLinkParms->szDriverIdent, szDriverIdent, MAX_DRIVER_IDENT_LEN - 1);
    pLinkParms->dwInstance = dwInstance;
    pLinkParms->eLinkMode = eLinkMode;
    pLinkParms->dwIstPriority = dwIstPriority;
}

EC_T_DWORD CreateLinkParmsSockRaw(EC_T_LINK_PARMS** pLinkParms, std::string name)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_SOCKRAW* pLinkParmsAdapter = EC_NULL;

    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_SOCKRAW*)OsMalloc(sizeof(EC_T_LINK_PARMS_SOCKRAW));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_SOCKRAW));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_SOCKRAW, sizeof(EC_T_LINK_PARMS_SOCKRAW), EC_LINK_PARMS_IDENT_SOCKRAW, 1, EcLinkMode_POLLING);

    /* set adapter name */
    std::copy(name.begin(),
              name.begin() + std::min(name.size(), sizeof(pLinkParmsAdapter->szAdapterName)),
              pLinkParmsAdapter->szAdapterName);

#if (defined DISABLE_FORCE_BROADCAST)
    /* Do not overwrite destination in frame with FF:FF:FF:FF:FF:FF, needed for EAP. */
    pLinkParmsAdapter->bDisableForceBroadcast = EC_TRUE;
#endif

    /* no errors */
    *pLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}


int main(int argc, char** argv)
{
    // parse options
    cmnCommandLineOptions options;
    std::string nicInstance;
    std::string eniFile;
    options.AddOptionOneValue("i", "instance", "NIC instance e.g. eth0",
                              cmnCommandLineOptions::REQUIRED_OPTION, &nicInstance);
    options.AddOptionOneValue("f", "file", "ENI file name",
                              cmnCommandLineOptions::REQUIRED_OPTION, &eniFile);
    std::string errorMessage;
    if (!options.Parse(argc, argv, errorMessage)) {
        std::cerr << "Error: " << errorMessage << std::endl;
        options.PrintUsage(std::cerr);
        return -1;
    } else {
        std::cout << "Args: NIC instance = " << nicInstance << "\n";
        std::cout << "Args: ENI file = " << eniFile << "\n";
    }

    EC_T_WORD wAutoIncAddress = 0;
    EC_T_BUS_SLAVE_INFO oBusSlaveInfo;

    int key = ' ';
    EC_T_CHAR szENIFilename[256] = {'\0'};
    sprintf(szENIFilename, "%s", eniFile.c_str());
    EC_T_CNF_TYPE eCnfType = eCnfType_Filename;
    EC_T_PBYTE pbyCnfData = (EC_T_PBYTE)&szENIFilename[0];
    EC_T_DWORD dwCnfDataLen = 256;

    EC_T_DWORD version;
    ecatGetVersion(&version);
    printf("Acontis Version: 0x%X\n", version);

    EC_T_CPUSET             CpuSet;
    EC_CPUSET_ZERO(CpuSet);
    EC_CPUSET_SET(CpuSet, 1);
    OsSetThreadAffinity(EC_NULL, CpuSet);
    EnableRealtimeEnvironment();

    EC_T_DWORD dwRet;
    EC_T_LINK_PARMS* pLinkParams = nullptr;
    dwRet = CreateLinkParmsSockRaw(&pLinkParams, nicInstance);
    printf("Returned driver: %s\n", pLinkParams->szDriverIdent);

    EC_T_INIT_MASTER_PARMS oInitParms;

    OsMemset(&oInitParms, 0, sizeof(EC_T_INIT_MASTER_PARMS));
    oInitParms.dwSignature                   = ATECAT_SIGNATURE;
    oInitParms.dwSize                        = sizeof(EC_T_INIT_MASTER_PARMS);
    oInitParms.pLinkParms                    = pLinkParams;
    oInitParms.pLinkParmsRed                 = EC_NULL;
    oInitParms.dwBusCycleTimeUsec            = 1000;
    oInitParms.dwMaxBusSlaves                = 256;
    oInitParms.dwMaxAcycFramesQueued         = 32;
    if (oInitParms.dwBusCycleTimeUsec >= 1000)
    {
        oInitParms.dwMaxAcycBytesPerCycle    = 4096;
    }
    else
    {
        oInitParms.dwMaxAcycBytesPerCycle    = 1500;
        oInitParms.dwMaxAcycFramesPerCycle   = 1;
        oInitParms.dwMaxAcycCmdsPerCycle     = 20;
    }
    oInitParms.dwEcatCmdMaxRetries           = 5;

    // ---------------------------------------------
    // IMPORTANT: set properly, or else SegFault
    // ---------------------------------------------
    CAtEmLogging            oLogging;
    EC_T_CHAR               szLogFileprefix[256] = {'\0'};
    G_pEcLogParms->dwLogLevel = EC_LOG_LEVEL_WARNING;
    oLogging.InitLogging(INSTANCE_MASTER_DEFAULT,
                         G_pEcLogParms->dwLogLevel,
                         LOG_ROLLOVER,
                         LOG_THREAD_PRIO,
                         2,
                         szLogFileprefix,
                         0x4000);
    oInitParms.LogParms.pfLogMsg = CAtEmLogging::LogMsg;
    oInitParms.LogParms.pLogContext = (struct _EC_T_LOG_CONTEXT*)&oLogging;
    oInitParms.LogParms.dwLogLevel = EC_LOG_LEVEL_VERBOSE;


    ETHERNET_ADDRESS srcMacAddress;
    static EC_T_PVOID           S_pvtJobThread    = EC_NULL;

    CEmNotification ceNotify(INSTANCE_MASTER_DEFAULT);
    EC_T_REGISTERRESULTS oRegisterResults;

    // init
    dwRet = ecatInitMaster(&oInitParms);
    if (dwRet != EC_E_NOERROR) {
        fprintf(stderr, "Cannot configure Link Layer, check pLinkParms, %s\n", ecatGetText(dwRet));
        goto error_handle;
    } else {
        printf("Configured Link Layer: %s\n", nicInstance.c_str());
    }

    dwRet = ecatGetSrcMacAddress(&srcMacAddress);
    if (dwRet != EC_E_NOERROR) {
        fprintf(stderr, "Cannot get MAC address: %s\n", ecatGetText(dwRet));
        goto error_handle;
    } else {
        printf("EtherCAT network adapter MAC: %02X-%02X-%02X-%02X-%02X-%02X\n",
               srcMacAddress.b[0],
               srcMacAddress.b[1],
               srcMacAddress.b[2],
               srcMacAddress.b[3],
               srcMacAddress.b[4],
               srcMacAddress.b[5]);
    }

    if (ecatConfigureMaster(eCnfType, pbyCnfData, dwCnfDataLen) != EC_E_NOERROR)
    {
        printf("ERROR: cannot configure EC-Master: %s\n", eniFile.c_str());
        goto error_handle;
    } else {
        printf("Configured network using: %s\n", eniFile.c_str());
    }

    // Register Client
    OsMemset(&oRegisterResults, 0, sizeof(EC_T_REGISTERRESULTS));
    dwRet = ecatRegisterClient(
        &CEmNotification::NotifyWrapper,
        &ceNotify,
        &oRegisterResults);
    if (dwRet != EC_E_NOERROR)
    {
        fprintf(stderr, "Cannot register client: %s (0x%lx))\n", ecatGetText(dwRet), dwRet);
        goto error_handle;
    } else {
        printf("Registered client, clientId = %d\n", oRegisterResults.dwClntId);
    }

    S_pvtJobThread = OsCreateThread((EC_T_CHAR *) "tEcJobTask",
                                    tEcJobTask,
                                    JOBS_THREAD_PRIO,
                                    JOBS_THREAD_STACKSIZE,
                                    (EC_T_VOID *)EC_NULL);

    // Scan Bus
    printf("===== Scanning slaves =====\n");
    dwRet = ecatScanBus(ETHERCAT_SCANBUS_TIMEOUT);
//    ceNotify.ProcessNotificationJobs();
    switch (dwRet) {
        case EC_E_NOERROR:
        case EC_E_BUSCONFIG_MISMATCH:
        case EC_E_LINE_CROSSED:
            printf("Printing slave infos\n");
            break;
        default:
            fprintf(stderr, "ERROR: Cannot scan bus: %s\n", ecatGetText(dwRet));
            break;
    }
    if (dwRet != EC_E_NOERROR) {
        goto error_handle;
    }

    dwRet = ecatGetBusSlaveInfo(EC_FALSE, wAutoIncAddress, &oBusSlaveInfo);
    if (dwRet != EC_E_NOERROR) {
        CMN_LOG_RUN_ERROR << "Failed to get bus slave info, wAutoIncAddress = " << wAutoIncAddress << "\n";
    }
    dwRet = ecatGetCfgSlaveInfo(EC_FALSE, wAutoIncAddress, &gCfgSlaveInfo);
    if (dwRet != EC_E_NOERROR) {
        CMN_LOG_RUN_ERROR << "Failed to get cfg slave info, wAutoIncAddress = " << wAutoIncAddress << "\n";
    }
    printf("Slave VendorID = 0x%X\n", oBusSlaveInfo.dwVendorId);
    printf("Slave InOffset Byte.Bit = %d.%d\n", gCfgSlaveInfo.dwPdOffsIn/8, gCfgSlaveInfo.dwPdOffsIn%8);
    printf("Slave OuOffset Byte.Bit = %d.%d\n", gCfgSlaveInfo.dwPdOffsOut/8, gCfgSlaveInfo.dwPdOffsOut%8);

    printf("===== End Scanning =====\n");

    // Set Master State
    dwRet = ecatSetMasterState(ETHERCAT_STATE_CHANGE_TIMEOUT, eEcatState_INIT);
    OsSleep(10);
    if (dwRet != EC_E_NOERROR) {
        printf("Cannot start set master state to INIT: %s\n", ecatGetText(dwRet));
    }
    else {
        printf("Set master state to INIT\n");
    }

    dwRet = ecatSetMasterState(ETHERCAT_STATE_CHANGE_TIMEOUT, eEcatState_PREOP);
    OsSleep(10);
    if (dwRet != EC_E_NOERROR) {
        CMN_LOG_INIT_ERROR << "Cannot start set master state to PREOP: " << ecatGetText(dwRet) << "\n";
    } else {
        CMN_LOG_INIT_WARNING << "Set master state to PREOP\n";
    }

    dwRet = ecatSetMasterState(ETHERCAT_STATE_CHANGE_TIMEOUT, eEcatState_SAFEOP);
    OsSleep(10);
    if (dwRet != EC_E_NOERROR) {
        CMN_LOG_INIT_ERROR << "Cannot start set master state to SAFEOP: " << ecatGetText(dwRet) << "\n";
    } else {
        CMN_LOG_INIT_WARNING << "Set master state to SAFEOP\n";
    }

    dwRet = ecatSetMasterState(ETHERCAT_STATE_CHANGE_TIMEOUT, eEcatState_OP);
    OsSleep(10);
    if (dwRet != EC_E_NOERROR) {
        CMN_LOG_INIT_ERROR << "Cannot start set master state to OP: " << ecatGetText(dwRet) << "\n";
    } else {
        CMN_LOG_INIT_WARNING << "Set master state to OP\n";
    }

    // loop until 'q' is pressed
    std::cout << "Press 'q' to quit" << std::endl;
    while (key != 'q') {
        key = cmnGetChar();
        OsSleep(20);
    }
    std::cout << "Quitting ..." << std::endl;

    printf("Stopping ECAT()\n");
    ecatStop(5000);
    ecJobTaskStopEvent = true;
    printf("DeinitMaster()\n");
    ecatDeinitMaster();
    return 0;

error_handle:
    printf("Error handling ... \n");
    ecatDeinitMaster();
    return -1;
}