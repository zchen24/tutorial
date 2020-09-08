/*-----------------------------------------------------------------------------
 * Logging.cpp
 * Copyright                acontis technologies GmbH, Weingarten, Germany
 * Response                 Stefan Zintgraf
 * Description              EtherCAT Master application logging
 *---------------------------------------------------------------------------*/

/*-INCLUDES------------------------------------------------------------------*/
#include "Logging.h"

#include <string.h>
#include <stdlib.h>

#ifdef VXWORKS
#include "vxWorks.h"
#include "sysLib.h"
#include "tickLib.h"
#if ((defined _WRS_VXWORKS_MAJOR) && (defined _WRS_VXWORKS_MINOR) && ( (_WRS_VXWORKS_MAJOR >= 7) || ((_WRS_VXWORKS_MAJOR == 6) && (_WRS_VXWORKS_MINOR >= 5)) ))
#else
#include <ifLib.h>
#endif
#endif /* VXWORKS */

/*-MACROS--------------------------------------------------------------------*/
/*#define NOPRINTF    1*/

/*-DEFINES-------------------------------------------------------------------*/
#if (defined UNDER_RTSS) || (defined __INTIME__)
    #define ABSOLUTE_LOG_FILE_PATH       "C:\\"
#elif (defined EC_VERSION_ECOS)
	#define FILESYS_8_3
#elif (defined EC_VERSION_RTXC)
    #define FILESYS_8_3
    #define ABSOLUTE_LOG_FILE_PATH       "b:\\"
#else
    #define ABSOLUTE_LOG_FILE_PATH       ""
#endif

#ifdef FILESYS_8_3
#define REM_ERRLOG_FILNAM       "rer"
#define LOC_ERRLOG_FILNAM       "er"
#define REM_MASTER_LOG_FILNAM   "rma"
#define LOC_MASTER_LOG_FILNAM   "ma"
#define REM_DCM_LOG_FILNAM      "rdc"
#define LOC_DCM_LOG_FILNAM      "dc"
#else
#define REM_ERRLOG_FILNAM       "rerror"
#define LOC_ERRLOG_FILNAM       (EC_T_CHAR*)"error"
#define REM_MASTER_LOG_FILNAM   "recmaster"
#define LOC_MASTER_LOG_FILNAM   (EC_T_CHAR*)"ecmaster"
#define REM_DCM_LOG_FILNAM      "rdcmlog"
#define LOC_DCM_LOG_FILNAM      (EC_T_CHAR*)"dcmlog"
#endif

#if (defined INCLUDE_ATEMRAS)
#define ERRLOG_FILNAM       REM_ERRLOG_FILNAM
#define MASTER_LOG_FILNAM   REM_MASTER_LOG_FILNAM
#define DCM_LOG_FILNAM      REM_DCM_LOG_FILNAM
#else
#define ERRLOG_FILNAM       LOC_ERRLOG_FILNAM
#define MASTER_LOG_FILNAM   LOC_MASTER_LOG_FILNAM
#define DCM_LOG_FILNAM      LOC_DCM_LOG_FILNAM
#endif

/* timestamp max. 10 digits, format "%010d%: " */
#define LOG_MSG_OFFSET_AFTER_TIMESTAMP 12

/*-GLOBAL VARIABLES-----------------------------------------------------------*/
#ifndef MAX_NUMOF_MASTER_INSTANCES
#define MAX_NUMOF_MASTER_INSTANCES 1
#endif
EC_T_LOG_PARMS G_aLogParms[MAX_NUMOF_MASTER_INSTANCES];

#ifdef INCLUDE_FRAME_SPY
CFrameLogMultiplexer* CFrameLogMultiplexer::G_aLogMultiplexer = EC_NULL;
#endif

#if (defined INCLUDE_FILE_LOGGING)
#if (defined EC_VERSION_FREERTOS) || (defined VXWORKS) || (defined __TKERNEL)\
    || (defined EC_VERSION_SYSBIOS) || (defined EC_VERSION_RIN32M3) || (defined EC_VERSION_XILINX_STANDALONE)\
    || (defined EC_VERSION_ETKERNEL) || (defined EC_VERSION_RTXC) || (defined EC_VERSION_RZT1) || (defined EC_VERSION_RZGNOOS) || (defined EC_VERSION_ECOS)\
    || (defined EC_VERSION_JSLWARE) || (defined EC_VERSION_UC3) || (defined EC_VERSION_UCOS) || (defined EC_VERSION_XMC)
EC_T_BOOL bLogFileEnb = EC_FALSE;
#else
EC_T_BOOL bLogFileEnb = EC_TRUE;
#endif
#endif /* INCLUDE_FILE_LOGGING */

EC_T_BOOL CAtEmLogging::s_bLogParmsArrayInitialized = EC_FALSE;

/*-CLASS FUNCTIONS-----------------------------------------------------------*/
/***************************************************************************************************/
/**
\brief  Create CAtEmLogging instance

\return -
*/
CAtEmLogging::CAtEmLogging(EC_T_VOID)
{
    m_dwMasterID = 0xffff;
    m_pvLogThreadObj    = EC_NULL;
    m_bLogTaskRunning = EC_FALSE;
    m_bShutdownLogTask = EC_FALSE;
    m_bSettling = EC_FALSE;
    m_dwNumMsgsSinceMsrmt = 0;
    m_dwLogLevel = EC_LOG_LEVEL_UNDEFINED;
    m_poInsertMsgLock = EC_NULL;
    m_poProcessMsgLock = EC_NULL;
    m_pchTempbuffer = EC_NULL;
    m_pFirstMsgBufferDesc = EC_NULL;
    m_pLastMsgBufferDesc = EC_NULL;
    m_pAllMsgBufferDesc = EC_NULL;
    m_pErrorMsgBufferDesc = EC_NULL;
    m_pDcmMsgBufferDesc = EC_NULL;
#if (defined INCLUDE_FILE_LOGGING)
    m_pchLogDir[0] = '\0';
    m_pchLogDir[MAX_PATH_LEN - 1] = '\0';
#endif

    if (!s_bLogParmsArrayInitialized)
    {
        OsMemset(G_aLogParms, 0, MAX_NUMOF_MASTER_INSTANCES * sizeof(EC_T_LOG_PARMS));
        s_bLogParmsArrayInitialized = EC_TRUE;
    }
}

/********************************************************************************/
/** \brief Initialize logging
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::InitLogging(
    EC_T_DWORD  dwMasterID,
    EC_T_DWORD  dwLogLevel,
    EC_T_WORD   wRollOver,
    EC_T_DWORD  dwPrio,
    EC_T_DWORD  dwCpuIndex,
    EC_T_CHAR*  szFilenamePrefix,
    EC_T_DWORD  dwStackSize,
    EC_T_DWORD  dwLogMsgBufferSize,
    EC_T_DWORD  dwErrMsgBufferSize,
    EC_T_DWORD  dwDcmMsgBufferSize
    )
{
    EC_T_CPUSET CpuSet;
#if (!defined XENOMAI)
    EC_T_BOOL   bOk;
#endif
    EC_T_CHAR   szLogFilename[256];

#if (!defined INCLUDE_FILE_LOGGING)
    EC_UNREFPARM(wRollOver);
#endif

    m_dwMasterID = dwMasterID;
    m_dwLogLevel = dwLogLevel;

    m_poInsertMsgLock = OsCreateLockTyped(eLockType_SPIN);
    m_poProcessMsgLock = OsCreateLock();
    m_pchTempbuffer = (EC_T_CHAR*)OsMalloc(2 * MAX_MESSAGE_SIZE);

    if (EC_NULL == m_pchTempbuffer)
    {
        LogMsg(pEcLogContext, EC_LOG_LEVEL_ERROR, "InitLogging: not enough memory for m_pchTempbuffer\n");
    }

    /* zero-terminate log msg (snprintf doesn't include this) */
    m_pchTempbuffer[0] = '\0';
    m_pchTempbuffer[2 * MAX_MESSAGE_SIZE - 1] = '\0';

    if ((EC_NULL != szFilenamePrefix) && ('\0' != szFilenamePrefix[0]))
    {
        OsSnprintf(szLogFilename, sizeof(szLogFilename) - 1, "%s_err", szFilenamePrefix);
    }
    else
    {
        OsSnprintf(szLogFilename, sizeof(szLogFilename) - 1, "%s", ERRLOG_FILNAM);
    }
    m_pErrorMsgBufferDesc = (MSG_BUFFER_DESC*)AddLogBuffer(
                                                    dwMasterID,
#if (defined INCLUDE_FILE_LOGGING)
                                                    wRollOver,
#endif
                                                    dwErrMsgBufferSize,
                                                    EC_TRUE,            /* skip duplicates */
                                                    (EC_T_CHAR*)"Err",  /* name of the logging (identification) */
#if (defined INCLUDE_FILE_LOGGING)
                                                    szLogFilename,
                                                    (EC_T_CHAR*)"log",  /* log file extension */
#endif
                                                    EC_FALSE,           /* print message on console? */
                                                    EC_TRUE );          /* logging with time stamp? */
    if (EC_NULL == m_pErrorMsgBufferDesc)
    {
        LogMsg(pEcLogContext, EC_LOG_LEVEL_ERROR, "InitLogging: not enough memory for m_pErrorMsgBufferDesc\n");
    }

    if ((EC_NULL != szFilenamePrefix) && ('\0' != szFilenamePrefix[0]))
    {
        OsSnprintf(szLogFilename, sizeof(szLogFilename) - 1, "%s", szFilenamePrefix);
    }
    else
    {
        OsSnprintf(szLogFilename, sizeof(szLogFilename) - 1, "%s", MASTER_LOG_FILNAM);
    }
    m_pAllMsgBufferDesc = (MSG_BUFFER_DESC*)AddLogBuffer(
                                                    dwMasterID,
#if (defined INCLUDE_FILE_LOGGING)
                                                    wRollOver,
#endif
                                                    dwLogMsgBufferSize,
                                                    EC_TRUE,            /* skip duplicates */
                                                    (EC_T_CHAR*)"Log",  /* name of the logging (identification) */
#if (defined INCLUDE_FILE_LOGGING)
                                                    szLogFilename,
                                                    (EC_T_CHAR*)"log",  /* log file extension */
#endif
                                                    EC_TRUE,            /* bPrintConsole */
                                                    EC_TRUE );          /* logging with time stamp? */
    if (EC_NULL == m_pAllMsgBufferDesc)
    {
        LogMsg(pEcLogContext, EC_LOG_LEVEL_ERROR, "InitLogging: not enough memory for m_pAllMsgBufferDesc\n");
    }

    if ((EC_NULL != szFilenamePrefix) && ('\0' != szFilenamePrefix[0]))
    {
        OsSnprintf(szLogFilename, sizeof(szLogFilename) - 1, "%s_dcm", szFilenamePrefix);
    }
    else
    {
        OsSnprintf(szLogFilename, sizeof(szLogFilename) - 1, "%s", DCM_LOG_FILNAM);
    }
    m_pDcmMsgBufferDesc = (MSG_BUFFER_DESC*)AddLogBuffer(
                                                    dwMasterID,
#if (defined INCLUDE_FILE_LOGGING)
                                                    wRollOver,
#endif
                                                    dwDcmMsgBufferSize,
                                                    EC_FALSE,           /* do not skip duplicates */
                                                    (EC_T_CHAR*)"DCM",  /* name of the logging (identification) */
#if (defined INCLUDE_FILE_LOGGING)
                                                    szLogFilename,
                                                    (EC_T_CHAR*)"csv",  /* log file extension */
#endif
                                                    EC_FALSE,           /* print message on console? */
                                                    EC_FALSE );         /* logging with time stamp? */
    if (EC_NULL == m_pDcmMsgBufferDesc)
    {
        LogMsg(pEcLogContext, EC_LOG_LEVEL_ERROR, "InitLogging: not enough memory for m_pDcmMsgBufferDesc\n");
    }

    OsDbgAssert(!m_bLogTaskRunning);
    m_bShutdownLogTask = EC_FALSE;
    EC_CPUSET_ZERO(CpuSet);
    EC_CPUSET_SET(CpuSet, dwCpuIndex);

#if (!defined NO_OS)
#if (defined INCLUDE_LOG_TASK)
    #if (defined __MET__)
        m_pvLogThreadObj = (EC_T_VOID*)_task_create((EC_T_WORD)0, (EC_T_DWORD)TECAT_LOGGING_INDEX, (EC_T_DWORD)this);
    #elif (defined XENOMAI)
        /* for Xenomai, pass the CPU affinity as upper 16bit of thread priority */
        m_pvLogThreadObj = OsCreateThread((EC_T_CHAR*)"tAtEmLog", tAtEmLogWrapper, (CpuSet << 16) | dwPrio, dwStackSize, this);
#else /* !__MET__ */
        m_pvLogThreadObj = OsCreateThread((EC_T_CHAR*)"tAtEmLog", tAtEmLogWrapper, dwPrio, dwStackSize, this);
#endif /* !__MET__ */
    while (!m_bLogTaskRunning) OsSleep(1);
#endif /* INCLUDE_LOG_TASK */
#endif /* !NO_OS */

#if (!defined XENOMAI)
    /* for Xenomai, the CPU affinity is set during task creating, see above */
    bOk = OsSetThreadAffinity(m_pvLogThreadObj, CpuSet);
    if (!bOk)
    {
        LogMsg(pEcLogContext, EC_LOG_LEVEL_ERROR, "Error: Set log task affinitiy, invalid CPU index %d\n", 0);
    }
#endif /* !XENOMAI */

#ifdef ECM_RUNTIME_LINKAGE
    if (m_dwMasterID < MAX_NUMOF_MASTER_INSTANCES)
    {
        G_aLogParms[m_dwMasterID].dwLogLevel = dwLogLevel;
        G_aLogParms[m_dwMasterID].pfLogMsg = CAtEmLogging::LogMsg;
        G_aLogParms[m_dwMasterID].pLogContext = (struct _EC_T_LOG_CONTEXT*)this;
    }
#endif
}

/********************************************************************************/
/** \brief Initialize logging
*
* \return message buffer descriptor
*/
struct _MSG_BUFFER_DESC* CAtEmLogging::AddLogBuffer(
    EC_T_DWORD  dwMasterID,
#if (defined INCLUDE_FILE_LOGGING)
    EC_T_WORD   wRollOver,
#endif
    EC_T_DWORD  dwBufferSize,       /* [in]  buffer size (number of buffered messages) */
    EC_T_BOOL   bSkipDuplicates,    /* [in]  EC_TRUE if duplicate messages shall be skipped */
    EC_T_CHAR*  szLogName,          /* [in]  name of the logging (identification) */
#if (defined INCLUDE_FILE_LOGGING)
    EC_T_CHAR*  szLogFilename,      /* [in]  log filename */
    EC_T_CHAR*  szLogFileExt,       /* [in]  log file extension */
#endif
    EC_T_BOOL   bPrintConsole,      /* [in]  print message on console? */
    EC_T_BOOL   bPrintTimestamp     /* [in]  logging with time stamp? */
    )
{
#if (defined INCLUDE_FILE_LOGGING)
    EC_T_CHAR   szLogFileNameTmp[MAX_PATH_LEN];
#endif
    MSG_BUFFER_DESC* pNewMsgBufferDesc = EC_NULL;
    EC_T_BOOL bLocked = EC_FALSE;
    EC_T_BOOL bOk = EC_FALSE;

    OsLock(m_poProcessMsgLock);
    bLocked = EC_TRUE;

    /* create buffers */
    pNewMsgBufferDesc = (MSG_BUFFER_DESC*)OsMalloc(sizeof(MSG_BUFFER_DESC));
    if (EC_NULL == pNewMsgBufferDesc)
    {
        LogMsg(pEcLogContext, EC_LOG_LEVEL_ERROR, "AddLogBuffer: cannot allocate message buffer descriptor\n");
        goto Exit;
    }
    OsMemset(pNewMsgBufferDesc,   0, sizeof(MSG_BUFFER_DESC));

#if (defined INCLUDE_FILE_LOGGING)
    if (m_pchLogDir[0] != '\0')
    {
        OsSnprintf(szLogFileNameTmp, sizeof(szLogFileNameTmp)-1, "%s%s%d", m_pchLogDir, szLogFilename, dwMasterID);
    }
    else
    {
        OsSnprintf(szLogFileNameTmp, sizeof(szLogFileNameTmp)-1, "%s%s%d", ABSOLUTE_LOG_FILE_PATH, szLogFilename, dwMasterID);
    }
#else
    EC_UNREFPARM(dwMasterID);
#endif
    bOk = InitMsgBuffer( pNewMsgBufferDesc,
                         MAX_MESSAGE_SIZE,
                         dwBufferSize,
                         bSkipDuplicates,
                         bPrintConsole,
                         bPrintTimestamp,
#if (defined INCLUDE_FILE_LOGGING)
                         szLogFileNameTmp,
                         szLogFileExt,
                         wRollOver,
#endif
                         szLogName
                        );
    if (!bOk)
    {
        goto Exit;
    }

    /* link buffer together */
    if (EC_NULL == m_pLastMsgBufferDesc)
    {
        /* create first buffer */
        OsDbgAssert(m_pFirstMsgBufferDesc == EC_NULL);
        m_pFirstMsgBufferDesc = m_pLastMsgBufferDesc = pNewMsgBufferDesc;
        pNewMsgBufferDesc->pNextMsgBuf = EC_NULL;
    }
    else
    {
        /* append to last buffer */
        OsDbgAssert(m_pLastMsgBufferDesc->pNextMsgBuf == EC_NULL);
        m_pLastMsgBufferDesc->pNextMsgBuf = pNewMsgBufferDesc;
        m_pLastMsgBufferDesc = pNewMsgBufferDesc;
    }
    bOk = EC_TRUE;

Exit:
    if (!bOk)
    {
        if (pNewMsgBufferDesc!=EC_NULL) OsFree(pNewMsgBufferDesc);
        pNewMsgBufferDesc = EC_NULL;
    }

    if (bLocked)
        OsUnlock(m_poProcessMsgLock);
    return pNewMsgBufferDesc;
}


/********************************************************************************/
/** \brief set log message buffer
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::SetMsgBuf(
    MSG_BUFFER_DESC* pMsgBufferDesc,
    EC_T_BYTE* pbyLogMem,
    EC_T_DWORD dwSize )
{
    pMsgBufferDesc->pbyNextLogMsg = pbyLogMem;
    pMsgBufferDesc->dwLogMemorySize = dwSize;
    pMsgBufferDesc->bLogBufferFull = EC_FALSE;
    pMsgBufferDesc->bNewLine = EC_FALSE;
    OsMemset(pbyLogMem, 0, dwSize);
    pMsgBufferDesc->pbyLogMemory = pbyLogMem;   /* initialize last as it will be active immediately after! */
}
EC_T_VOID CAtEmLogging::SetLogMsgBuf( EC_T_BYTE* pbyLogMem, EC_T_DWORD dwSize )
{
    SetMsgBuf( m_pAllMsgBufferDesc, pbyLogMem, dwSize );
}
EC_T_VOID CAtEmLogging::SetLogErrBuf( EC_T_BYTE* pbyLogMem, EC_T_DWORD dwSize )
{
    SetMsgBuf( m_pErrorMsgBufferDesc, pbyLogMem, dwSize );
}
EC_T_VOID CAtEmLogging::SetLogDcmBuf( EC_T_BYTE* pbyLogMem, EC_T_DWORD dwSize )
{
    SetMsgBuf( m_pDcmMsgBufferDesc, pbyLogMem, dwSize );
}

/********************************************************************************/
/** \brief De-initialize logging
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::DeinitLogging(EC_T_VOID)
{
MSG_BUFFER_DESC* pCurrMsgBuf;
MSG_BUFFER_DESC* pNextMsgBuf;

#ifdef ECM_RUNTIME_LINKAGE
    if (m_dwMasterID < MAX_NUMOF_MASTER_INSTANCES)
    {
        OsMemset(&G_aLogParms[m_dwMasterID], 0, sizeof(EC_T_LOG_PARMS));
    }
#endif

    m_bShutdownLogTask = EC_TRUE;

    while (m_bLogTaskRunning) OsSleep(1);

    /* shutdown all message buffers */
    pNextMsgBuf = m_pFirstMsgBufferDesc;
    while (EC_NULL != pNextMsgBuf)
    {
        DeinitMsgBuffer(pNextMsgBuf);
        pNextMsgBuf = pNextMsgBuf->pNextMsgBuf;
    }

#if (defined __MET__)
    _task_destroy( (_task_id) m_pvLogThreadObj);
#else
    OsDeleteThreadHandle(m_pvLogThreadObj);
#endif

    m_pvLogThreadObj = EC_NULL;

    /* free all message buffers */
    pNextMsgBuf = m_pFirstMsgBufferDesc;
    while (EC_NULL != pNextMsgBuf)
    {
        pCurrMsgBuf = pNextMsgBuf;
        pNextMsgBuf = pCurrMsgBuf->pNextMsgBuf;
        OsFree(pCurrMsgBuf);
    }
    /* unlink buffers */
    m_pFirstMsgBufferDesc = EC_NULL;
    m_pLastMsgBufferDesc = EC_NULL;
    m_pAllMsgBufferDesc = EC_NULL;
    m_pErrorMsgBufferDesc = EC_NULL;
    m_pDcmMsgBufferDesc = EC_NULL;

    OsDeleteLock(m_poInsertMsgLock);
    OsDeleteLock(m_poProcessMsgLock);
    SafeOsFree(m_pchTempbuffer);
    m_pchTempbuffer = EC_NULL;
}

/********************************************************************************/
/** \brief logging thread
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::tAtEmLogWrapper(EC_T_VOID* pvParm)
{
    CAtEmLogging *pInst = (CAtEmLogging*)pvParm;

    OsDbgAssert(EC_NULL != pInst);
    if (pInst)
    {
        pInst->tAtEmLog(EC_NULL);
    }
}

/********************************************************************************/
/** \brief cyclically process all messages
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::tAtEmLog(EC_T_VOID* pvParm)
{
    EC_UNREFPARM(pvParm);

    m_bLogTaskRunning = EC_TRUE;
    while (!m_bShutdownLogTask)
    {
        ProcessAllMsgs();
        OsSleep(1);
    }
    ProcessAllMsgs();
    m_bLogTaskRunning = EC_FALSE;
#if (defined EC_VERSION_RTEMS)
    rtems_task_delete(RTEMS_SELF);
#endif
}

EC_T_VOID CAtEmLogging::ProcessAllMsgs(EC_T_VOID)
{
    for (MSG_BUFFER_DESC* pNextMsgBuf = m_pFirstMsgBufferDesc; EC_NULL != pNextMsgBuf; pNextMsgBuf = pNextMsgBuf->pNextMsgBuf)
    {
        ProcessMsgs(pNextMsgBuf);
    }
}

/********************************************************************************/
/** \brief Initialize message buffer
*
* \return N/A
*/
EC_T_BOOL CAtEmLogging::InitMsgBuffer
(MSG_BUFFER_DESC*   pMsgBufferDesc      /* [in]  pointer to message buffer descriptor */
,EC_T_DWORD         dwMsgSize           /* [in]  size of a single message */
,EC_T_DWORD         dwNumMsgs           /* [in]  number of messages */
,EC_T_BOOL          bSkipDuplicates     /* [in]  EC_TRUE if duplicate messages shall be skipped */
,EC_T_BOOL          bPrintConsole       /* [in]  print message on console? */
,EC_T_BOOL          bPrintTimestamp     /* [in]  logging with time stamp? */
#if (defined INCLUDE_FILE_LOGGING)
,EC_T_CHAR*         szMsgLogFileName    /* [in]  message log file name */
,EC_T_CHAR*         szMsgLogFileExt     /* [in]  message log file name */
,EC_T_WORD          wRollOver           /* [in]  roll over counter */
#endif
,EC_T_CHAR*         szBufferName        /* [in]  name of the logging buffer */
)
{
    EC_T_BOOL  bOk = EC_FALSE;
    EC_T_CHAR* pchMsgBuffer = EC_NULL;
    EC_T_DWORD dwBufSiz;
    EC_T_DWORD dwCnt;
#if (defined INCLUDE_FILE_LOGGING)
    EC_T_CHAR  szfileNameTemp[MAX_PATH_LEN] = {0};
#endif

    pMsgBufferDesc->dwMsgSize = dwMsgSize;
    pMsgBufferDesc->dwNumMsgs = dwNumMsgs;
    pMsgBufferDesc->bPrintTimestamp = bPrintTimestamp;
    pMsgBufferDesc->bPrintConsole = bPrintConsole;
    pMsgBufferDesc->dwNextEmptyMsgIndex = 0;
    pMsgBufferDesc->dwNextPrintMsgIndex = 0;
    pMsgBufferDesc->wEntryCounter       = 0;
    pMsgBufferDesc->pbyLogMemory = EC_NULL;
    pMsgBufferDesc->dwLogMemorySize = 0;
    pMsgBufferDesc->pbyNextLogMsg = EC_NULL;
    pMsgBufferDesc->bLogBufferFull = EC_FALSE;
    pMsgBufferDesc->bSkipDuplicateMessages = bSkipDuplicates;
    pMsgBufferDesc->dwNumDuplicates = 0;
    pMsgBufferDesc->pszLastMsg = EC_NULL;

    pMsgBufferDesc->paMsg = (LOG_MSG_DESC*)OsMalloc(dwNumMsgs*sizeof(LOG_MSG_DESC));
    if (pMsgBufferDesc->paMsg == EC_NULL)
    {
        OsPrintf("CAtEmLogging::InitMsgBuffer: cannot get memory for logging buffer '%s'\n", szBufferName);
        goto Exit;
    }
    OsMemset(pMsgBufferDesc->paMsg, 0, dwNumMsgs*sizeof(LOG_MSG_DESC));

    dwBufSiz = dwNumMsgs * (dwMsgSize + 1);
    pchMsgBuffer = (EC_T_CHAR*)OsMalloc(dwBufSiz);
    if (pchMsgBuffer == EC_NULL)
    {
        OsPrintf("CAtEmLogging::InitMsgBuffer: cannot get memory for logging buffer '%s'\n", szBufferName);
        goto Exit;
    }

    /* Same as below. Needed to prevent false positive from static code analysis. */
    pMsgBufferDesc->paMsg[0].szMsgBuffer = pchMsgBuffer;

    OsMemset(pchMsgBuffer,0,dwBufSiz);
    for( dwCnt=0; dwCnt < dwNumMsgs; dwCnt++ )
    {
        pMsgBufferDesc->paMsg[dwCnt].szMsgBuffer = &pchMsgBuffer[dwCnt*(dwMsgSize+1)];
    }

#if (defined INCLUDE_FILE_LOGGING)
    pMsgBufferDesc->wLogFileIndex = 0;
    pMsgBufferDesc->wEntryCounterLimit = wRollOver;
    OsStrncpy(pMsgBufferDesc->szMsgLogFileName, szMsgLogFileName, sizeof(pMsgBufferDesc->szMsgLogFileName) - 1);
    OsStrncpy(pMsgBufferDesc->szMsgLogFileExt,  szMsgLogFileExt,  sizeof(pMsgBufferDesc->szMsgLogFileExt) - 1);

    if (0 != pMsgBufferDesc->wEntryCounterLimit )
    {
#ifdef FILESYS_8_3
        OsSnprintf(szfileNameTemp, sizeof(szfileNameTemp) - 1, "%s_%x.%s", pMsgBufferDesc->szMsgLogFileName, pMsgBufferDesc->wLogFileIndex, pMsgBufferDesc->szMsgLogFileExt);
#else
        OsSnprintf(szfileNameTemp, sizeof(szfileNameTemp) - 1, "%s.%x.%s", pMsgBufferDesc->szMsgLogFileName, pMsgBufferDesc->wLogFileIndex, pMsgBufferDesc->szMsgLogFileExt);
#endif
    }
    else
    {
        OsSnprintf(szfileNameTemp, sizeof(szfileNameTemp) - 1, "%s.%s", pMsgBufferDesc->szMsgLogFileName, pMsgBufferDesc->szMsgLogFileExt);
    }

    if (bLogFileEnb)
    {
        pMsgBufferDesc->pfMsgFile = OsFopen( szfileNameTemp, "w+");
        if (pMsgBufferDesc->pfMsgFile == EC_NULL)
        {
#if (!defined NOPRINTF)
            OsPrintf("ERROR: cannot create EtherCAT log file %s\n", szfileNameTemp);
#endif
            OsSleep(3000);
        }
    }
    else
    {
        pMsgBufferDesc->pfMsgFile = EC_NULL;
        pMsgBufferDesc->szMsgLogFileName[0] = 0;
    }
    OsStrncpy(pMsgBufferDesc->szLogName, szBufferName, MAX_PATH_LEN - 1);
#endif /* INCLUDE_FILE_LOGGING */

    pMsgBufferDesc->bIsInitialized = EC_TRUE;
    bOk = EC_TRUE;

Exit:
    if (!bOk)
    {
        SafeOsFree(pMsgBufferDesc->paMsg);
        SafeOsFree(pchMsgBuffer);
    }
    return bOk;
}

/********************************************************************************/
/** \brief De-Init message buffer
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::DeinitMsgBuffer
(MSG_BUFFER_DESC*   pMsgBufferDesc
)
{
CEcTimer oTimeout;

    if (pMsgBufferDesc->bIsInitialized)
    {
        /* let the log task print out all messages */
        if (pMsgBufferDesc->dwNextPrintMsgIndex != pMsgBufferDesc->dwNextEmptyMsgIndex)
        {
            OsPrintf("Store unsaved messages in '%s' message/logging buffer...", pMsgBufferDesc->szLogName);
            oTimeout.Start(3000);
            while (pMsgBufferDesc->dwNextPrintMsgIndex != pMsgBufferDesc->dwNextEmptyMsgIndex)
            {
                ProcessAllMsgs();
                OsSleep(100);
                if (oTimeout.IsElapsed())
                {
                    OsPrintf(".");
                    oTimeout.Start(3000);
                }
            }
            OsPrintf(" done!\n");
        }

        OsFree(pMsgBufferDesc->paMsg[0].szMsgBuffer);
        OsFree(pMsgBufferDesc->paMsg);

#if (defined INCLUDE_FILE_LOGGING)
        if (EC_NULL != pMsgBufferDesc->pfMsgFile)
        {
            OsFclose(pMsgBufferDesc->pfMsgFile);
        }
        pMsgBufferDesc->pfMsgFile = EC_NULL;
#endif

        pMsgBufferDesc->dwNextEmptyMsgIndex = 0;
        pMsgBufferDesc->dwNextPrintMsgIndex = 0;
        pMsgBufferDesc->bIsInitialized = EC_FALSE;
        pMsgBufferDesc->pbyLogMemory = EC_NULL;
        pMsgBufferDesc->dwLogMemorySize = 0;
        pMsgBufferDesc->pbyNextLogMsg = EC_NULL;
        pMsgBufferDesc->bLogBufferFull = EC_FALSE;
        pMsgBufferDesc->pszLastMsg = EC_NULL;
    }
}

/********************************************************************************/
/** \brief Process log message
*
* \return N/A
*/
EC_T_DWORD CAtEmLogging::LogMsg(struct _EC_T_LOG_CONTEXT* pContext, EC_T_DWORD dwLogMsgSeverity, const EC_T_CHAR* szFormat, ...)
{
	EC_T_DWORD  dwRetVal = EC_E_ERROR;
	EC_T_VALIST vaArgs;

    if (!s_bLogParmsArrayInitialized || (EC_NULL == pContext))
    {
        EC_VASTART(vaArgs, szFormat);
#if (!defined NOPRINTF)
        OsVprintf(szFormat, vaArgs);
#endif
        dwRetVal = EC_E_NOERROR;
        EC_VAEND(vaArgs);
    }
    else
    {
        /* discard if loglevel 0 */
        if (0 == ((CAtEmLogging*)pContext)->GetLogLevel()) return EC_E_NOERROR;

        if (dwLogMsgSeverity > ((CAtEmLogging*)pContext)->GetLogLevel()) return EC_E_NOERROR;

        // TODO: use m_pAllMsgBufferDesc to write normal and error log file and log to link layer
        if (dwLogMsgSeverity <= EC_LOG_LEVEL_ERROR)
        {
            EC_VASTART(vaArgs, szFormat);
            dwRetVal = ((CAtEmLogging*)pContext)->InsertNewMsgVa(((CAtEmLogging*)pContext)->m_pErrorMsgBufferDesc, dwLogMsgSeverity, szFormat, vaArgs);
            EC_VAEND(vaArgs);
        }
        EC_VASTART(vaArgs, szFormat);
        dwRetVal = ((CAtEmLogging*)pContext)->InsertNewMsgVa(((CAtEmLogging*)pContext)->m_pAllMsgBufferDesc, dwLogMsgSeverity, szFormat, vaArgs);
        EC_VAEND(vaArgs);
    }
    return dwRetVal;
}

EC_T_DWORD CAtEmLogging::LogMsgStub(struct _EC_T_LOG_CONTEXT* pContext, EC_T_DWORD dwLogMsgSeverity, const EC_T_CHAR* szFormat, ...)
{
    EC_UNREFPARM(pContext); EC_UNREFPARM(dwLogMsgSeverity); EC_UNREFPARM(szFormat);
    return EC_E_NOERROR;
}

EC_T_DWORD CAtEmLogging::LogMsgOsPrintf(struct _EC_T_LOG_CONTEXT* pContext, EC_T_DWORD dwLogMsgSeverity, const EC_T_CHAR* szFormat, ...)
{
    EC_T_DWORD  dwRetVal = EC_E_ERROR;
    EC_T_VALIST vaArgs;

    EC_UNREFPARM(pContext); EC_UNREFPARM(dwLogMsgSeverity);

    EC_VASTART(vaArgs, szFormat);
    OsVprintf(szFormat, vaArgs);
    EC_VAEND(vaArgs);

    dwRetVal = EC_E_NOERROR;
    return dwRetVal;
}

/********************************************************************************/
/** \brief Insert a new message into message buffer with Log Message Severity
*
* \return N/A
*/
EC_T_DWORD CAtEmLogging::InsertNewMsgVa
(MSG_BUFFER_DESC*   pMsgBufferDesc
 ,EC_T_DWORD        dwLogMsgSeverity
,const
 EC_T_CHAR*         szFormat
,EC_T_VALIST        vaArgs
)
{
    EC_T_DWORD    dwRes       = EC_E_NOERROR;
    EC_T_BOOL     bBufferFull = EC_FALSE;
    LOG_MSG_DESC* pNewMsg     = EC_NULL;
    EC_T_CHAR*    szMsgBuffer = EC_NULL; /* without timestamp */
    EC_T_DWORD    dwMsgSize   = 0; /* decreased by timestamp */
    EC_T_DWORD    dwLen       = 0; /* increased by timestamp */

    /* select default message queue if needed */
    if (pMsgBufferDesc == EC_NULL)
    {
        pMsgBufferDesc = m_pAllMsgBufferDesc;
    }
    /* check state */
    if ((EC_NULL == pMsgBufferDesc) || !pMsgBufferDesc->bIsInitialized || m_bShutdownLogTask)
    {
        dwRes = EC_E_INVALIDSTATE;
        goto Exit;
    }
    /* get message descriptor */
    OsLock(m_poInsertMsgLock);
    {
        EC_T_DWORD dwNewNextEmpty = 0;

        pNewMsg = &pMsgBufferDesc->paMsg[pMsgBufferDesc->dwNextEmptyMsgIndex];
        pNewMsg->dwLen = 0;

        dwNewNextEmpty = pMsgBufferDesc->dwNextEmptyMsgIndex + 1;
        if (dwNewNextEmpty >= pMsgBufferDesc->dwNumMsgs )
        {
            dwNewNextEmpty = 0;
        }
        if (dwNewNextEmpty == pMsgBufferDesc->dwNextPrintMsgIndex )
        {
            bBufferFull = EC_TRUE;
        }
        else
        {
            pMsgBufferDesc->dwNextEmptyMsgIndex = dwNewNextEmpty;
        }
    }
    OsUnlock(m_poInsertMsgLock);

    if (bBufferFull)
    {
        dwRes = EC_E_NOMEMORY;
        goto Exit;
    }
    /* fill message descriptor */
    pNewMsg->dwSeverity      = dwLogMsgSeverity;

    /* timestamp max. 10 digits, format "%010d%: " */
    if (pMsgBufferDesc->bPrintTimestamp)
    {
        szMsgBuffer = &pNewMsg->szMsgBuffer[LOG_MSG_OFFSET_AFTER_TIMESTAMP];
        dwMsgSize = pMsgBufferDesc->dwMsgSize - LOG_MSG_OFFSET_AFTER_TIMESTAMP;
        pNewMsg->dwMsgTimestamp = OsQueryMsecCount();
        dwLen = LOG_MSG_OFFSET_AFTER_TIMESTAMP; /* timestamp serialized in CAtEmLogging::ProcessMsgs(...) */
    }
    else
    {
        szMsgBuffer = pNewMsg->szMsgBuffer;
        dwMsgSize = pMsgBufferDesc->dwMsgSize;
        pNewMsg->dwMsgTimestamp = 0;
        dwLen = 0;
    }

    /* format message */
    pNewMsg->dwLen = dwLen + EcVsnprintf(szMsgBuffer, dwMsgSize, szFormat, vaArgs);

    OnLogMsg(szMsgBuffer);

    if (FilterMsg(szMsgBuffer))
    {
        szMsgBuffer[0] = '\0';
        pNewMsg->szMsgBuffer[0] = '\0';
        pNewMsg->dwLen = 0;
    }

    OsMemoryBarrier();
    pNewMsg->bValid = EC_TRUE;

Exit:
    return dwRes;
}

/********************************************************************************/
/** \brief Forward to next logging buffer (memory logging)
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::SelectNextLogMemBuffer
(MSG_BUFFER_DESC*   pMsgBufferDesc )
{
    pMsgBufferDesc->pbyNextLogMsg = pMsgBufferDesc->pbyNextLogMsg + OsStrlen(pMsgBufferDesc->pbyNextLogMsg);
    if (pMsgBufferDesc->pbyNextLogMsg >= (pMsgBufferDesc->pbyLogMemory + pMsgBufferDesc->dwLogMemorySize - 3*MAX_MESSAGE_SIZE) )
    {
        /* stop logging if memory is full */
        LogMsg(pEcLogContext, EC_LOG_LEVEL_INFO, "logging buffer %s is full, logging stopped!\n", pMsgBufferDesc->szLogName);
        pMsgBufferDesc->bLogBufferFull = EC_TRUE;
    }

    /* zero-terminate log msg (snprintf doesn't include this) */
    pMsgBufferDesc->pbyNextLogMsg[0] = '\0';
    pMsgBufferDesc->pbyNextLogMsg[MAX_MESSAGE_SIZE - 1] = '\0';
}

EC_T_VOID CAtEmLogging::PrintConsole(EC_T_CHAR* szMsgBuffer)
{
    /* printf */
#if (!defined NOPRINTF)
    OsPrintf("%s", szMsgBuffer);
#endif
}

/********************************************************************************/
/** \brief Process all messages of a message buffer
*
* \return N/A
*/
EC_T_VOID CAtEmLogging::ProcessMsgs
(MSG_BUFFER_DESC*   pMsgBufferDesc )
{
    EC_T_DWORD    dwNewNextPrint              = 0;
    LOG_MSG_DESC* pCurrMsg                    = EC_NULL;
    EC_T_BOOL     bLocked                     = EC_FALSE;
#if (defined INCLUDE_FILE_LOGGING)
    FILE*         pFileHandle                 = EC_NULL;
    EC_T_BOOL     bRollOver                   = EC_FALSE;
    EC_T_CHAR     szfileNameTemp[MAX_PATH_LEN];
#endif
    EC_T_DWORD    dwNumMsgLeft                = 20;
    EC_T_BOOL     bSkipDuplicate              = EC_FALSE;
    EC_T_DWORD    dwNumDuplicatesBeforeNewMsg = 0;

#if (defined INCLUDE_FILE_LOGGING)
    szfileNameTemp[0] = '\0';
#endif

    if (pMsgBufferDesc->bIsInitialized)
    {
        OsLock(m_poProcessMsgLock);
        bLocked = EC_TRUE;
        while (pMsgBufferDesc->dwNextPrintMsgIndex != pMsgBufferDesc->dwNextEmptyMsgIndex)
        {
            OsDbgAssert(pMsgBufferDesc->bIsInitialized);

            /* return after maximum number of processed messages */
            if (dwNumMsgLeft == 0)
            {
                break;
            }

            pCurrMsg = &pMsgBufferDesc->paMsg[pMsgBufferDesc->dwNextPrintMsgIndex];
            /* wait until message complete */
            if (!pCurrMsg->bValid)
            {
                break;
            }

            /* message complete, not filtered */
            if (pCurrMsg->dwLen != 0)
            {
                /* finalize szMsgBuffer (timestamp, new line)
                   szMsgBuffer contains timestamp, message and new line, because e.g. RTX64 requires terminating new - line directly appended to the message */
                if (pMsgBufferDesc->bPrintTimestamp)
                {
                    EC_T_INT nLen = EcSnprintf(pCurrMsg->szMsgBuffer, LOG_MSG_OFFSET_AFTER_TIMESTAMP - 1, "%010d", (EC_T_INT)pCurrMsg->dwMsgTimestamp);
                    pCurrMsg->szMsgBuffer[nLen] = ':';
                    pCurrMsg->szMsgBuffer[nLen + 1] = ' ';
                }
                if (pCurrMsg->szMsgBuffer[pCurrMsg->dwLen - 1] != '\n')
                {
                    pCurrMsg->dwLen = EC_AT_MOST(MAX_MESSAGE_SIZE - 2, pCurrMsg->dwLen + 1);
                    pCurrMsg->szMsgBuffer[pCurrMsg->dwLen - 1] = '\n';
                    pCurrMsg->szMsgBuffer[pCurrMsg->dwLen] = '\0';
                }

                /* skip duplicate */
                bSkipDuplicate = EC_FALSE;
                dwNumDuplicatesBeforeNewMsg = 0;
                if (pMsgBufferDesc->bSkipDuplicateMessages)
                {
                    /* don't check timestamp for duplicate */
                    EC_T_CHAR* szMsgBufferDuplicateCheck = pCurrMsg->szMsgBuffer + ((pMsgBufferDesc->bPrintTimestamp) ? LOG_MSG_OFFSET_AFTER_TIMESTAMP : 0);
                    EC_T_DWORD dwLenDuplicateCheck = pCurrMsg->dwLen - ((pMsgBufferDesc->bPrintTimestamp) ? LOG_MSG_OFFSET_AFTER_TIMESTAMP : 0);

                    if (pMsgBufferDesc->pszLastMsg == EC_NULL)
                    {
                        /* first message */
                        pMsgBufferDesc->pszLastMsg = szMsgBufferDuplicateCheck;
                    }
                    else if (OsStrncmp(pMsgBufferDesc->pszLastMsg, szMsgBufferDuplicateCheck, dwLenDuplicateCheck) == 0)
                    {
                        /* same message as before, just increment duplicate pointer */
                        pMsgBufferDesc->dwNumDuplicates++;
                        bSkipDuplicate = EC_TRUE;
                    }
                    else
                    {
                        /* new message */
                        dwNumDuplicatesBeforeNewMsg = pMsgBufferDesc->dwNumDuplicates;
                        pMsgBufferDesc->dwNumDuplicates = 0;
                        pMsgBufferDesc->pszLastMsg = szMsgBufferDuplicateCheck;
                        if (dwNumDuplicatesBeforeNewMsg > 0 && pMsgBufferDesc->pszLastMsg[0] == '\0')
                        {
                            /* ignore empty duplicates... */
                            dwNumDuplicatesBeforeNewMsg = 0;
                        }
                    }
                }
                if (!bSkipDuplicate) dwNumMsgLeft--;

#if (defined INCLUDE_FILE_LOGGING)
                if (bLogFileEnb && (pMsgBufferDesc->pfMsgFile != EC_NULL))
                {
                    pFileHandle = pMsgBufferDesc->pfMsgFile;

                    if (!bSkipDuplicate) pMsgBufferDesc->wEntryCounter++;
                    if (0 != pMsgBufferDesc->wEntryCounterLimit && pMsgBufferDesc->pfMsgFile != EC_NULL)
                    {
                        if (pMsgBufferDesc->wEntryCounter >= pMsgBufferDesc->wEntryCounterLimit)
                        {
                            bRollOver = EC_TRUE;
                            pMsgBufferDesc->wEntryCounter = 0;
                            pMsgBufferDesc->wLogFileIndex++;

#ifdef FILESYS_8_3
                            OsSnprintf(szfileNameTemp, sizeof(szfileNameTemp) - 1, "%s_%x.%s", pMsgBufferDesc->szMsgLogFileName, pMsgBufferDesc->wLogFileIndex, pMsgBufferDesc->szMsgLogFileExt);
#else
                            OsSnprintf(szfileNameTemp, sizeof(szfileNameTemp) - 1, "%s.%x.%s", pMsgBufferDesc->szMsgLogFileName, pMsgBufferDesc->wLogFileIndex, pMsgBufferDesc->szMsgLogFileExt);
#endif
                        }
                        else
                        {
                            bRollOver = EC_FALSE;
                        }

                    }
                }
#endif /* INCLUDE_FILE_LOGGING */

                if (pMsgBufferDesc->bPrintConsole && !bSkipDuplicate)
                {
                    /* print skip messages */
                    if (dwNumDuplicatesBeforeNewMsg > 0)
                    {
                        EC_T_CHAR szIdenticalMessagesSkipped[64];
                        szIdenticalMessagesSkipped[63] = '\0';
                        EcSnprintf(szIdenticalMessagesSkipped, 63, "%d identical messages skipped\n", dwNumDuplicatesBeforeNewMsg);
                        PrintConsole(szIdenticalMessagesSkipped);
                    }

                    /* print timestamp, message */
                    PrintConsole(pCurrMsg->szMsgBuffer);
                }

#if (defined INCLUDE_FILE_LOGGING)
                if (pMsgBufferDesc->pbyLogMemory != EC_NULL)
                {
                    if (!pMsgBufferDesc->bLogBufferFull && !bSkipDuplicate)
                    {
                        EC_T_DWORD dwWritten = 0;

                        /* print skip messages */
                        if (dwNumDuplicatesBeforeNewMsg > 0)
                        {
                            if (pMsgBufferDesc->bPrintTimestamp)
                            {
                                dwWritten = dwWritten + OsSnprintf((EC_T_CHAR*)(pMsgBufferDesc->pbyNextLogMsg + dwWritten), MAX_MESSAGE_SIZE - dwWritten - 1, "%06d : ", (EC_T_INT)pCurrMsg->dwMsgTimestamp);
                            }
                            dwWritten = dwWritten + OsSnprintf((EC_T_CHAR*)(pMsgBufferDesc->pbyNextLogMsg + dwWritten), MAX_MESSAGE_SIZE - dwWritten - 1, "%d identical messages skipped\n", dwNumDuplicatesBeforeNewMsg);
                            SelectNextLogMemBuffer(pMsgBufferDesc);
                        }
                        if (!pMsgBufferDesc->bLogBufferFull)
                        {
                            dwWritten = 0;

                            /* memory logging */
                            dwWritten = dwWritten + OsSnprintf((EC_T_CHAR*)(pMsgBufferDesc->pbyNextLogMsg + dwWritten), MAX_MESSAGE_SIZE - dwWritten - 1, "%s", pCurrMsg->szMsgBuffer);

                            SelectNextLogMemBuffer(pMsgBufferDesc);
                        }
                    }
                }
                else if (EC_NULL != pFileHandle)
                {
                    /* don't use fprintf, some platforms don't support it! */

                    if (!bSkipDuplicate)
                    {
                        /* print skipped messages */
                        if (dwNumDuplicatesBeforeNewMsg > 0)
                        {
                            if (pMsgBufferDesc->bPrintTimestamp)
                            {
                                OsSnprintf(m_pchTempbuffer, 2 * MAX_MESSAGE_SIZE - 1, "%06d : ", (EC_T_INT)pCurrMsg->dwMsgTimestamp);
                                OsFwrite(m_pchTempbuffer, OsStrlen(m_pchTempbuffer), 1, pFileHandle);
                            }
                            OsSnprintf(m_pchTempbuffer, 2 * MAX_MESSAGE_SIZE - 1, "%d identical messages skipped\n", dwNumDuplicatesBeforeNewMsg);
                            OsFwrite(m_pchTempbuffer, OsStrlen(m_pchTempbuffer), 1, pFileHandle);
                        }

                        /* print message, including new line */
                        OsSnprintf(m_pchTempbuffer, 2 * MAX_MESSAGE_SIZE - 1, "%s", pCurrMsg->szMsgBuffer);
                        OsFwrite(m_pchTempbuffer, OsStrlen(m_pchTempbuffer), 1, pFileHandle);
                        OsFflush(pFileHandle);
                    }
                }
#endif
            }

            pCurrMsg->bValid = EC_FALSE;
            OsMemoryBarrier();
            dwNewNextPrint = pMsgBufferDesc->dwNextPrintMsgIndex + 1;
            if (dwNewNextPrint >= pMsgBufferDesc->dwNumMsgs )
            {
                dwNewNextPrint = 0;
            }
            pMsgBufferDesc->dwNextPrintMsgIndex = dwNewNextPrint;

#if (defined INCLUDE_FILE_LOGGING)
            if (bRollOver)
            {
                /* do roll over */
                OsFclose(pFileHandle);

                pFileHandle = OsFopen(szfileNameTemp, "w+");
                if (pFileHandle == EC_NULL)
                {
#if (!defined NOPRINTF)
                    OsPrintf("ERROR: cannot create EtherCAT log file %s\n", szfileNameTemp);
#endif
                    OsSleep(3000);
                }

                pMsgBufferDesc->pfMsgFile = pFileHandle;

                bRollOver = EC_FALSE;
            }
#endif /* INCLUDE_FILE_LOGGING */
        }
    }
    if (bLocked)
        OsUnlock(m_poProcessMsgLock);

    return;
}

/********************************************************************************/
/** \brief application DCM message function
*
* \return N/A
*/
EC_T_DWORD CAtEmLogging::LogDcm(const EC_T_CHAR* szFormat, ...)
{
EC_T_VALIST vaArgs;
EC_T_DWORD  dwRes = EC_E_NOERROR;

    EC_VASTART(vaArgs, szFormat);
    dwRes = InsertNewMsgVa(m_pDcmMsgBufferDesc, EC_LOG_LEVEL_INFO, szFormat, vaArgs);
    EC_VAEND(vaArgs);
    return dwRes;
}

/********************************************************************************/
/** \brief set log thread affinity
*
* \return N/A
*/
EC_T_BOOL CAtEmLogging::SetLogThreadAffinity(EC_T_DWORD dwCpuIndex)
{
EC_T_CPUSET CpuSet;
EC_T_BOOL bOk = EC_TRUE;
EC_UNREFPARM(dwCpuIndex);

    if (EC_NULL != m_pvLogThreadObj)
    {
        EC_CPUSET_ZERO(CpuSet);
        EC_CPUSET_SET(CpuSet, dwCpuIndex);
        bOk = OsSetThreadAffinity(m_pvLogThreadObj, CpuSet);
    }
    return bOk;
}

#if (defined INCLUDE_FILE_LOGGING)
/********************************************************************************/
/** \brief set log directory
*
* \return EC_E_NOERROR or EC_E_NOMEMORY if szLogDir too long
*/
EC_T_DWORD CAtEmLogging::SetLogDir(EC_T_CHAR* szLogDir)
{
    if (OsStrlen(szLogDir) >= MAX_PATH_LEN)
    {
        return EC_E_NOMEMORY;
    }
    OsStrncpy(m_pchLogDir, szLogDir, OsStrlen(szLogDir) + 1);
    return EC_E_NOERROR;
}
#endif /* INCLUDE_FILE_LOGGING */

#ifdef INCLUDE_FRAME_SPY
CFrameLogMultiplexer::CFrameLogMultiplexer()
{
    m_pvLoggerListLock = OsCreateLock();
}

CFrameLogMultiplexer::~CFrameLogMultiplexer()
{
    if (EC_NULL != m_pvLoggerListLock)
    {
        OsLock(m_pvLoggerListLock);
        for (CLoggerDescList::CNode* pNode = m_LoggerList.GetFirstNode(); EC_NULL != pNode; m_LoggerList.GetNext(pNode))
        {
            m_LoggerList.RemoveAt(pNode);
        }
        emLogFrameDisable(m_dwMasterId);
        OsUnlock(m_pvLoggerListLock);

        OsDeleteLock(m_pvLoggerListLock);
    }
}

/********************************************************************************/
/** \brief add logger to list
*
* \return N/A
*/
EC_T_VOID CFrameLogMultiplexer::AddFrameLogger(EC_T_DWORD dwMasterId, EC_T_VOID* pvContext, EC_T_PFLOGFRAME_CB pvLogFrameCallBack)
{
    if (EC_NULL == G_aLogMultiplexer)
    {
        G_aLogMultiplexer = EC_NEW(CFrameLogMultiplexer[MAX_NUMOF_MASTER_INSTANCES]);
        for (EC_T_DWORD i = 0; i < MAX_NUMOF_MASTER_INSTANCES; i++)
        {
            G_aLogMultiplexer[i].m_dwMasterId = i;
        }
    }
    G_aLogMultiplexer[dwMasterId].AddFrameLogger(pvContext, pvLogFrameCallBack);
}

/********************************************************************************/
/** \brief search logger in list and remove
*
* \return N/A
*/
EC_T_VOID CFrameLogMultiplexer::RemoveFrameLogger(EC_T_DWORD dwMasterId, EC_T_VOID* pvContext, EC_T_PFLOGFRAME_CB pvLogFrameCallBack)
{
    if (EC_NULL != G_aLogMultiplexer)
    {
        G_aLogMultiplexer[dwMasterId].RemoveFrameLogger(pvContext, pvLogFrameCallBack);

        /* delete array if all multiplexers are unused */
        for (EC_T_DWORD i = 0; i < MAX_NUMOF_MASTER_INSTANCES; i++)
        {
            if (!G_aLogMultiplexer[i].m_LoggerList.IsEmpty())
            {
                return;
            }
        }
        SafeDeleteArray(G_aLogMultiplexer);
    }
}

CFrameSpy::~CFrameSpy()
{
    Uninstall();
}

EC_T_VOID CFrameSpy::Install(EC_T_DWORD dwMasterId)
{
    if (dwMasterId > MAX_NUMOF_MASTER_INSTANCES) return;

    if (m_dwMasterId != dwMasterId)
    {
        CFrameLogMultiplexer::AddFrameLogger(dwMasterId, (EC_T_VOID*)this, StaticFrameHandler);
        m_dwMasterId = dwMasterId;
    }
}

EC_T_VOID CFrameSpy::Uninstall(EC_T_VOID)
{
    if (0xffff != m_dwMasterId)
    {
        CFrameLogMultiplexer::RemoveFrameLogger(m_dwMasterId, (EC_T_VOID*)this, StaticFrameHandler);
        m_dwMasterId = 0xffff;
    }
}
void CFrameSpy::FrameHandler(EC_T_DWORD dwLogFrameFlags, EC_T_DWORD dwFrameSize, EC_T_BYTE* pbyFrame)
{
    EC_T_WORD wCmdLen;
    EC_T_WORD wEtherCAT_HeaderOffset = ETHERNET_FRAME_LEN;
    EC_T_BYTE* pbyData = pbyFrame;

    /* skip non-EtherCAT frames */
    if (ETHERNET_FRAME_TYPE_BKHF != EC_ETHFRM_GET_FRAMETYPE(pbyFrame)) return;

    /* skip EtherCAT frames without datagrams */
    if (dwFrameSize < wEtherCAT_HeaderOffset + ETYPE_EC_OVERHEAD) return;

    /* parse all commands */
    for (pbyData += wEtherCAT_HeaderOffset + 2; EC_NULL != pbyData;
    pbyData = (0 != EC_AL_CMDHDRLEN_GET_NEXT(&((ETYPE_EC_CMD_HEADER*)pbyData)->uLen)) ? pbyData + wCmdLen : EC_NULL)
    {
        ETYPE_EC_CMD_HEADER* pCmdHdr = (ETYPE_EC_CMD_HEADER*)pbyData;
        wCmdLen = (EC_T_WORD)(EC_CMDHDRLEN_GET_LEN(&pCmdHdr->uLen) + ETYPE_EC_OVERHEAD);

        /* call datagram handler */
        DatagramHandler(dwLogFrameFlags, dwFrameSize, pbyFrame, pCmdHdr, wCmdLen);
    }
}
#endif

#if (defined INCLUDE_PCAP_RECORDER)
/*-LOGGING-------------------------------------------------------------------*/
#undef  pLogMsgCallback
#undef  pEcLogContext
#undef  dwEcLogLevel
#define dwEcLogLevel      (GetLogParms()->dwLogLevel)
#define pLogMsgCallback   (GetLogParms()->pfLogMsg)
#define pEcLogContext     (GetLogParms()->pLogContext)


/*-CLASS FUNCTIONS-----------------------------------------------------------*/
CPcapRecorder::CPcapRecorder(EC_T_DWORD dwBufCnt, EC_T_DWORD dwPrio /* LOG_THREAD_PRIO */, EC_T_DWORD dwMasterId, const EC_T_CHAR* szFileName)
: CPcapFileBufferedWriter(dwBufCnt, dwPrio), m_dwMasterId(dwMasterId), m_szFileName(szFileName)
{
    m_nLoggedFrameCount = 0;
    OsDbgAssert(ETHERNET_MAX_FRAME_LEN < ETHERNET_MAX_FRAMEBUF_LEN - sizeof(EC_T_LINK_FRAME_BUF_ENTRY_FRAME_DESC));
    if (EC_NULL != m_szFileName)
    {
        Open(m_szFileName);
    }
    if (0xffff != m_dwMasterId)
    {
        Install(m_dwMasterId);
    }
}

CPcapRecorder::~CPcapRecorder()
{
    if (0xffff != m_dwMasterId)
    {
        Uninstall();
    }
    Stop(15000);
    FlushBuffer();
    if (EC_NULL != m_pfHandle)
    {
        Close();
    }
}

EC_T_VOID CPcapRecorder::Install(EC_T_DWORD dwMasterId)
{
    if (dwMasterId > MAX_NUMOF_MASTER_INSTANCES) return;

    if (m_dwMasterId != dwMasterId)
    {
        CFrameLogMultiplexer::AddFrameLogger(dwMasterId, (EC_T_VOID*)this, LogFrameStatic);
        m_dwMasterId = dwMasterId;
    }
}

EC_T_VOID CPcapRecorder::Uninstall()
{
    if (0xffff != m_dwMasterId)
    {
        CFrameLogMultiplexer::RemoveFrameLogger(m_dwMasterId, (EC_T_VOID*)this, LogFrameStatic);
        m_dwMasterId = 0xffff;
    }
}

EC_T_VOID CPcapFileBufferedWriter::FlushBuffer(EC_T_VOID)
{
    EC_T_LINK_FRAME_BUF_ENTRY Entry;
    OsMemset(&Entry, 0, sizeof(EC_T_LINK_FRAME_BUF_ENTRY));
    while (m_FrameBuffer.RemoveNoLock(Entry))
    {
        struct pcap_pkthdr FrameHeader;
        OsMemset(&FrameHeader, 0, sizeof(struct pcap_pkthdr));

        FrameHeader.TimeStamp.dwSec = (EC_T_DWORD)(Entry.Desc.qwTimestamp / 1000000000);
        FrameHeader.TimeStamp.dwUsec = (EC_T_DWORD)((Entry.Desc.qwTimestamp % 1000000000) / 1000);

        FrameHeader.caplen = FrameHeader.len = Entry.Desc.dwSize;
        WriteFrame(&FrameHeader, Entry.abyData);
    }
}

EC_T_VOID CPcapFileBufferedWriter::AddFrame(EC_T_DWORD dwFrameSize, EC_T_BYTE* pbyFrame)
{
    EC_T_LINK_FRAME_BUF_ENTRY Entry;
    OsMemset(&Entry, 0, sizeof(EC_T_LINK_FRAME_BUF_ENTRY));

    if ((EC_NULL == pbyFrame) || (0 == dwFrameSize)) return;

    if (dwFrameSize > ETHERNET_MAX_FRAMEBUF_LEN - sizeof(EC_T_LINK_FRAME_BUF_ENTRY_FRAME_DESC)) return;

    if (m_FrameBuffer.IsFull())
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CPcapRecorder overflow!"));
        return;
    }
    OsMemcpy(Entry.abyData, pbyFrame, dwFrameSize);

#if (defined EC_VERSION_WINDOWS)
    Entry.Desc.qwTimestamp = OsQueryMsecCount();
    Entry.Desc.qwTimestamp = Entry.Desc.qwTimestamp * 1000000;
#else
    OsSystemTimeGet(&Entry.Desc.qwTimestamp);
#endif
    /* CRC should be already stripped off */
    Entry.Desc.dwSize = dwFrameSize;
    OsDbgAssert(0 != dwFrameSize);
    m_FrameBuffer.Add(Entry);
}

EC_T_VOID CPcapRecorder::LogFrame(EC_T_DWORD dwLogFlags, EC_T_DWORD dwFrameSize, EC_T_BYTE* pbyFrame)
{
    EC_T_LINK_FRAME_BUF_ENTRY Entry;
    OsMemset(&Entry, 0, sizeof(EC_T_LINK_FRAME_BUF_ENTRY));

    EC_UNREFPARM(dwLogFlags);
    if ((EC_NULL == pbyFrame) || (0 == dwFrameSize)) return;

    if (dwFrameSize > ETHERNET_MAX_FRAMEBUF_LEN - sizeof(EC_T_LINK_FRAME_BUF_ENTRY_FRAME_DESC)) return;

    if (m_FrameBuffer.IsFull())
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CPcapRecorder overflow!"));
        return;
    }

    m_nLoggedFrameCount++;
    OsMemcpy(Entry.abyData, pbyFrame, dwFrameSize);

#if (defined EC_VERSION_WINDOWS)
    Entry.Desc.qwTimestamp = OsQueryMsecCount();
    Entry.Desc.qwTimestamp = Entry.Desc.qwTimestamp * 1000000;
#else
    OsSystemTimeGet(&Entry.Desc.qwTimestamp);
#endif

    /* CRC should be already stripped off */
    Entry.Desc.dwSize = dwFrameSize;
    OsDbgAssert(0 != dwFrameSize);
    m_FrameBuffer.Add(Entry);

    EcLogMsg(EC_LOG_LEVEL_VERBOSE_CYC, (pEcLogContext, EC_LOG_LEVEL_VERBOSE_CYC, "%d: CPcapRecorder::LogFrame(): frame %d",
        m_dwMasterId, m_nLoggedFrameCount));
}
#endif /* INCLUDE_PCAP_RECORDER */

/*-END OF SOURCE FILE--------------------------------------------------------*/
