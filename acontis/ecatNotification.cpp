/*-----------------------------------------------------------------------------
 * ecatNotification.cpp
 * Copyright                acontis technologies GmbH, Weingarten, Germany
 * Response                 Paul Bussmann
 * Description              EtherCAT Master generic notification handler
 *---------------------------------------------------------------------------*/

/*-LOGGING-------------------------------------------------------------------*/
#define pEcLogParms GetLogParms()

/*-INCLUDES------------------------------------------------------------------*/
#include "ecatNotification.h"
#include "SlaveInfo.h"

#ifndef EXCLUDE_RAS
#include <AtEmRasError.h>
#endif

/*-DEFINES-------------------------------------------------------------------*/
#define MAX_MSG_PER_ERROR   1 /* max. number of error messages printed */

#define SB_PORT_BLOCK_TIMEOUT   2000        /* msecs to wait for invalid slave node block */

/*-CLASS FUNCTIONS-----------------------------------------------------------*/
/*****************************************************************************/
/**
\brief  Constructor.
*/
CEmNotification::CEmNotification(
    EC_T_DWORD      dwMasterInstance,   /**< [in]   Master Instance */
    EC_T_BOOL       bRasClient          /**< [in]   Remote API client */
                                )
{
    m_dwMasterInstance = dwMasterInstance;

    OsMemset(m_adwErrorCounter, 0, sizeof(m_adwErrorCounter));

    m_bAllDevsOperational               = EC_FALSE;

    m_bRasServerDisconnect              = EC_FALSE;
    m_bRasClient                        = bRasClient;
    m_bDidIssueBlock                    = EC_FALSE;

    OsMemset(&m_oSlaveJobQueue, 0, sizeof(T_SLAVEJOBQUEUE));

    m_oSlaveJobQueue.pRead              = &(m_oSlaveJobQueue.Jobs[0]);
    m_oSlaveJobQueue.pWrite             = &(m_oSlaveJobQueue.Jobs[0]);

    m_dwClientID                        = INVALID_CLIENT_ID;
}

/*****************************************************************************/
/** \brief  EtherCAT notification
*
* This function is called on EtherCAT events.
* No EtherCAT functions may be called here (unless explicitly allowed in the documentation)!!
* It should be left as fast as possible.
*
* \return Currently always EC_E_NOERROR has to be returned.
*/
EC_T_DWORD CEmNotification::ecatNotify(
    EC_T_DWORD          dwCode,         /**< [in]   Notification code */
    EC_T_NOTIFYPARMS*   pParms          /**< [in]   Notification data */
                                      )
{
    printf("Running CEmNotification::ecatNotify dwCode = %d\n", dwCode);

    EC_T_ERROR_NOTIFICATION_DESC*   pErrorNotificationDesc  = (EC_T_ERROR_NOTIFICATION_DESC*)pParms->pbyInBuf;
    EC_T_NOTIFICATION_DESC*         pNotificationDesc       = (EC_T_NOTIFICATION_DESC*)pParms->pbyInBuf;
    static EC_T_DWORD               s_dwClearErrorMsecCount = 0;
    EC_T_DWORD                      dwRetVal                = EC_E_NOERROR;
    EC_T_DWORD                      dwRes                   = EC_E_ERROR;
    T_SLAVEJOBS                     oCurJob;

    switch (dwCode)
    {
        /************************/
        /* generic notification */
        /************************/
    case EC_NOTIFY_STATECHANGED: /* GEN|1 */
        {
            EC_T_STATECHANGE* pStateChangeParms = (EC_T_STATECHANGE*)pParms->pbyInBuf;

            EcLogMsg(EC_LOG_LEVEL_WARNING, (pEcLogContext, EC_LOG_LEVEL_WARNING, ecatGetText(EC_TXT_MASTER_STATE_CHANGE), ecatDeviceStateText(pStateChangeParms->oldState), ecatDeviceStateText(pStateChangeParms->newState)));

            if (eEcatState_OP == pStateChangeParms->newState)
            {
                m_bAllDevsOperational = EC_TRUE;
            }
        } break;
    case EC_NOTIFY_ETH_LINK_CONNECTED: /* GEN|2 */
        {
            /* Hint: No error, but show error clearance in Error.log */
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_CABLE_CONNECTED)));
        } break;
    case EC_NOTIFY_SB_STATUS:   /* GEN|3 */
        {
            /* Scanbus did finish it's processing when this Notification is called */
            dwRes = pNotificationDesc->desc.ScanBusNtfyDesc.dwResultCode;

            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SB_RESULT_OK), pNotificationDesc->desc.ScanBusNtfyDesc.dwSlaveCount));

            if (EC_E_NOERROR != dwRes)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Scan Bus returned with error: %s (0x%x)\n", ecatGetText(dwRes), dwRes));
            }
        } break;
    case EC_NOTIFY_DC_STATUS:   /* GEN|4 */
        {
            EC_T_DWORD  dwStatus = 0;

            /* Initialization of DC Instance finished when this notification is called */
            dwStatus  = EC_GETDWORD((pParms->pbyInBuf));

            if (EC_E_NOERROR != dwStatus)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Error: DC Initialization failed. ErrorCode = 0x%x\n", dwStatus));
            }
        } break;
    case EC_NOTIFY_DC_SLV_SYNC: /* GEN|5 */
        {
            /* This notification is called if state of slave deviation surveillance changes */
            EC_T_DC_SYNC_NTFY_DESC* pNtfySlv = &pNotificationDesc->desc.SyncNtfyDesc;
            if (pNtfySlv->IsInSync)
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_DCSLVSYNC_INSYNC), pNtfySlv->dwDeviation));
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_DCSLVSYNC_OUTOFSYNC), pNtfySlv->dwDeviation));
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "DC Slave \"%s\", EtherCAT address = %d\n", pNtfySlv->SlaveProp.achName, pNtfySlv->SlaveProp.wStationAddress));
            }
        } break;
    case EC_NOTIFY_DCM_SYNC:  /* GEN|9 */
        {
            /* This notification is called if state of DCM error monitoring changes */
            EC_T_DCM_SYNC_NTFY_DESC* pDcmInSyncNotify = &pNotificationDesc->desc.DcmInSyncDesc;
            if (pDcmInSyncNotify->IsInSync)
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_DCM_INSYNC), pDcmInSyncNotify->nCtlErrorNsecCur, pDcmInSyncNotify->nCtlErrorNsecAvg, pDcmInSyncNotify->nCtlErrorNsecMax));
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_DCM_OUTOFSYNC), pDcmInSyncNotify->nCtlErrorNsecCur, pDcmInSyncNotify->nCtlErrorNsecAvg, pDcmInSyncNotify->nCtlErrorNsecMax));
            }
        } break;
    case EC_NOTIFY_DCX_SYNC:  /* GEN|10 */
    {
        /* This notification is called if state of DCX error monitoring changes */
        EC_T_DCX_SYNC_NTFY_DESC* pDcxInSyncNotify = &pNotificationDesc->desc.DcxInSyncDesc;
        if (pDcxInSyncNotify->IsInSync)
        {
            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_DCX_INSYNC), pDcxInSyncNotify->nCtlErrorNsecCur,
                pDcxInSyncNotify->nCtlErrorNsecAvg,
                pDcxInSyncNotify->nCtlErrorNsecMax,
                pDcxInSyncNotify->nTimeStampDiff));
        }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_DCX_OUTOFSYNC), pDcxInSyncNotify->nCtlErrorNsecCur,
                pDcxInSyncNotify->nCtlErrorNsecAvg,
                pDcxInSyncNotify->nCtlErrorNsecMax,
                pDcxInSyncNotify->nTimeStampDiff));
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(pDcxInSyncNotify->dwErrorCode)));
        }
    } break;
    case EC_NOTIFY_SLAVE_STATECHANGED: /* GEN|21 */
        {
            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SLAVE_STATECHANGED),
                pNotificationDesc->desc.SlaveStateChangedDesc.SlaveProp.wStationAddress,
                ecatDeviceStateText(pNotificationDesc->desc.SlaveStateChangedDesc.newState)));
        } break;
    case EC_NOTIFY_SLAVES_STATECHANGED: /* GEN|22 */
        {
            EC_T_DWORD dwSlaveIdx = 0;
            for (dwSlaveIdx = 0; dwSlaveIdx < pNotificationDesc->desc.SlavesStateChangedDesc.wCount; dwSlaveIdx++)
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SLAVE_STATECHANGED),
                    pNotificationDesc->desc.SlavesStateChangedDesc.SlaveStates[dwSlaveIdx].wStationAddress,
                    ecatDeviceStateText((EC_T_STATE)pNotificationDesc->desc.SlavesStateChangedDesc.SlaveStates[dwSlaveIdx].byState)));
            }
        } break;

    case EC_NOTIFY_RAWCMD_DONE:     /* GEN|100 */
        {
            /* This notification is called to provide the response to a ecatQueueRawCmd Call */
            EC_T_RAWCMDRESPONSE_NTFY_DESC* pRawCmdDesc = &((EC_T_NOTIFICATION_DESC*)pParms->pbyInBuf)->desc.RawCmdRespNtfyDesc;

            EC_T_DWORD dwLength = pRawCmdDesc->dwLength;
            if (dwLength > ETHERNET_MAX_FRAME_LEN)
            {
                dwLength = ETHERNET_MAX_FRAME_LEN;
            }
            OsMemcpy(oCurJob.JobData.RawCmdJob.abyData, pRawCmdDesc->pbyData, dwLength);

            EnqueueJob(&oCurJob, dwCode, pRawCmdDesc, sizeof(EC_T_RAWCMDRESPONSE_NTFY_DESC)-sizeof(EC_T_BYTE*));
        } break;
    case EC_NOTIFY_SLAVE_PRESENCE:      /* GEN|101 */
        {
            if (pNotificationDesc->desc.SlavePresenceDesc.bPresent)
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SLAVE_PRESENT), pNotificationDesc->desc.SlavePresenceDesc.wStationAddress));
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SLAVE_ABSENT), pNotificationDesc->desc.SlavePresenceDesc.wStationAddress));
            }
        } break;
    case EC_NOTIFY_SLAVES_PRESENCE:     /* GEN|102 */
        {
            EC_T_DWORD dwSlaveIdx = 0;
            for (dwSlaveIdx = 0; dwSlaveIdx < pNotificationDesc->desc.SlavesPresenceDesc.wCount; dwSlaveIdx++)
            {
                if (pNotificationDesc->desc.SlavesPresenceDesc.SlavePresence[dwSlaveIdx].bPresent)
                {
                    EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SLAVE_PRESENT), pNotificationDesc->desc.SlavesPresenceDesc.SlavePresence[dwSlaveIdx].wStationAddress));
                }
                else
                {
                    EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SLAVE_ABSENT), pNotificationDesc->desc.SlavesPresenceDesc.SlavePresence[dwSlaveIdx].wStationAddress));
                }
            }
        } break;
    case EC_NOTIFY_MASTER_RED_STATECHANGED: /* GEN|104 */
        {
            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_MASTER_RED_STATE_CHANGE), ((EC_TRUE == pNotificationDesc->desc.StatusCode) ? "ACTIVE" : "INACTIVE")));
        } break;
    case EC_NOTIFY_MASTER_RED_FOREIGN_SRC_MAC: /* GEN|105 */
        {
            /* nothing to do */
        } break;

        /************************/
        /* mailbox notification */
        /************************/
    case EC_NOTIFY_MBOXRCV:     /* MBOXRCV|0 */
            {
                EC_T_MBXTFER* pMbxTfer = EC_NULL; /* mailbox transfer object */

                pMbxTfer = (EC_T_MBXTFER*)pParms->pbyInBuf;
                OsDbgAssert(pMbxTfer != EC_NULL);

                switch (pMbxTfer->eMbxTferType)
                {
                    /***************************************************************************************************************************/
                case eMbxTferType_COE_SDO_DOWNLOAD:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_COE_SDO_DNLD_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    break;

                    /***************************************************************************************************************************/
                case eMbxTferType_COE_SDO_UPLOAD:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_COE_SDO_UPLD_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    break;

                    /***************************************************************************************************************************/
                case eMbxTferType_COE_GETODLIST:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_COE_GETODL_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    break;

                    /***************************************************************************************************************************/
                case eMbxTferType_COE_GETOBDESC:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_COE_GETOBDESC_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    break;

                    /***************************************************************************************************************************/
                case eMbxTferType_COE_GETENTRYDESC:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        /* don't show message if sub-index does not exist */
                        if ((pMbxTfer->dwErrorCode != EC_E_INVALIDDATA)
                        &&  (pMbxTfer->dwErrorCode != EC_E_SDO_ABORTCODE_OFFSET))
                        {
                            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_COE_GETENTRYDESC_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                        }
                    }
                    pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    break;

                    /***************************************************************************************************************************/
                case eMbxTferType_COE_EMERGENCY:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_COE_EMRG_TFER_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_COE_EMRG), pMbxTfer->dwTferId, pMbxTfer->dwDataLen,
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.wStationAddress,
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.wErrorCode,
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.byErrorRegister,
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.abyData[0],
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.abyData[1],
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.abyData[2],
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.abyData[3],
                            (EC_T_INT)pMbxTfer->MbxData.CoE_Emergency.abyData[4]));
                    }
                    break;
#ifdef INCLUDE_FOE_SUPPORT
                    /***************************************************************************************************************************/
                case eMbxTferType_FOE_SEG_DOWNLOAD:
                    {
                        EnqueueJob(&oCurJob, dwCode, pMbxTfer, sizeof(EC_T_MBXTFER));
                    } break;
                case eMbxTferType_FOE_SEG_UPLOAD:
                    if (eMbxTferStatus_TferWaitingForContinue == pMbxTfer->eTferStatus)
                    {
                        /* received segment size */
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Foe segment of size %d uploaded.\n", pMbxTfer->dwDataLen));
                        EnqueueJob(&oCurJob, dwCode, pMbxTfer, sizeof(EC_T_MBXTFER));
                    }
                    break;

                case eMbxTferType_FOE_FILE_DOWNLOAD:
                    if (pMbxTfer->eTferStatus == eMbxTferStatus_TferReqError)
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_FOE_DNLD_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                        pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    }
                    else if (pMbxTfer->eTferStatus != eMbxTferStatus_TferDone)
                    {
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "FoE download %3d%%\n", ((pMbxTfer->MbxData.FoE.dwTransferredBytes * 100) / pMbxTfer->MbxData.FoE.dwFileSize)));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "FoE download done\n"));

                        pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    }
                    break;
                    /***************************************************************************************************************************/
                case eMbxTferType_FOE_FILE_UPLOAD:
                    if (pMbxTfer->eTferStatus == eMbxTferStatus_TferReqError)
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_FOE_UPLD_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                        pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    }
                    else if (pMbxTfer->eTferStatus != eMbxTferStatus_TferDone)
                    {
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "FoE upload %3d%%\n", ((pMbxTfer->MbxData.FoE.dwTransferredBytes * 100) / pMbxTfer->dwDataLen)));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "FoE upload done\n"));

                        pMbxTfer->eTferStatus = eMbxTferStatus_Idle;
                    }
                    break;
#endif /* INCLUDE_FOE_SUPPORT */
#ifdef INCLUDE_SOE_SUPPORT
                    /***************************************************************************************************************************/
                case eMbxTferType_SOE_WRITEREQUEST:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SOE_WRITE_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    break;
                    /***************************************************************************************************************************/
                case eMbxTferType_SOE_READREQUEST:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SOE_READ_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    break;
                    /***************************************************************************************************************************/
                case eMbxTferType_SOE_EMERGENCY:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SOE_EMRG_TFER_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SOE_EMRG),
                            pMbxTfer->dwTferId,
                            pMbxTfer->dwDataLen,
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Emergency.wStationAddress,
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Emergency.wHeader,
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Emergency.abyData[0],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Emergency.abyData[1],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Emergency.abyData[2],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Emergency.abyData[3],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Emergency.abyData[4]));

                    }
                    break;
                    /***************************************************************************************************************************/
                case eMbxTferType_SOE_NOTIFICATION:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SOE_NOTIFIC_TFER_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_SOE_NOTIFICATION), pMbxTfer->dwTferId, pMbxTfer->dwDataLen,
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.wStationAddress,
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.wHeader,
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.wIdn,
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.abyData[0],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.abyData[1],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.abyData[2],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.abyData[3],
                            (EC_T_INT)pMbxTfer->MbxData.SoE_Notification.abyData[4]));
                    }
                    break;
#endif
#ifdef INCLUDE_VOE_SUPPORT
                case eMbxTferType_VOE_MBX_WRITE:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_VOE_DNLD_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    break;
                case eMbxTferType_VOE_MBX_READ:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_VOE_UPLD_ERROR), pMbxTfer->eTferStatus, pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode)));
                    }
                    break;
#endif
#ifdef INCLUDE_AOE_SUPPORT
                case eMbxTferType_AOE_WRITE:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_AOE_CMD_ERROR), pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode),
                            pMbxTfer->MbxData.AoE_Response.dwErrorCode, pMbxTfer->MbxData.AoE_Response.dwCmdResult));
                    }
                    break;
                case eMbxTferType_AOE_READ:
                    if ((pMbxTfer->eTferStatus != eMbxTferStatus_TferDone) || (pMbxTfer->dwErrorCode != EC_E_NOERROR))
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_AOE_CMD_ERROR), pMbxTfer->dwErrorCode, ecatGetText(pMbxTfer->dwErrorCode),
                            pMbxTfer->MbxData.AoE_Response.dwErrorCode, pMbxTfer->MbxData.AoE_Response.dwCmdResult));
                    }
                    break;
#endif
                default:
                    OsDbgAssert(EC_FALSE);
                    break;
            }
        } break;
#ifdef INCLUDE_COE_PDO_SUPPORT
#error "Notification needs to be done ! Unconditionally !"
    case EC_NOTIFY_COE_TX_PDO:  /* MBOXRCV|1 */
        {
            OsDbgAssert(EC_FALSE);
        } break;
#endif
        /************************/
        /* Scanbus notification */
        /************************/
    case EC_NOTIFY_SB_MISMATCH:
    case EC_NOTIFY_SB_DUPLICATE_HC_NODE:
        {
            EnqueueJob(&oCurJob, dwCode, &pNotificationDesc->desc.ScanBusMismatch, sizeof(EC_T_SB_MISMATCH_DESC));
        } break;

        /**********************/
        /* error notification */
        /**********************/
    case EC_NOTIFY_CYCCMD_WKC_ERROR:    /* ERR|1 */
        {
            m_adwErrorCounter[dwCode & 0xFFFF]++;
            if (m_adwErrorCounter[dwCode & 0xFFFF] > MAX_MSG_PER_ERROR)
                break;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_CYCCMD_WKC_ERROR),
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_MASTER_INITCMD_WKC_ERROR:    /* ERR|2 */
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_MASTINITCMD_WKC_ERROR),
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_SLAVE_INITCMD_WKC_ERROR:     /* ERR|3 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.WkcErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMD_WKC_ERROR),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_EOE_MBXSND_WKC_ERROR:    /* ERR|7 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.WkcErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_EOEMBXSND_WKC_ERROR),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_COE_MBXSND_WKC_ERROR:    /* ERR|8 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.WkcErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_COEMBXSND_WKC_ERROR),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_FOE_MBXSND_WKC_ERROR:    /* ERR|9 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.WkcErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_FOEMBXSND_WKC_ERROR),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_VOE_MBXSND_WKC_ERROR:    /* ERR|34 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.WkcErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_VOEMBXSND_WKC_ERROR),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_FRAME_RESPONSE_ERROR: /* ERR|10 */
        {
        /*
        This error will be indicated if the actually received Ethernet frame does
        not match to the frame expected or if a expected frame was not received.

          Missing response (timeout, eRspErr_NO_RESPONSE) acyclic frames:
          Acyclic Ethernet frames are internally queued by the master and sent to the slaves
          at a later time (usually after sending cyclic frames).
          The master will monitor the time between queueing such a frame and receiving the result.
          If a maximum time is exceeded then this error will be indicated.
          This maximum time will be determined by the parameter MasterConfig.dwEcatCmdTimeout when
          the master is initialized.
          The master will retry sending the frame if the master configuration parameter dwEcatCmdMaxRetries
          is set to a value greater than 1.
          In case of a retry the eRspErr_RETRY_FAIL error is signaled, if the number of retries has
          elapsed the eRspErr_NO_RESPONSE error is signaled.
          Possible reasons:
          a)    the frame was not received at all (due to bus problems)
          In this case the achErrorInfo[] member of the error notification descriptor will
          contain the string "L".
          b)    the frame was sent too late by the master due to an improper configuration.
          In this case the achErrorInfo[] member of the error notification descriptor will
          contain the string "T".
          to avoid this error the configuration may be changed as follows:
          - higher value for master configuration parameter dwMaxSentQueuedFramesPerCyc
          - higher timeout value (master configuration parameter dwEcatCmdTimeout)
          Note: If the frame was sent too late by the master (due to improper configuration values)
          it will also be received too late and the master then signals an eRspErr_WRONG_IDX
          or eRspErr_UNEXPECTED error (as the master then doesn't expect to receive this frame).

            Missing response (timeout, eRspErr_NO_RESPONSE) cyclic frames:
            A response to all cyclic frames must occur until the next cycle starts.
            If the first cyclic frame is sent the master checks whether all cyclic frames of the
            last cycle were received. If there is one frame missing this error is indicated.
            Possible reasons:
            a)  the frame was not received (due to bus problems)
            b)  too many or too long frames are sent by the master due to a improper configuration.
            */
            m_adwErrorCounter[dwCode & 0xFFFF]++;
            if (m_adwErrorCounter[dwCode & 0xFFFF] > MAX_MSG_PER_ERROR)
                break;
            {
                const EC_T_CHAR* pszTextCause = EC_NULL;
                switch (pErrorNotificationDesc->desc.FrameRspErrDesc.EErrorType)
                {
                case eRspErr_NO_RESPONSE:   pszTextCause = ecatGetText(EC_TXT_FRAME_RESPONSE_ERRTYPE_NO);          break;
                case eRspErr_WRONG_IDX:     pszTextCause = ecatGetText(EC_TXT_FRAME_RESPONSE_ERRTYPE_WRONG);       break;
                case eRspErr_UNEXPECTED:    pszTextCause = ecatGetText(EC_TXT_FRAME_RESPONSE_ERRTYPE_UNEXPECTED);  break;
                case eRspErr_RETRY_FAIL:    pszTextCause = ecatGetText(EC_TXT_FRAME_RESPONSE_ERRTYPE_RETRY_FAIL);  break;
                default:                    pszTextCause = (EC_T_CHAR*)"@@internal error@@";  break;
                }
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_FRMRESP_NORETRY), pszTextCause, (pErrorNotificationDesc->desc.FrameRspErrDesc.bIsCyclicFrame ? ecatGetText(EC_TXT_FRAME_TYPE_CYCLIC) : ecatGetText(EC_TXT_FRAME_TYPE_ACYCLIC))));
            }

            if (pErrorNotificationDesc->desc.FrameRspErrDesc.bIsCyclicFrame)
            {
                if (pErrorNotificationDesc->achErrorInfo[0] != '\0')
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_ADDERRINFO), pErrorNotificationDesc->achErrorInfo));
                }

                if (pErrorNotificationDesc->desc.FrameRspErrDesc.EErrorType == eRspErr_WRONG_IDX)
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_CMDIDXACTVAL), pErrorNotificationDesc->desc.FrameRspErrDesc.byEcCmdHeaderIdxAct));
                }
            }
            else
            {
                if ((pErrorNotificationDesc->desc.FrameRspErrDesc.EErrorType != eRspErr_UNEXPECTED))
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_CMDIDXSETVAL), pErrorNotificationDesc->desc.FrameRspErrDesc.byEcCmdHeaderIdxSet));
                }

                if ((pErrorNotificationDesc->desc.FrameRspErrDesc.EErrorType == eRspErr_UNEXPECTED)
                 || (pErrorNotificationDesc->desc.FrameRspErrDesc.EErrorType == eRspErr_WRONG_IDX))
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_CMDIDXACTVAL), pErrorNotificationDesc->desc.FrameRspErrDesc.byEcCmdHeaderIdxAct));
                }
            }
        } break;
    case EC_NOTIFY_SLAVE_INITCMD_RESPONSE_ERROR: /* ERR|11 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.InitCmdErrDesc.SlaveProp;

            switch (pErrorNotificationDesc->desc.InitCmdErrDesc.EErrorType)
            {
            case eInitCmdErr_NO_RESPONSE:
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMDRSPERR_NR), pSlaveProp->achName, pSlaveProp->wStationAddress,
                        pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
                } break;
            case eInitCmdErr_VALIDATION_ERR:
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMDRSPERR_VE), pSlaveProp->achName, pSlaveProp->wStationAddress,
                        pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
                } break;
            case eInitCmdErr_FAILED:
                {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMDRSPERR_FLD), pSlaveProp->achName, pSlaveProp->wStationAddress, 
                    pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
                } break;
            case eInitCmdErr_NOT_PRESENT:
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMDRSPERR_NPR), pSlaveProp->achName,
                    pSlaveProp->wStationAddress, pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
            } break;
            case eInitCmdErr_ALSTATUS_ERROR:
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMDRSPERR_ALS), pSlaveProp->achName,
                    pSlaveProp->wStationAddress, pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
            } break;
            case eInitCmdErr_MBXSLAVE_ERROR:
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMDRSPERR_MBX), pSlaveProp->achName,
                    pSlaveProp->wStationAddress, pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
            } break;
            case eInitCmdErr_PDI_WATCHDOG:
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVINITCMDRSPERR_PWD), pSlaveProp->achName,
                    pSlaveProp->wStationAddress, pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
            } break;
            default:
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "@@internal error@@\n"));
                } break;
            }
        } break;
    case EC_NOTIFY_MASTER_INITCMD_RESPONSE_ERROR:   /* ERR|12 */
        {
            switch (pErrorNotificationDesc->desc.InitCmdErrDesc.EErrorType)
            {
            case eInitCmdErr_NO_RESPONSE:
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_MASTINITCMDRSPERR_NR), pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
                } break;
            case eInitCmdErr_VALIDATION_ERR:
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_MASTINITCMDRSPERR_VE), pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
                } break;
            default:
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "@@internal error@@\n"));
                } break;
            }
        } break;
    case EC_NOTIFY_MBSLAVE_INITCMD_TIMEOUT:     /* ERR|14 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.InitCmdErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_MBSLV_INITCMDTO), pSlaveProp->achName, pSlaveProp->wStationAddress,
                pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
        } break;
    case EC_NOTIFY_NOT_ALL_DEVICES_OPERATIONAL:     /* ERR|15 */
        {
            m_bAllDevsOperational = EC_FALSE;
            m_adwErrorCounter[dwCode & 0xFFFF]++;
            if (m_adwErrorCounter[dwCode & 0xFFFF] > MAX_MSG_PER_ERROR)
                break;
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_NOT_ALL_DEVS_OP)));
        } break;
    case EC_NOTIFY_ETH_LINK_NOT_CONNECTED: /* ERR|16 */
        {
            m_adwErrorCounter[dwCode & 0xFFFF]++;
            if (m_adwErrorCounter[dwCode & 0xFFFF] > MAX_MSG_PER_ERROR)
                break;
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_CABLE_NOT_CONNECTED)));
        } break;
    case EC_NOTIFY_RED_LINEBRK:         /* ERR|18 */
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_REDLINEBREAK),
                pErrorNotificationDesc->desc.RedChangeDesc.wNumOfSlavesMain,
                pErrorNotificationDesc->desc.RedChangeDesc.wNumOfSlavesRed));
        } break;
    case EC_NOTIFY_STATUS_SLAVE_ERROR:  /* ERR|19 */
        {
            m_adwErrorCounter[dwCode & 0xFFFF]++;
            if (m_adwErrorCounter[dwCode & 0xFFFF] > MAX_MSG_PER_ERROR)
                break;
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVERR_DETECTED)));
        } break;
    case EC_NOTIFY_SLAVE_ERROR_STATUS_INFO:     /* ERR|20 */
        {
            EC_T_SLAVE_ERROR_INFO_DESC* pSlaveError = &pErrorNotificationDesc->desc.SlaveErrInfoDesc;
            EC_T_SLAVE_PROP*            pSlaveProp  = &pSlaveError->SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVERR_INFO),
                pSlaveProp->achName, pSlaveProp->wStationAddress,
                ecatSlaveStateText(pErrorNotificationDesc->desc.SlaveErrInfoDesc.wStatus),
                "",
                pErrorNotificationDesc->desc.SlaveErrInfoDesc.wStatus,
                SlaveDevStatusCodeText((pErrorNotificationDesc->desc.SlaveErrInfoDesc.wStatusCode)),
                pErrorNotificationDesc->desc.SlaveErrInfoDesc.wStatusCode));
        } break;
    case EC_NOTIFY_SLAVE_NOT_ADDRESSABLE:       /* ERR|21 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.WkcErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLV_NOT_ADDRABLE), pSlaveProp->achName, pSlaveProp->wStationAddress));
        } break;
#ifdef INCLUDE_SOE_SUPPORT
    case EC_NOTIFY_SOE_MBXSND_WKC_ERROR:    /* ERR|23 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.WkcErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SOEMBXSND_WKC_ERROR),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                EcatCmdShortText(pErrorNotificationDesc->desc.WkcErrDesc.byCmd),
                pErrorNotificationDesc->desc.WkcErrDesc.dwAddr,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcAct,
                pErrorNotificationDesc->desc.WkcErrDesc.wWkcSet));
        } break;
    case EC_NOTIFY_SOE_WRITE_ERROR:         /* ERR|24 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.InitCmdErrDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SOEMBX_WRITE_ERROR), pSlaveProp->achName, pSlaveProp->wStationAddress,
                pErrorNotificationDesc->desc.InitCmdErrDesc.achStateChangeName));
        } break;
#endif
    case EC_NOTIFY_MBSLAVE_COE_SDO_ABORT:   /* ERR|25 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.SdoAbortDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_MBSLV_SDO_ABORT),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                ecatGetText(pErrorNotificationDesc->desc.SdoAbortDesc.dwErrorCode),
                pErrorNotificationDesc->desc.SdoAbortDesc.dwErrorCode,
                pErrorNotificationDesc->desc.SdoAbortDesc.wObjIndex,
                pErrorNotificationDesc->desc.SdoAbortDesc.bySubIndex));
        } break;
    case EC_NOTIFY_CLIENTREGISTRATION_DROPPED:  /* ERR|26 */
        {
            EC_T_DWORD dwDeinitForConfiguration = EC_FALSE;

            if (pParms->dwInBufSize >= sizeof(EC_T_DWORD))
            {
                dwDeinitForConfiguration = *((EC_T_DWORD*)pParms->pbyInBuf);
            }
            if (dwDeinitForConfiguration == 1)
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Re-configuring EC-Master\n"));

                m_dwClientID = INVALID_CLIENT_ID;
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_CLNTREGDROP)));
            }
        } break;


    case EC_NOTIFY_RED_LINEFIXED:  /* ERR|27 */
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_REDLINEFIXED)));
        } break;
#ifdef INCLUDE_FOE_SUPPORT
    case EC_NOTIFY_FOE_MBSLAVE_ERROR:   /* ERR|28 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.FoeErrorDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_MBSLV_FOE_ABORT),
                pSlaveProp->achName,
                pSlaveProp->wStationAddress,
                pErrorNotificationDesc->desc.FoeErrorDesc.achErrorString,
                pErrorNotificationDesc->desc.FoeErrorDesc.dwErrorCode));

        } break;
#endif
    case EC_NOTIFY_MBXRCV_INVALID_DATA: /* ERR|29 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.MbxRcvInvalidDataDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_MBXRCV_INVALID_DATA), pSlaveProp->achName, pSlaveProp->wStationAddress));
        } break;
    case EC_NOTIFY_PDIWATCHDOG:         /* ERR|30 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.PdiWatchdogDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_PDIWATCHDOG), pSlaveProp->achName, pSlaveProp->wStationAddress));
        } break;
    case EC_NOTIFY_SLAVE_NOTSUPPORTED:      /* ERR|31 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.SlaveNotSupportedDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLAVE_NOTSUPPORTED), pSlaveProp->achName, pSlaveProp->wStationAddress));
        } break;
    case EC_NOTIFY_SLAVE_UNEXPECTED_STATE:      /* ERR|32 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.SlaveUnexpectedStateDesc.SlaveProp;
            EC_T_STATE       curState   = pErrorNotificationDesc->desc.SlaveUnexpectedStateDesc.curState;
            EC_T_STATE       expState   = pErrorNotificationDesc->desc.SlaveUnexpectedStateDesc.expState;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLAVE_UNEXPECTED_STATE), pSlaveProp->achName, pSlaveProp->wStationAddress,
                ecatDeviceStateText(curState), ecatDeviceStateText(expState)));
        } break;
    case EC_NOTIFY_ALL_DEVICES_OPERATIONAL:     /* ERR|33 */
        {
            m_bAllDevsOperational = EC_TRUE;
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_ALL_DEVS_OP)));
        } break;
    case EC_NOTIFY_EEPROM_CHECKSUM_ERROR:       /* ERR|35 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.EEPROMChecksumErrorDesc.SlaveProp;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_EEPROM_CHECKSUM_ERROR), pSlaveProp->achName, pSlaveProp->wStationAddress));
        } break;
    case EC_NOTIFY_LINE_CROSSED:                /* ERR|36 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pNotificationDesc->desc.CrossedLineDesc.SlaveProp;
            EC_T_WORD        wPort      = pNotificationDesc->desc.CrossedLineDesc.wInputPort;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_LINE_CROSSED), pSlaveProp->achName, pSlaveProp->wAutoIncAddr, pSlaveProp->wStationAddress, wPort));
        } break;
    case EC_NOTIFY_JUNCTION_RED_CHANGE:         /* ERR|37 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.JunctionRedChangeDesc.SlaveProp;
            EC_T_BOOL        bLineBreak = pErrorNotificationDesc->desc.JunctionRedChangeDesc.bLineBreak;
            EC_T_WORD        wPort = pErrorNotificationDesc->desc.JunctionRedChangeDesc.wPort;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_JUNCTION_RED_CHANGE), bLineBreak, pSlaveProp->achName, pSlaveProp->wStationAddress, wPort));
        } break;
    case EC_NOTIFY_SLAVES_UNEXPECTED_STATE:      /* ERR|38 */
        {
            EC_T_DWORD dwSlaveIdx = 0;
            for (dwSlaveIdx = 0; dwSlaveIdx < pErrorNotificationDesc->desc.SlavesUnexpectedStateDesc.wCount; dwSlaveIdx++)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLAVES_UNEXPECTED_STATE),
                    pErrorNotificationDesc->desc.SlavesUnexpectedStateDesc.SlaveStates[dwSlaveIdx].wStationAddress,
                    ecatDeviceStateText(pErrorNotificationDesc->desc.SlavesUnexpectedStateDesc.SlaveStates[dwSlaveIdx].curState),
                    ecatDeviceStateText(pErrorNotificationDesc->desc.SlavesUnexpectedStateDesc.SlaveStates[dwSlaveIdx].expState)));
            }
        } break;
    case EC_NOTIFY_SLAVES_ERROR_STATUS:         /* ERR|39 */
        {
            EC_T_DWORD dwSlaveIdx = 0;
            for (dwSlaveIdx = 0; dwSlaveIdx < pErrorNotificationDesc->desc.SlavesErrDesc.wCount; dwSlaveIdx++)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_SLVERR_INFO), "",
                    pErrorNotificationDesc->desc.SlavesErrDesc.SlaveError[dwSlaveIdx].wStationAddress,
                    SlaveDevStateText(((pErrorNotificationDesc->desc.SlavesErrDesc.SlaveError[dwSlaveIdx].wStatus) & 0xf)),
                    (((pErrorNotificationDesc->desc.SlavesErrDesc.SlaveError[dwSlaveIdx].wStatus) & 0x10) ? " ERROR" : ""),
                    pErrorNotificationDesc->desc.SlavesErrDesc.SlaveError[dwSlaveIdx].wStatus,
                    SlaveDevStatusCodeText((pErrorNotificationDesc->desc.SlavesErrDesc.SlaveError[dwSlaveIdx].wStatusCode)),
                    pErrorNotificationDesc->desc.SlavesErrDesc.SlaveError[dwSlaveIdx].wStatusCode));
            }
        } break;
    case EC_NOTIFY_FRAMELOSS_AFTER_SLAVE:       /* ERR|40 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.FramelossAfterSlaveDesc.SlaveProp;
            EC_T_WORD        wPort = pErrorNotificationDesc->desc.FramelossAfterSlaveDesc.wPort;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_FRAMELOSS_AFTER_SLAVE),
                pSlaveProp->achName, pSlaveProp->wAutoIncAddr, pSlaveProp->wStationAddress, wPort));
        } break;
    case EC_NOTIFY_S2SMBX_ERROR:       /* ERR|41 */
        {
            EC_T_SLAVE_PROP* pSlaveProp = &pErrorNotificationDesc->desc.S2SMbxErrorDesc.SlaveProp;
            EC_T_WORD        wTargetFixedAddress = pErrorNotificationDesc->desc.S2SMbxErrorDesc.wTargetFixedAddress;
            EC_T_DWORD       dwErrorCode = pErrorNotificationDesc->desc.S2SMbxErrorDesc.dwErrorCode;

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_S2SMBX_ERROR),
                pSlaveProp->achName, pSlaveProp->wStationAddress, wTargetFixedAddress, dwErrorCode));
        } break;
#ifdef INCLUDE_HOTCONNECT
    /*case EC_NOTIFY_HC_DETECTALLGROUPS:      / * HC| 1: HC Detect All Groups done * / */
    case EC_NOTIFY_HC_DETECTADDGROUPS:      /* HC| 2: HC Enhance Detect All Groups done */
    case EC_NOTIFY_HC_PROBEALLGROUPS:       /* HC| 3: HC Probe All Groups done */
        {
            /* HC Group Detection did finish it's processing when this Notification is called */
            dwRes = pNotificationDesc->desc.HCDetAllGrps.dwResultCode;

            if (EC_E_NOERROR == dwRes)
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_HC_DETAGRESULT_OK),
                    pNotificationDesc->desc.HCDetAllGrps.dwGroupCount,
                    pNotificationDesc->desc.HCDetAllGrps.dwGroupsPresent,
                    pNotificationDesc->desc.HCDetAllGrps.dwGroupMask));
            }

            if (EC_E_NOERROR != dwRes)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, ecatGetText(EC_TXT_HC_DETAGRESULT_ERROR),
                    ecatGetText(pNotificationDesc->desc.HCDetAllGrps.dwResultCode),
                    pNotificationDesc->desc.HCDetAllGrps.dwResultCode,
                    pNotificationDesc->desc.HCDetAllGrps.dwGroupCount));
            }
        } break;
#endif /* INCLUDE_HOTCONNECT */
    case EC_NOTIFY_HC_TOPOCHGDONE:          /* HC| 4:  HC Topology Change done */
        {
            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, ecatGetText(EC_TXT_HC_TOPOCHGDONE), ecatGetText(pNotificationDesc->desc.StatusCode), pNotificationDesc->desc.StatusCode));
        } break;
    default:
        {
            /* print notification name and code */
            OsDbgAssert(EC_FALSE);
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "ecatNotify: name = %s code = 0x%x\n", ecatGetNotifyText(dwCode), dwCode));
        } break;
    }

    /* clear error counter every 5 seconds */
    if (OsQueryMsecCount() > s_dwClearErrorMsecCount)
    {
        /* reset error counters */
        OsMemset( m_adwErrorCounter, 0, sizeof(m_adwErrorCounter));
        s_dwClearErrorMsecCount = OsQueryMsecCount() + 5000;
    }
    return dwRetVal;
}

/*****************************************************************************/
/**
\brief  Process Notification Jobs.

This function processes the results enqueued by EcatNotify in an asynchronous Matter
\return EC_FALSE on error, EC_TRUE otherwise.
*/
EC_T_BOOL CEmNotification::ProcessNotificationJobs(EC_T_VOID)
{
    T_SLAVEJOBS TmpJob     = {0};
    EC_T_BOOL   bProcessed = EC_FALSE;

    /* process all jobs */
    while (DequeueJob(&TmpJob))
    {
        switch (TmpJob.dwCode)
        {
        case EC_NOTIFY_SB_MISMATCH:
        case EC_NOTIFY_SB_DUPLICATE_HC_NODE:
            {
                EC_T_SB_MISMATCH_DESC*  pScanBusMismatchJob = &TmpJob.JobData.SBSlaveMismatchJob;
                EC_T_DWORD              dwRes               = 0;

                /* Check if we have a mismatch for the first slave on the bus */
                if (pScanBusMismatchJob->wPrevAIncAddress != INVALID_AUTO_INC_ADDR)
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Slave mismatch after %d, Auto-inc Address 0x%04x, port %d:",
                        pScanBusMismatchJob->wPrevFixedAddress,
                        pScanBusMismatchJob->wPrevAIncAddress,
                        pScanBusMismatchJob->wPrevPort));
                }
                else
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Scan bus mismatch for the first slave on bus"));
                }
                /* handle hot connect group identification error */
                if (!pScanBusMismatchJob->bIdentValidationError)
                {
                    if (0 != pScanBusMismatchJob->dwCfgProdCode)
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Expected Slave: %s (0x%x), %s (0x%x)\n",
                            SlaveProdCodeText((T_eEtherCAT_Vendor)pScanBusMismatchJob->dwCfgVendorId, (T_eEtherCAT_ProductCode)pScanBusMismatchJob->dwCfgProdCode),
                            pScanBusMismatchJob->dwCfgProdCode,
                            SlaveVendorText((T_eEtherCAT_Vendor)pScanBusMismatchJob->dwCfgVendorId),
                            pScanBusMismatchJob->dwCfgVendorId));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Expected Slave: ----\n"));
                    }
                    if (0 != pScanBusMismatchJob->dwBusProdCode)
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Found Slave...: %s (0x%x), %s (0x%x)\n",                            
                            SlaveProdCodeText((T_eEtherCAT_Vendor)pScanBusMismatchJob->dwBusVendorId, (T_eEtherCAT_ProductCode)pScanBusMismatchJob->dwBusProdCode),
                            pScanBusMismatchJob->dwBusProdCode,
                            SlaveVendorText((T_eEtherCAT_Vendor)pScanBusMismatchJob->dwBusVendorId),
                            pScanBusMismatchJob->dwBusVendorId));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Found Slave...: ----\n"));
                    }
                }
                else
                {
                    if (EC_NOTIFY_SB_DUPLICATE_HC_NODE == TmpJob.dwCode)
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR,
                            "Identification value of slave %s (Auto inc. address: 0x%x) is duplicated: %d",
                            SlaveProdCodeText((T_eEtherCAT_Vendor)pScanBusMismatchJob->dwBusVendorId, (T_eEtherCAT_ProductCode)pScanBusMismatchJob->dwBusProdCode),
                            pScanBusMismatchJob->wBusAIncAddress,
                            pScanBusMismatchJob->dwCmdData));
                    }
                    else
                    {
                        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR,
                            "Identification value of slave %s (Auto inc. address: 0x%x) is incorrect. Actual: %d -> expected: %d",
                            SlaveProdCodeText((T_eEtherCAT_Vendor)pScanBusMismatchJob->dwBusVendorId, (T_eEtherCAT_ProductCode)pScanBusMismatchJob->dwBusProdCode),
                            pScanBusMismatchJob->wBusAIncAddress,
                            pScanBusMismatchJob->dwCmdData,
                            pScanBusMismatchJob->dwCmdVData));
                    }
                }
                /* unexpected bus slave can't block first slave on the network */
                if ((INVALID_AUTO_INC_ADDR != pScanBusMismatchJob->wBusAIncAddress) && (INVALID_AUTO_INC_ADDR != pScanBusMismatchJob->wPrevAIncAddress))
                {
                    if (0 < emHCGetNumGroupMembers(m_dwMasterInstance, 1))
                    {
                        /* now block unexpected bus slave */
                        dwRes = emBlockNode(m_dwMasterInstance, pScanBusMismatchJob, SB_PORT_BLOCK_TIMEOUT);
                        if (dwRes != EC_E_NOERROR)
                        {
                            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Cannot block unexpected bus slave! (Result = %s (0x%x))\n", ecatGetText(dwRes), dwRes));
                        }
                        else
                        {
                            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Slave blocked: %d (0x%x) port %d\n", pScanBusMismatchJob->wPrevFixedAddress, pScanBusMismatchJob->wPrevFixedAddress, pScanBusMismatchJob->wPrevPort));
                            m_bDidIssueBlock = EC_TRUE;
                        }
                    }
                }
            } break;
#ifdef INCLUDE_FOE_SUPPORT
        case eMbxTferType_FOE_SEG_DOWNLOAD:
            {
                EC_T_MBXTFER* pMbxTfer = &(TmpJob.JobData.MbxTferJob);
                if (eMbxTferStatus_TferWaitingForContinue == pMbxTfer->eTferStatus)
                {
                    /* read from file to application's buffer */
                    m_SegmentedFoeDownload.dwDataLen =
                        (EC_T_DWORD)OsFread(pMbxTfer->MbxTferDesc.pbyMbxTferDescData, 1, pMbxTfer->MbxData.FoE.dwRequestedBytes, (FILE *)m_SegmentedFoeDownload.hFile);

                    /* queue application's buffer to be transferred to slave */
                    emFoeSegmentedDownloadReq(m_dwMasterInstance, pMbxTfer, 0, EC_NULL, 0, 0, 0, 0);
                }
            } break;
#endif /* INCLUDE_FOE_SUPPORT */
        default:
            continue;            /* next job queued */
        } /* switch job type */

        bProcessed = EC_TRUE;
    } /* while job queued */

#if (defined NO_OS)
    ((CAtEmLogging*)GetLogParms()->pLogContext)->ProcessAllMsgs();
#endif

    return bProcessed;
}

#ifndef EXCLUDE_RAS
/*****************************************************************************/
/**
\brief  AtEmRas Layer Notification function callback.

\return EC_E_NOERROR on success, error code otherwise.
*/
EC_T_DWORD CEmNotification::emRasNotify(
    EC_T_DWORD          dwCode,     /**< [in]   Notification code identifier */
    EC_T_NOTIFYPARMS*   pParms      /**< [in]   Notification data portion */
                                       )
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;

    if ((EC_NULL == pParms))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    switch (dwCode)
    {
    case ATEMRAS_NOTIFY_CONNECTION:         /* GENERIC RAS | 1 */
        {
            ATEMRAS_PT_CONNOTIFYDESC    pConNot = EC_NULL;
            if (sizeof(ATEMRAS_T_CONNOTIFYDESC) > pParms->dwInBufSize)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "RAS Invalid Parameter size for ATEMRAS_NOTIFY_CONNECTION\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }

            pConNot = (ATEMRAS_PT_CONNOTIFYDESC)pParms->pbyInBuf;

            if (pConNot->dwCause == EC_E_NOERROR)
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "RAS Connection changed: Established!\n"));
                m_bRasServerDisconnect = EC_FALSE;
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "RAS Connection changed: Cause: %s (0x%lx)\n",
                    ((EMRAS_E_ERROR == (pConNot->dwCause&EMRAS_E_ERROR)) ? ecatGetText(pConNot->dwCause) : ecatGetText(pConNot->dwCause)), pConNot->dwCause));
                m_bRasServerDisconnect = EC_TRUE;
            }
        } break;
    case ATEMRAS_NOTIFY_REGISTER:           /* GENERIC RAS | 2 */
        {
            /* ATEMRAS_PT_REGNOTIFYDESC pReg = EC_NULL; */

            if (sizeof(ATEMRAS_T_REGNOTIFYDESC) > pParms->dwInBufSize)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "RAS Invalid Parameter size for ATEMRAS_NOTIFY_REGISTER\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }

/*
            pReg = (ATEMRAS_PT_REGNOTIFYDESC) pParms->pbyInBuf;
            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Client registered from cookie 0x%lx Instance 0x%lx Id 0x%lx Result %s (0x%lx)",
                pReg->dwCookie, pReg->dwInstanceId, pReg->dwClientId,
                ((EMRAS_E_ERROR == (pReg->dwResult&EMRAS_E_ERROR))?ecatGetText(EC_LOWORD(pReg->dwResult)):ecatGetText(EC_LOWORD(pReg->dwResult))),
                pReg->dwResult));
*/
        } break;
    case ATEMRAS_NOTIFY_UNREGISTER:         /* GENERIC RAS | 3 */
        {
            /* ATEMRAS_PT_REGNOTIFYDESC pReg = EC_NULL; */

            if (sizeof(ATEMRAS_T_REGNOTIFYDESC) > pParms->dwInBufSize)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "RAS Invalid Parameter size for ATEMRAS_NOTIFY_UNREGISTER\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }

/*
            pReg = (ATEMRAS_PT_REGNOTIFYDESC) pParms->pbyInBuf;
           EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Client un-registered from cookie 0x%lx Instance 0x%lx Id 0x%lx Result %s (0x%lx)",
                pReg->dwCookie, pReg->dwInstanceId, pReg->dwClientId,
                ((EMRAS_E_ERROR == (pReg->dwResult&EMRAS_E_ERROR))?ecatGetText(EC_LOWORD(pReg->dwResult)):ecatGetText(EC_LOWORD(pReg->dwResult))),
                pReg->dwResult));
*/
        } break;
    case ATEMRAS_NOTIFY_MARSHALERROR:       /* ERROR RAS | 1 */
        {
            ATEMRAS_PT_MARSHALERRORDESC     pMarshNot = EC_NULL;
            if (sizeof(ATEMRAS_T_MARSHALERRORDESC) > pParms->dwInBufSize)
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid Parameter size for ATEMRAS_NOTIFY_MARSHALERROR\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }

            pMarshNot = (ATEMRAS_PT_MARSHALERRORDESC)pParms->pbyInBuf;
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Marshaling error! Cookie: 0x%lx\n", pMarshNot->dwCookie));
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Command: 0x%lx, Cause: %s (0x%lx)\n",
                pMarshNot->dwCommandCode,
                ((EMRAS_E_ERROR == (pMarshNot->dwCause&EMRAS_E_ERROR)) ? ecatGetText(pMarshNot->dwCause) : ecatGetText(pMarshNot->dwCause)),
                pMarshNot->dwCause));
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Protocol Header: 0x%lx\n", pMarshNot->dwLenStatCmd));
        } break;
    case ATEMRAS_NOTIFY_ACKERROR:           /* ERROR RAS | 2 */
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Acknowledge error! Error: %s (0x%lx)\n", ecatGetText(EC_GETDWORD(pParms->pbyInBuf)), EC_GETDWORD(pParms->pbyInBuf)));
        } break;
    case ATEMRAS_NOTIFY_NONOTIFYMEMORY:     /* ERROR RAS | 3 */
        {
            ATEMRAS_T_NONOTIFYMEMORYDESC NoNotifyMemoryDesc;
            ATEMRAS_PT_NONOTIFYMEMORYDESC pNoNotifyMemoryDescSerialized = (ATEMRAS_PT_NONOTIFYMEMORYDESC)pParms->pbyInBuf;
            OsDbgAssert(pParms->dwInBufSize >= sizeof(ATEMRAS_T_NONOTIFYMEMORYDESC));
            NoNotifyMemoryDesc.dwCookie = EC_GET_FRM_DWORD(&pNoNotifyMemoryDescSerialized->dwCookie);
            NoNotifyMemoryDesc.dwCode = EC_GET_FRM_DWORD(&pNoNotifyMemoryDescSerialized->dwCode);

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Out of notification memory! %s (0x%08X), cookie 0x%lx.\n",
                ecatGetNotifyText(NoNotifyMemoryDesc.dwCode), NoNotifyMemoryDesc.dwCode, NoNotifyMemoryDesc.dwCookie));
        } break;
    case ATEMRAS_NOTIFY_STDNOTIFYMEMORYSMALL:   /* ERROR RAS | 4 */
        {
            ATEMRAS_T_NONOTIFYMEMORYDESC NoNotifyMemoryDesc;
            ATEMRAS_PT_NONOTIFYMEMORYDESC pNoNotifyMemoryDescSerialized = (ATEMRAS_PT_NONOTIFYMEMORYDESC)pParms->pbyInBuf;
            OsDbgAssert(pParms->dwInBufSize >= sizeof(ATEMRAS_T_NONOTIFYMEMORYDESC));
            NoNotifyMemoryDesc.dwCookie = EC_GET_FRM_DWORD(&pNoNotifyMemoryDescSerialized->dwCookie);
            NoNotifyMemoryDesc.dwCode = EC_GET_FRM_DWORD(&pNoNotifyMemoryDescSerialized->dwCode);

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Out of non-mailbox notification memory! %s (0x%08X), cookie 0x%lx.\n",
                ecatGetNotifyText(NoNotifyMemoryDesc.dwCode), NoNotifyMemoryDesc.dwCode, NoNotifyMemoryDesc.dwCookie));
        } break;
    case ATEMRAS_NOTIFY_MBXNOTIFYMEMORYSMALL:   /* ERROR RAS | 5 */
        {
            ATEMRAS_T_NONOTIFYMEMORYDESC NoNotifyMemoryDesc;
            ATEMRAS_PT_NONOTIFYMEMORYDESC pNoNotifyMemoryDescSerialized = (ATEMRAS_PT_NONOTIFYMEMORYDESC)pParms->pbyInBuf;
            OsDbgAssert(pParms->dwInBufSize >= sizeof(ATEMRAS_T_NONOTIFYMEMORYDESC));
            NoNotifyMemoryDesc.dwCookie = EC_GET_FRM_DWORD(&pNoNotifyMemoryDescSerialized->dwCookie);
            NoNotifyMemoryDesc.dwCode = EC_GET_FRM_DWORD(&pNoNotifyMemoryDescSerialized->dwCode);

            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Out of mailbox notification memory! %s (0x%08X), cookie 0x%lx.\n",
                ecatGetNotifyText(NoNotifyMemoryDesc.dwCode), NoNotifyMemoryDesc.dwCode, NoNotifyMemoryDesc.dwCookie));
        } break;
    default:
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "emRasNotify: name = %s code = 0x%x\n", ecatGetNotifyText(dwCode), dwCode));
        } break;
    }

    dwRetVal = EC_E_NOERROR;
Exit:
    return dwRetVal;
}
#endif /* !EXCLUDE_RAS */

/*****************************************************************************/
/** \brief  C Wrapper function to ecatNotify
*
* \return  Status value.
*/
EC_T_DWORD CEmNotification::NotifyWrapper(
    EC_T_DWORD         dwCode,  /**< [in]   Notifcation code */
    EC_T_NOTIFYPARMS*  pParms   /**< [in]   Notification parameters */
                                         )
{
    printf("Running NotifyWrapper\n");
    EC_T_DWORD       dwRetVal        = EC_E_NOERROR;
    CEmNotification* pNotifyInstance = EC_NULL;

    if ((EC_NULL == pParms) || (EC_NULL == pParms->pCallerData))
    {
        fprintf(stderr, "NotifyWrapper: Invalid parameters\n");
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    pNotifyInstance = (CEmNotification*)pParms->pCallerData;

    dwRetVal = pNotifyInstance->ecatNotify(dwCode, pParms);
    return dwRetVal;
Exit:
    return dwRetVal;
}

/*****************************************************************************/
/**
\brief  Reset Error counters.
*/
EC_T_VOID CEmNotification::ResetErrorCounters(EC_T_VOID)
{
    OsMemset(m_adwErrorCounter, 0, sizeof(m_adwErrorCounter));
}

/*****************************************************************************/
/**
\brief  Increment Pointer in Queue.

\return New Pointer.
*/
PT_SLAVEJOBS CEmNotification::IncrementJobPtr(PT_SLAVEJOBS pPtr)
{
    PT_SLAVEJOBS pRet = EC_NULL;

    if (EC_NULL == pPtr)
    {
        goto Exit;
    }

    if ((EC_T_PVOID)(pPtr + 1) > (EC_T_PVOID)(&m_oSlaveJobQueue.Jobs[JOB_QUEUE_LENGTH - 1]))
    {
        /* wrap */
        pRet = &(m_oSlaveJobQueue.Jobs[0]);
    }
    else
    {
        pRet = (pPtr + 1);
    }
Exit:
    return pRet;
}

/*****************************************************************************/
/**
\brief  EnqueueJob.

Enqueue new Job to queue.
\return EC_TRUE on success, EC_FALSE otherwise.
*/
EC_T_BOOL CEmNotification::EnqueueJob(
    PT_SLAVEJOBS pJob,
    EC_T_DWORD dwCode,
    EC_T_VOID* pSrc,
    EC_T_DWORD dwSize
                                     )
{
    EC_T_BOOL    bRet      = EC_FALSE;
    PT_SLAVEJOBS pNewWrite = EC_NULL;

    pNewWrite = IncrementJobPtr(m_oSlaveJobQueue.pWrite);

    if (pNewWrite == m_oSlaveJobQueue.pRead)
    {
        /* no more space in queue */
        goto Exit;
    }

    /* store job */
    pJob->dwCode = dwCode;
    OsMemcpy(&pJob->JobData, pSrc, dwSize);
    OsMemcpy(m_oSlaveJobQueue.pWrite, pJob, sizeof(T_SLAVEJOBS));

    /* increment write pointer */
    m_oSlaveJobQueue.pWrite = pNewWrite;

    bRet = EC_TRUE;
Exit:
    if (!bRet)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "ERROR: Unable to enqueue %s job! Missing calls to ProcessNotificationJobs or queue too small.\n", ecatGetNotifyText(dwCode)));
    }

    return bRet;
}

/*****************************************************************************/
/**
\brief  DequeueJob.

Dequeue next job if possible.
\return EC_TRUE on success, EC_FALSE otherwise.
*/
EC_T_BOOL CEmNotification::DequeueJob(
    PT_SLAVEJOBS    pJob    /**< [out]  Dequeued Job Object */
                                     )
{
    EC_T_BOOL bRet = EC_FALSE;

    if ((EC_NULL == pJob) || (m_oSlaveJobQueue.pRead == m_oSlaveJobQueue.pWrite))
    {
        goto Exit;
    }

    OsMemcpy(pJob, m_oSlaveJobQueue.pRead, sizeof(T_SLAVEJOBS));

    /* Increment Read Pointer */
    m_oSlaveJobQueue.pRead = IncrementJobPtr(m_oSlaveJobQueue.pRead);

    bRet = EC_TRUE;
Exit:
    return bRet;
}

EC_T_DWORD CEmNotification::SetNotificationEnabled(EC_T_DWORD dwCode, EC_T_DWORD dwEnabled)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_DWORD dwRes = EC_E_NOERROR;
    EC_T_IOCTLPARMS oIoCtlParms;
    EC_T_SET_NOTIFICATION_ENABLED_PARMS oNotificationCtlParms;
    OsMemset(&oIoCtlParms, 0, sizeof(EC_T_IOCTLPARMS));
    OsMemset(&oNotificationCtlParms, 0, sizeof(EC_T_SET_NOTIFICATION_ENABLED_PARMS));

    oIoCtlParms.pbyInBuf = (EC_T_BYTE*)&oNotificationCtlParms;
    oIoCtlParms.dwInBufSize = sizeof(EC_T_SET_NOTIFICATION_ENABLED_PARMS);

    oNotificationCtlParms.dwCode = dwCode;
    oNotificationCtlParms.dwEnabled = dwEnabled;

    /* generally enable notification */
    if (EC_NOTIFICATION_ENABLED == dwEnabled)
    {
        oNotificationCtlParms.dwClientId = 0;
        dwRes = emIoControl(m_dwMasterInstance, EC_IOCTL_SET_NOTIFICATION_ENABLED, &oIoCtlParms);
        if (EC_E_NOERROR != dwRes)
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "SetNotificationEnabled(...): %s (0x%x)", ecatGetText(dwRes), dwRes));
            dwRetVal = dwRes;
            goto Exit;
        }
    }

    /* enable notification for this instance */
    oNotificationCtlParms.dwClientId = m_dwClientID;
    dwRes = emIoControl(m_dwMasterInstance, EC_IOCTL_SET_NOTIFICATION_ENABLED, &oIoCtlParms);
    if (EC_E_NOERROR != dwRes)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "SetNotificationEnabled(...): %s (0x%x)", ecatGetText(dwRes), dwRes));
        dwRetVal = dwRes;
        goto Exit;
    }
    dwRetVal = EC_E_NOERROR;
Exit:
    return dwRetVal;
}

/*-END OF SOURCE FILE--------------------------------------------------------*/
