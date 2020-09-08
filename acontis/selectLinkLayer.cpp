/*-----------------------------------------------------------------------------
 * selectLinkLayer.cpp
 * Copyright                acontis technologies GmbH, Weingarten, Germany
 * Response                 Paul Bussmann
 * Description              EC-Master link layer selection
 *---------------------------------------------------------------------------*/

/*-INCLUDES------------------------------------------------------------------*/
#include "Logging.h"
#include "selectLinkLayer.h"

/*-DEFINES-------------------------------------------------------------------*/

/*-FUNCTION-DEFINITIONS------------------------------------------------------*/
/********************************************************************************/
/** \brief  Parse for the ip-address and return an byte array (EC_T_BYTE[4])
*
* \return  EC_TRUE in case of success, EC_FALSE in case of an invalid parameter or
*          SYNTAX_ERROR (-2) .
*/
EC_T_BOOL ParseIpAddress
(
    EC_T_CHAR* ptcWord,
    EC_T_BYTE* pbyIpAddress)
{
    EC_T_CHAR* ptcTmp   = EC_NULL;
    EC_T_INT   nCnt     = 0;
    EC_T_BOOL bRetVal = EC_TRUE;

    if(EC_NULL == pbyIpAddress )
    {
        bRetVal = EC_FALSE;
        goto Exit;
    }

    /* Get IP address */
    ptcTmp = ptcWord;
    for (nCnt = 0; nCnt < 4; nCnt++)
    {
        if (ptcTmp == EC_NULL)
        {
            bRetVal = EC_FALSE;
            goto Exit;
        }
        pbyIpAddress[nCnt] = (EC_T_BYTE)OsStrtol(ptcTmp, EC_NULL, 10);
        if(nCnt < 2)
        {
            ptcTmp = OsStrtok( EC_NULL, ".");
        }
        else if(nCnt < 3)
        {
            ptcTmp = OsStrtok( EC_NULL, " ");
        }
    }

    Exit:

    return bRetVal;
}

/********************************************************************************/
/** Parse next command line argument
*
* Return: pointer to the next argument.
*/
EC_T_CHAR* GetNextWord(EC_T_CHAR **ppCmdLine, EC_T_CHAR *pStorage)
{
    EC_T_CHAR *pWord;

    EC_UNREFPARM(ppCmdLine);
    EC_UNREFPARM(pStorage);

    pWord = (EC_T_CHAR *)OsStrtok(NULL, " ");

    return pWord;
}

/***************************************************************************************************/
/**
\brief  Parses string parameter value from the command line.

\return EC_TRUE if successfully parsed, EC_FALSE on syntax errors.
*/
static EC_T_BOOL ParseString(
    EC_T_CHAR**     ptcWord,
    EC_T_CHAR**     lpCmdLine,
    EC_T_CHAR*      tcStorage)
{
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if (((*ptcWord) == EC_NULL) || (OsStrncmp( (*ptcWord), "-", 1) == 0))
    {
        return EC_FALSE;
    }

    return EC_TRUE;
}

/***************************************************************************************************/
/**
\brief  Parses EC_T_DWORD parameter value from the command line.

\return EC_TRUE if successfully parsed, EC_FALSE on syntax errors.
*/
static EC_T_BOOL ParseDword(
    EC_T_CHAR**     ptcWord,
    EC_T_CHAR**     lpCmdLine,
    EC_T_CHAR*      tcStorage,
    EC_T_DWORD*     pdwValue)
{
    if ( !ParseString(ptcWord, lpCmdLine, tcStorage) )
    {
        return EC_FALSE;
    }
    *pdwValue = (EC_T_DWORD)OsStrtol((*ptcWord), NULL, 0);

    return EC_TRUE;
}

/***************************************************************************************************/
/**
\brief  Parses EC_T_LINKMODE parameter value from the command line.

\return EC_TRUE if successfully parsed, EC_FALSE on syntax errors.
*/
EC_T_BOOL ParseLinkMode(
    EC_T_CHAR**     ptcWord,
    EC_T_CHAR**     lpCmdLine,
    EC_T_CHAR*      tcStorage,
    EC_T_LINKMODE*  peLinkMode)
{
    EC_T_DWORD dwMode = 0;
    EC_T_BOOL bRes = EC_FALSE;

    if (ParseDword(ptcWord, lpCmdLine, tcStorage, &dwMode))
    {
        if (dwMode == 0)
        {
            *peLinkMode = EcLinkMode_INTERRUPT;
            bRes = EC_TRUE;
        }
        else if (dwMode == 1)
        {
            *peLinkMode = EcLinkMode_POLLING;
            bRes = EC_TRUE;
        }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid Link Layer Mode (mode == %d)\n", dwMode));
        }
    }
    return bRes;
}

/***************************************************************************************************/
/**
\brief  Fill common link layer parameters
*/
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

#if (defined INCLUDE_EMLLANTAIOS)
/***************************************************************************************************/
/**
\brief  Try to create ANTAIOS link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineAntaios(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_ANTAIOS* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-antaios") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_ANTAIOS*)OsMalloc(sizeof(EC_T_LINK_PARMS_ANTAIOS));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }

    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_ANTAIOS));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_ANTAIOS, sizeof(EC_T_LINK_PARMS_ANTAIOS), EC_LINK_PARMS_IDENT_ANTAIOS, 2, EcLinkMode_POLLING);

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLANTAIOS */

#if (defined INCLUDE_EMLLALTERATSE)

/***************************************************************************************************/
/**
\brief  Try to create Altera TSE link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineAlteraTse(EC_T_CHAR** ptcWord,
    EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage,
    EC_T_BOOL*  pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms
)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_ALTERATSE* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-alteratse") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_ALTERATSE*)OsMalloc(sizeof(EC_T_LINK_PARMS_ALTERATSE));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_ALTERATSE));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_ALTERATSE,
        sizeof(EC_T_LINK_PARMS_ALTERATSE), EC_LINK_PARMS_IDENT_ALTERATSE, 1, EcLinkMode_POLLING);

    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode)
        )
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    if (pLinkParmsAdapter->linkParms.dwInstance <= 1 || pLinkParmsAdapter->linkParms.dwInstance > 2)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Port number must be between 1 and 2\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    pLinkParmsAdapter->abyMac[0] = 0x01;
    pLinkParmsAdapter->abyMac[1] = 0xB4;
    pLinkParmsAdapter->abyMac[2] = 0xC3;
    pLinkParmsAdapter->abyMac[3] = 0xDD;
    pLinkParmsAdapter->abyMac[4] = 0xEE;
    pLinkParmsAdapter->abyMac[5] = 0xFF;

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLALTERATSE */

#if (defined INCLUDE_EMLLCCAT)
/***************************************************************************************************/
/**
\brief  Try to create CCAT link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineCCAT(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                 EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_CCAT* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-ccat") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_CCAT*)OsMalloc(sizeof(EC_T_LINK_PARMS_CCAT));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_CCAT));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_CCAT, sizeof(EC_T_LINK_PARMS_CCAT), EC_LINK_PARMS_IDENT_CCAT, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* parse optional parameters  */
    pLinkParmsAdapter->eCcatType = eCCAT_PCI;

    /* get Interface */
    if (!ParseString(ptcWord, lpCmdLine, tcStorage))
    {
        if (EC_NULL != pbGetNextWord)
        {
            *pbGetNextWord = EC_FALSE;
        }
    }
    else
    {
        if (0 == OsStricmp((*ptcWord), "eim"))
        {
            pLinkParmsAdapter->eCcatType              = eCCAT_EIM;
            pLinkParmsAdapter->qwCcatBase             = 0xf0000000;
            pLinkParmsAdapter->dwCcatSize             = 0x02000000;
            pLinkParmsAdapter->dwRxBufferCnt          = 20;
            pLinkParmsAdapter->dwTxBufferCnt          = 20;
        }
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLCCAT */

#if (defined INCLUDE_EMLLCPSW)
/***************************************************************************************************/
/**
\brief  Parses EC_T_CPSW_TYPE parameter value from the command line.

\return EC_TRUE if successfully parsed, EC_FALSE on syntax errors.
*/
EC_T_BOOL ParseCPSWType(
    EC_T_CHAR**     ptcWord,
    EC_T_CHAR**     lpCmdLine,
    EC_T_CHAR*      tcStorage,
    EC_T_CPSW_TYPE*  peType)
{
    EC_T_BOOL bRes = EC_FALSE;

    if (ParseString(ptcWord, lpCmdLine, tcStorage))
    {
        bRes = EC_TRUE;
        if (0 == OsStricmp((*ptcWord), "am33XX"))       { *peType = eCPSW_AM33XX; }
        else if (0 == OsStricmp((*ptcWord), "am437X"))  { *peType = eCPSW_AM437X; }
        else if (0 == OsStricmp((*ptcWord), "am57X"))   { *peType = eCPSW_AM57X;  }
        else if (0 == OsStricmp((*ptcWord), "am387X"))  { *peType = eCPSW_AM387X; }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid CPSW type (type == %s)\n", ptcWord));
            bRes = EC_FALSE;
        }
    }
    return bRes;
}

/***************************************************************************************************/
/**
\brief  Try to create CPSW link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineCPSW(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                 EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_CPSW* pLinkParmsAdapter = EC_NULL;
EC_T_DWORD dwPhyConnection = 0;

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-cpsw") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_CPSW*)OsMalloc(sizeof(EC_T_LINK_PARMS_CPSW));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_CPSW));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_CPSW, sizeof(EC_T_LINK_PARMS_CPSW), EC_LINK_PARMS_IDENT_CPSW, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: port (instance), mode, port priority, Master flag*/
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode)
        || !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwPortPrio)
        || /* parse bMaster */ !ParseString(ptcWord, lpCmdLine, tcStorage) )
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    pLinkParmsAdapter->bMaster = ((*ptcWord)[0] == 'm');

    /* set default parameters */
    pLinkParmsAdapter->bNotUseDmaBuffers = EC_TRUE;
    pLinkParmsAdapter->ePhyInterface = ePHY_GMII;
    pLinkParmsAdapter->eCpswType = eCPSW_AM33XX;
    pLinkParmsAdapter->dwPhyAddr = pLinkParmsAdapter->linkParms.dwInstance - 1; /* 0 -> Port1, 1 -> Port2 */

    /* parse optional parameters  */

    /* get reference board */
    if (!ParseString(ptcWord, lpCmdLine, tcStorage))
    {
        if (EC_NULL != pbGetNextWord)
        {
            *pbGetNextWord = EC_FALSE;
        }
    }
    else
    {
        if (0 == OsStricmp((*ptcWord), "bone"))             { pLinkParmsAdapter->eCpswType = eCPSW_AM33XX; }
        else if (0 == OsStricmp((*ptcWord), "am437x-idk"))  { pLinkParmsAdapter->eCpswType = eCPSW_AM437X; }
        else if (0 == OsStricmp((*ptcWord), "am572x-idk"))  { pLinkParmsAdapter->eCpswType = eCPSW_AM57X; pLinkParmsAdapter->ePhyInterface = ePHY_RGMII; }
        else if (0 == OsStricmp((*ptcWord), "387X_evm"))    { pLinkParmsAdapter->eCpswType = eCPSW_AM387X; }
        else if (0 == OsStricmp((*ptcWord), "custom"))
        {
            EC_T_DWORD dwNotUseDmaBuffers = 0;
            /* parse CpswType, PHY address, PHY connection type, use DMA */
            if (!ParseCPSWType(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->eCpswType)
                || !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwPhyAddr)
                || !ParseDword(ptcWord, lpCmdLine, tcStorage, &dwPhyConnection)
                || !ParseDword(ptcWord, lpCmdLine, tcStorage, &dwNotUseDmaBuffers))
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "error parsing CPSW parameters for custom board\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
            pLinkParmsAdapter->ePhyInterface = (dwPhyConnection == 1) ? ePHY_RGMII : ePHY_GMII;
            pLinkParmsAdapter->bNotUseDmaBuffers = (dwNotUseDmaBuffers == 0) ? EC_FALSE : EC_TRUE;
        }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid RefBoard value\n"));
            dwRetVal = EC_E_INVALIDPARM;
            goto Exit;
        }
    }

    /* verify parameter values */
    if (pLinkParmsAdapter->linkParms.dwInstance < 1 || pLinkParmsAdapter->linkParms.dwInstance > 2)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Port number must be 1 or 2\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    if (pLinkParmsAdapter->dwPortPrio != 0 && pLinkParmsAdapter->dwPortPrio != 1)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Port priority must be 0 or 1\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    pLinkParmsAdapter->bPhyRestartAutoNegotiation = EC_TRUE;

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLCPSW */

#if (defined INCLUDE_EMLLDUMMY)
/***************************************************************************************************/
/**
\brief  Try to create dummy link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineDummy(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_DUMMY* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-dummy") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_DUMMY*)OsMalloc(sizeof(EC_T_LINK_PARMS_DUMMY));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_DUMMY));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_DUMMY, sizeof(EC_T_LINK_PARMS_DUMMY), EC_LINK_PARMS_IDENT_DUMMY, 1, EcLinkMode_POLLING);

    /* get Instance and Mode */
    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLDUMMY */

#if (defined INCLUDE_EMLLDW3504)
/***************************************************************************************************/
/**
\brief  Try to create DW3504 link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineDW3504(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_DW3504* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-DW3504") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_DW3504*)OsMalloc(sizeof(EC_T_LINK_PARMS_DW3504));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_DW3504));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_DW3504, sizeof(EC_T_LINK_PARMS_DW3504), EC_LINK_PARMS_IDENT_DW3504, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters */
    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* get reference board */
    if (!ParseString(ptcWord, lpCmdLine, tcStorage))
    {
        if (EC_NULL != pbGetNextWord)
        {
            *pbGetNextWord = EC_FALSE;
        }

        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    else
    {
        if (0 == OsStricmp((*ptcWord), "socrates"))
        {
            pLinkParmsAdapter->eDW3504Type = eDW3504_CycloneV;
            pLinkParmsAdapter->ePhyInterface = ePHY_MII;
            if (pLinkParmsAdapter->linkParms.dwInstance == 1)
            {
                pLinkParmsAdapter->dwRegisterBasePhys = 0xFF700000;
                pLinkParmsAdapter->dwPhyAddr = 0;
            }
            else if (pLinkParmsAdapter->linkParms.dwInstance == 2)
            {
                pLinkParmsAdapter->dwRegisterBasePhys = 0xFF702000;
                pLinkParmsAdapter->dwPhyAddr = 1;
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
        }
        else if (0 == OsStricmp((*ptcWord), "rd55up06"))
        {
            pLinkParmsAdapter->eDW3504Type = eDW3504_CycloneV;
            pLinkParmsAdapter->ePhyInterface = ePHY_SGMII;
            if (pLinkParmsAdapter->linkParms.dwInstance == 2)
            {
                pLinkParmsAdapter->dwRegisterBasePhys = 0xFF702000;
                pLinkParmsAdapter->dwPhyAddr = 0;
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 2\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
        }
        else if (0 == OsStricmp((*ptcWord), "r12ccpu"))
        {
            pLinkParmsAdapter->eDW3504Type = eDW3504_CycloneV;
            if (pLinkParmsAdapter->linkParms.dwInstance == 1)
            {
                pLinkParmsAdapter->ePhyInterface = ePHY_GMII;
                pLinkParmsAdapter->dwRegisterBasePhys = 0xFF700000;
                pLinkParmsAdapter->dwPhyAddr = 0;
            }
            else if (pLinkParmsAdapter->linkParms.dwInstance == 2)
            {
                pLinkParmsAdapter->ePhyInterface = ePHY_SGMII;
                pLinkParmsAdapter->dwRegisterBasePhys = 0xFF702000;
                pLinkParmsAdapter->dwPhyAddr = 0;
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
        }
        else if (0 == OsStricmp((*ptcWord), "lces1"))
        {
            pLinkParmsAdapter->eDW3504Type = eDW3504_LCES1;
            pLinkParmsAdapter->ePhyInterface = ePHY_MII;
            if (pLinkParmsAdapter->linkParms.dwInstance == 1)
            {
                pLinkParmsAdapter->dwRegisterBasePhys = 0x44000000;
                pLinkParmsAdapter->dwPhyAddr = 0;
            }
            else if (pLinkParmsAdapter->linkParms.dwInstance == 2)
            {
                pLinkParmsAdapter->dwRegisterBasePhys = 0x44002000;
                pLinkParmsAdapter->dwPhyAddr = 4;
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
        }
        else if (0 == OsStricmp((*ptcWord), "rzn1"))
        {
            pLinkParmsAdapter->eDW3504Type = eDW3504_RZN1;
            pLinkParmsAdapter->ePhyInterface = ePHY_RGMII;
            if (pLinkParmsAdapter->linkParms.dwInstance == 1)
            {
                pLinkParmsAdapter->dwRegisterBasePhys = 0x44000000;
                pLinkParmsAdapter->dwPhyAddr = 8;
            }
            else if (pLinkParmsAdapter->linkParms.dwInstance == 2)
            {
                pLinkParmsAdapter->dwRegisterBasePhys = 0x44002000;
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
        }
        else if (0 == OsStricmp((*ptcWord), "custom"))
        {
            /* get DW3504 type */
            (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
            if (0 == OsStricmp((*ptcWord), "cycloneV"))
            {
                pLinkParmsAdapter->eDW3504Type = eDW3504_CycloneV;
                if (pLinkParmsAdapter->linkParms.dwInstance == 1)
                {
                    pLinkParmsAdapter->dwRegisterBasePhys = 0xFF700000;
                }
                else if (pLinkParmsAdapter->linkParms.dwInstance == 2)
                {
                    pLinkParmsAdapter->dwRegisterBasePhys = 0xFF702000;
                }
                else
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
                    dwRetVal = EC_E_INVALIDPARM;
                    goto Exit;
                }
            }
            else if (0 == OsStricmp((*ptcWord), "lces1"))
            {
                pLinkParmsAdapter->eDW3504Type = eDW3504_LCES1;
                if (pLinkParmsAdapter->linkParms.dwInstance == 1)
                {
                    pLinkParmsAdapter->dwRegisterBasePhys = 0x44000000;
                }
                else if (pLinkParmsAdapter->linkParms.dwInstance == 2)
                {
                    pLinkParmsAdapter->dwRegisterBasePhys = 0x44002000;
                }
                else
                {
                    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
                    dwRetVal = EC_E_INVALIDPARM;
                    goto Exit;
                }
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid eDW3504Type value\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
            /* get PHY interface */
            (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
            if (0 == OsStricmp((*ptcWord), "fixed")) pLinkParmsAdapter->ePhyInterface = ePHY_FIXED_LINK;
            else if (0 == OsStricmp((*ptcWord), "mii"))   pLinkParmsAdapter->ePhyInterface = ePHY_MII;
            else if (0 == OsStricmp((*ptcWord), "rmii"))  pLinkParmsAdapter->ePhyInterface = ePHY_RMII;
            else if (0 == OsStricmp((*ptcWord), "gmii"))  pLinkParmsAdapter->ePhyInterface = ePHY_GMII;
            else if (0 == OsStricmp((*ptcWord), "sgmii")) pLinkParmsAdapter->ePhyInterface = ePHY_SGMII;
            else if (0 == OsStricmp((*ptcWord), "rgmii")) pLinkParmsAdapter->ePhyInterface = ePHY_RGMII;
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid PhyInterface value\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
            /* get PHY address */
            if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwPhyAddr))
            {
                pLinkParmsAdapter->dwPhyAddr = 0;
            }
        }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid RefBoard value\n"));
            dwRetVal = EC_E_INVALIDPARM;
            goto Exit;
        }
    }
    /* set default values */
#if (defined __arm__) || (defined __ICCARM__)
    pLinkParmsAdapter->bNotUseDmaBuffers = EC_TRUE;
#else
    pLinkParmsAdapter->bNotUseDmaBuffers = EC_FALSE;
#endif

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLDW3504 */

#if (defined INCLUDE_EMLLEG20T)
/***************************************************************************************************/
/**
\brief  Try to create EG20T link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineEG20T(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_EG20T* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-eg20t") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_EG20T*)OsMalloc(sizeof(EC_T_LINK_PARMS_EG20T));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_CCAT));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_EG20T, sizeof(EC_T_LINK_PARMS_EG20T), EC_LINK_PARMS_IDENT_EG20T, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLEG20T */

#if (defined INCLUDE_EMLLEMAC)
/***************************************************************************************************/
/**
\brief  Try to create EMAC link layer parameters according current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineEMAC(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                 EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_EMAC* pLinkParmsAdapter = EC_NULL;

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-emac") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_EMAC*)OsMalloc(sizeof(EC_T_LINK_PARMS_EMAC));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_EMAC));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_EMAC, sizeof(EC_T_LINK_PARMS_EMAC), EC_LINK_PARMS_IDENT_EMAC, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* default parameters */
    pLinkParmsAdapter->dwRegisterBase = 0x86000000;

    /* get reference board */
    if (!ParseString(ptcWord, lpCmdLine, tcStorage))
    {
        if (EC_NULL != pbGetNextWord)
        {
            *pbGetNextWord = EC_FALSE;
        }
    }
    else
    {
        if(0 == OsStricmp((*ptcWord), "MC2002E"))
        {
            pLinkParmsAdapter->dwRegisterBase   = 0x80000000;
            pLinkParmsAdapter->dwRegisterLength = 0x2000;
            pLinkParmsAdapter->bNotUseDmaBuffers = EC_TRUE;
        }
        else if (0 == OsStricmp((*ptcWord), "custom"))
        {
            /* get register base */
            (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
            if (EC_NULL != ptcWord)
            {
                pLinkParmsAdapter->dwRegisterBase = (EC_T_DWORD)OsStrtoul((*ptcWord),0,16);
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Missing parameter RegisterBase\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
            /* get register length */
            (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
            if (EC_NULL != ptcWord)
            {
                pLinkParmsAdapter->dwRegisterLength = (EC_T_DWORD)OsStrtoul((*ptcWord),0,16);
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Missing parameter RegisterLength\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
            /* use dma buffers */
            (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
            if (EC_NULL != ptcWord)
            {
                pLinkParmsAdapter->bNotUseDmaBuffers = (EC_T_BOOL)OsStrtoul((*ptcWord),0,16);
            }
            else
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Missing parameter NotUseDmaBuffers\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
        }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid RefBoard value\n"));
            dwRetVal = EC_E_INVALIDPARM;
            goto Exit;
        }
    }
    pLinkParmsAdapter->dwPhyAddr        = 0;
    pLinkParmsAdapter->dwRxInterrupt    = 1; /* from xparameters.h */

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLEMAC */

#if (defined INCLUDE_EMLLETSEC)

#if (EC_ARCH == EC_ARCH_ARM)
#if (defined EC_VERSION_VXWORKS)
#define TSEC_CFG_TWRLS1021A    /* Freescale TWR-L1021A-PB board */
#elif (defined EC_VERSION_QNX)
#define TSEC_CFG_TQMLS_LS102XA /* Freescale TWR-L1021A-PB board */
#else
#if (!defined TSEC_CFG_TWRLS1021A)
#define TSEC_CFG_TWRLS1021A    /* Freescale TWR-L1021A-PB board */
#endif
#endif
#elif (EC_ARCH == EC_ARCH_PPC)
#define TSEC_CFG_P2020RDB      /* Freescale P2020RDB reference board */
#endif

/* Uncomment if needed */
/* Instron "MPC8548 MiniTower" board */
/* #define TSEC_CFG_ISTMPC8548 */
/* Freescale TWR-P1025 board */
/* #define TSEC_CFG_TWRP1025 */
/* XJ Electric Corp */
/* #define TSEC_CFG_XJ_EPU20C */

#if (defined VXWORKS) && ((defined TSEC_CFG_P2020RDB) || (defined TSEC_CFG_ISTMPC8548) || (defined TSEC_CFG_XJ_EPU20C))
extern "C" {
UINT32 sysGetPeripheralBase (void); /* from sysLib.c */
#  include <hwif/intCtlr/vxbEpicIntCtlr.h>
}
#endif

/***************************************************************************************************/
/**
\brief  Try to create ETSEC link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineETSEC(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                  EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_ETSEC* pLinkParmsAdapter = EC_NULL;
EC_T_DWORD dwCcsrbar = 0x0;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-fsletsec") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_ETSEC*)OsMalloc(sizeof(EC_T_LINK_PARMS_ETSEC));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_ETSEC));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_ETSEC, sizeof(EC_T_LINK_PARMS_ETSEC), EC_LINK_PARMS_IDENT_ETSEC, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
#ifdef TSEC_CFG_P2020RDB

    /*
     * The following configuration is for Freescale P2020RDB board
     * (CPU P2020E, VxWorks 6.8 PPC / Linux 3.0.9-PREEMPT PPC)
     */

    if (pLinkParmsAdapter->linkParms.dwInstance > 3)
    {
       EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Device unit must be <= 3! There are only 3 eTSEC's on P2020RDB.\n"));
       dwRetVal = EC_E_INVALIDPARM;
       goto Exit;
    }

    {
       EC_T_BYTE abyStationAddress[] = {0x00, 0x04, 0x9F, 0x01, 0x79, 0x00};
       memcpy(pLinkParmsAdapter->abyStationAddress, abyStationAddress, 6);
    }

    // don't change fundamental settings in the ETSEC like endianess
    pLinkParmsAdapter->bMaster = EC_FALSE;
    pLinkParmsAdapter->eETSECType = eETSEC_P2020RDB;

#if (defined VXWORKS)
    dwCcsrbar = sysGetPeripheralBase();
#elif defined LINUX
    dwCcsrbar = 0xffe00000; /* from p2020si.dtsi */

    /* Get interrupt number from Nth eTSEC device in PowerPC device tree */
    pLinkParmsAdapter->dwRxInterrupt = pLinkParmsAdapter->linkParms.dwInstance - 1;
#endif

    pLinkParmsAdapter->dwPhyMdioBase = dwCcsrbar + 0x24000; /* eTSEC 1 */
    pLinkParmsAdapter->dwTbiPhyAddr = 16; /* Dummy address assigned to internal TBI PHY */
    pLinkParmsAdapter->oMiiBusMtx = EC_NULL; /* LinkOsCreateLock(eLockType_DEFAULT); */

    if (pLinkParmsAdapter->linkParms.dwInstance == 1) /* eTSEC1 (4 x 1Gb switchports) */
    {
       pLinkParmsAdapter->abyStationAddress[5] = 0x81;
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x24000;
       pLinkParmsAdapter->dwPhyAddr = ETSEC_FIXED_LINK; /* RGMII, No Phy, Switchport */
       pLinkParmsAdapter->dwFixedLinkVal = ETSEC_LINKFLAG_1000baseT_Full | ETSEC_LINKFLAG_LINKOK;
#if (defined VXWORKS)
       pLinkParmsAdapter->dwRxInterrupt = EPIC_TSEC1RX_INT_VEC; /* from vxbEpicIntCtlr.h */
#endif
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 2) /* eTSEC2 */
    {
       pLinkParmsAdapter->abyStationAddress[5] = 0x82;
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x25000;
       pLinkParmsAdapter->dwPhyAddr = 0; /* SGMII Phy on addr. 0 (from P2020RDB Ref. Manual) */
#if (defined VXWORKS)
       pLinkParmsAdapter->dwRxInterrupt = EPIC_TSEC2RX_INT_VEC; /* from vxbEpicIntCtlr.h */
#endif
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 3) /* eTSEC3 */
    {
       pLinkParmsAdapter->abyStationAddress[5] = 0x83;
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x26000;
       pLinkParmsAdapter->dwPhyAddr = 1; /* RGMII Phy on addr. 1 (from P2020RDB Ref. Manual) */
#if (defined VXWORKS)
       pLinkParmsAdapter->dwRxInterrupt = EPIC_TSEC3RX_INT_VEC; /* from vxbEpicIntCtlr.h */
#endif
    }

    pLinkParmsAdapter->dwLocalMdioBase = pLinkParmsAdapter->dwRegisterBase;

#elif defined(TSEC_CFG_TWRP1025)
    /*
     * The following configuration is for Freescale TWR-P1025 board
     * (CPU P1025, Freescale-Linuxkernel 3.0.4)
     * Hardware resource informations (phy-addr, interrupts, io-base, ...)
     * are taken from Linux-kernel's device tree for the TWR-P1025 (twr-p1025_32b.dts)
     */

    if (pLinkParmsAdapter->linkParms.dwInstance > 2) /* TWR-P1025 has 3 eTSEC's, but only eTSEC1 and eTSEC3 are routed out to RJ45 ports */
    {
       EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Device unit must be <= 2! There are only 2 eTSEC's on P1025TWR.\n"));
       return EC_FALSE;
    }

    {
       EC_T_BYTE abyStationAddress[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xBA, 0xBE };
       memcpy(pLinkParmsAdapter->abyStationAddress, abyStationAddress, sizeof(pLinkParmsAdapter->abyStationAddress));
    }

    pLinkParmsAdapter->bMaster = EC_FALSE;
    pLinkParmsAdapter->eETSECType = eETSEC_TWRP1025;
    dwCcsrbar = 0xffe00000;

    pLinkParmsAdapter->dwPhyMdioBase = dwCcsrbar + 0x24000; /* eTSEC 1, MDIO */
    pLinkParmsAdapter->dwTbiPhyAddr = 16; /* Dummy address assigned to internal TBI PHY */
    pLinkParmsAdapter->oMiiBusMtx = EC_NULL; /* LinkOsCreateLock(eLockType_DEFAULT); */

    /* Get interrupt number from Nth eTSEC device in PowerPC device tree */
    pLinkParmsAdapter->dwRxInterrupt = dwInstance - 1;

    if (pLinkParmsAdapter->linkParms.dwInstance == 1) /* eTSEC1 */
    {
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0xB0000; /* eTSEC1, Group 0 */
       pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x24000; /* eTSEC1, MDIO */
       pLinkParmsAdapter->dwPhyAddr = 2;
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 2) /* eTSEC3 */
    {
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0xB2000; /* eTSEC3, Group 0 */
       pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x26000; /* eTSEC3, MDIO */
       pLinkParmsAdapter->dwPhyAddr = 1;
    }

#elif defined(TSEC_CFG_ISTMPC8548)

    /*
     * The following configuration is for the Instron "MPC8548 MiniTower" board
     * (CPU MPC8548, VxWorks 6.9 PPC)
     */

    if (dwInstance > 2) /* MPC8548 has 4 eTSEC's, but only the first 2 are routed out to RJ45 ports */
    {
       EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Device unit must be 1 or 2!\n"));
       dwRetVal = EC_E_INVALIDPARM;
       goto Exit;
    }

    {
       EC_T_BYTE abyStationAddress[] = { 0x00, 0x02, 0xCE, 0x90, 0x02, 0x24 };
       memcpy(pLinkParmsAdapter->abyStationAddress, abyStationAddress, sizeof(pLinkParmsAdapter->abyStationAddress));
    }

    pLinkParmsAdapter->bMaster = EC_FALSE;
    pLinkParmsAdapter->eETSECType = eETSEC_ISTMPC8548;

    dwCcsrbar = sysGetPeripheralBase();

    pLinkParmsAdapter->dwPhyMdioBase = dwCcsrbar + 0x24000; /* eTSEC 0 */
    pLinkParmsAdapter->dwTbiPhyAddr = 16; /* Dummy address assigned to internal TBI PHY */
    pLinkParmsAdapter->oMiiBusMtx = EC_NULL; /* LinkOsCreateLock(eLockType_DEFAULT); */

    if (dwInstance == 1) /* eTSEC1 (Assigned to VxWorks, don't use!) */
    {
       pLinkParmsAdapter->abyStationAddress[5] = 0x24;
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x24000;
       pLinkParmsAdapter->dwPhyAddr = 25; /* from hwconf.c */
       pLinkParmsAdapter->dwRxInterrupt = EPIC_TSEC1RX_INT_VEC; /* from vxbEpicIntCtlr.h */
    }
    else if (dwInstance == 2) /* eTSEC2 (This one is used by EtherCAT) */
    {
       pLinkParmsAdapter->abyStationAddress[5] = 0x25;
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x25000;
       pLinkParmsAdapter->dwPhyAddr = 26; /* from hwconf.c */
       pLinkParmsAdapter->dwRxInterrupt = EPIC_TSEC2RX_INT_VEC; /* from vxbEpicIntCtlr.h */
    }

    pLinkParmsAdapter->dwLocalMdioBase = pLinkParmsAdapter->dwRegisterBase;

#elif defined(TSEC_CFG_XJ_EPU20C)

    /*
     * The following configuration is for the Instron "XJ Electric Corp EPU20C" board
     */

    if (pLinkParmsAdapter->linkParms.dwInstance > 2) /* MPC8536 has 2 eTSEC's, but only the first 2 are routed out to RJ45 ports */
    {
       EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Device unit must be 1 or 2!\n"));
       dwRetVal = EC_E_INVALIDPARM;
       goto Exit;
    }

    {
       EC_T_BYTE abyStationAddress[] = { 0x00, 0x02, 0xCE, 0x90, 0x02, 0x24 };
       memcpy(pLinkParmsAdapter->abyStationAddress, abyStationAddress, sizeof(pLinkParmsAdapter->abyStationAddress));
    }

    pLinkParmsAdapter->bMaster = EC_FALSE;
    pLinkParmsAdapter->eETSECType = eETSEC_XJ_EPU20C;
    dwCcsrbar = sysGetPeripheralBase();

    pLinkParmsAdapter->dwPhyMdioBase = dwCcsrbar + 0x24000; /* eTSEC 0 */
    pLinkParmsAdapter->dwTbiPhyAddr = 16; /* Dummy address assigned to internal TBI PHY */
    pLinkParmsAdapter->oMiiBusMtx = EC_NULL; /* LinkOsCreateLock(eLockType_DEFAULT); */

    if (pLinkParmsAdapter->linkParms.dwInstance == 1) /* eTSEC1 (Assigned to VxWorks, don't use!) */
    {
       pLinkParmsAdapter->abyStationAddress[5] = 0x24;
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x24000;
       pLinkParmsAdapter->dwPhyAddr = 0; /* from hwconf.c */
       pLinkParmsAdapter->dwRxInterrupt = EPIC_TSEC1RX_INT_VEC; /* from vxbEpicIntCtlr.h */
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 2) /* eTSEC2 (This one is used by EtherCAT) */
    {
       pLinkParmsAdapter->abyStationAddress[5] = 0x25;
       pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x26000;
       pLinkParmsAdapter->dwPhyAddr = 1; /* from hwconf.c */
       pLinkParmsAdapter->dwRxInterrupt = EPIC_TSEC2RX_INT_VEC; /* from vxbEpicIntCtlr.h */
    }

    pLinkParmsAdapter->dwLocalMdioBase = pLinkParmsAdapter->dwRegisterBase;
#elif defined(TSEC_CFG_TWRLS1021A)
    /*
     * The following configuration is for Freescale TWR-LS1021A-PB board
     * (CPU LS1021A, Freescale-Linuxkernel 3.12)
     * Hardware resource informations (phy-addr, interrupts, io-base, ...)
     * are taken from Linux-kernel's device tree for the TWR-LS1021A (ls1021a.dts and ls1021a.dtsi)
     */

    if (2 < pLinkParmsAdapter->linkParms.dwInstance || 1 > pLinkParmsAdapter->linkParms.dwInstance) /* TWR-LS1021A-PB has 3 eTSEC's, but only two of them can be routed out to RJ45 ports */
    {
       EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Device unit must be 1 or 2! The 3. PHY on TWR-LS1012A-PB is not yet supported\n"));
       dwRetVal = EC_E_INVALIDPARM;
       goto Exit;
    }

    {
       EC_T_BYTE abyStationAddress[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xBA, 0xBE };
       memcpy(pLinkParmsAdapter->abyStationAddress, abyStationAddress, sizeof(pLinkParmsAdapter->abyStationAddress));
    }

    dwCcsrbar = 0x2D00000;

    /* Initialization and settings for ETSEC */
    pLinkParmsAdapter->bMaster = EC_TRUE;
    pLinkParmsAdapter->eETSECType = eETSEC_TWRLS1021A;

    pLinkParmsAdapter->dwPhyMdioBase = dwCcsrbar + 0x24000; /* eTSEC 1, MDIO */
    pLinkParmsAdapter->dwTbiPhyAddr = 16; /* Dummy address assigned to internal TBI PHY */
    pLinkParmsAdapter->oMiiBusMtx = EC_NULL; /* LinkOsCreateLock(eLockType_DEFAULT); */

    /* Get interrupt number from Nth eTSEC device in PowerPC device tree */
    pLinkParmsAdapter->dwRxInterrupt = pLinkParmsAdapter->linkParms.dwInstance - 1;

#if (defined __arm__)
    pLinkParmsAdapter->bNotUseDmaBuffers = EC_TRUE;
#endif

    if (pLinkParmsAdapter->linkParms.dwInstance == 1) /* eTSEC1 */
    {
        pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x10000; /* eTSEC1, Group 0 */
        pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x24000; /* eTSEC1, MDIO */
#ifdef  EC_VERSION_ETKERNEL
        pLinkParmsAdapter->dwTbiPhyAddr = 0x1f;
#endif
        pLinkParmsAdapter->dwPhyAddr = 2;
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 2) /* eTSEC2 */
    {
        pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x50000; /* eTSEC3, Group 0 */
        pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x24000; /* eTSEC3, MDIO */
#ifdef  EC_VERSION_ETKERNEL
        pLinkParmsAdapter->dwTbiPhyAddr = 0x1e;
#endif
        pLinkParmsAdapter->dwPhyAddr = 0;
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 3) /* eTSEC3 */
    {
        pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x90000; /* eTSEC3, Group 0 */
        pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x24000; /* eTSEC3, MDIO */
        pLinkParmsAdapter->dwPhyAddr = 1;
    }
#elif defined(TSEC_CFG_TQMLS_LS102XA)
    /*
     * The following configuration is for TQMLS-LS102xA module (CPU LS1021)
     */

    if (pLinkParmsAdapter->linkParms.dwInstance > 3)
    {
       EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Device unit must be <= 3!\n"));
       dwRetVal = EC_E_INVALIDPARM;
       goto Exit;
    }

    {
       EC_T_BYTE abyStationAddress[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xBA, 0xBE };
       memcpy(pLinkParmsAdapter->abyStationAddress, abyStationAddress, sizeof(pLinkParmsAdapter->abyStationAddress));
    }

    pLinkParmsAdapter->bMaster = EC_FALSE;
    pLinkParmsAdapter->eETSECType = eETSEC_TQMLS_LS102XA;

    dwCcsrbar = 0x2D00000;

    pLinkParmsAdapter->dwPhyMdioBase = dwCcsrbar + 0x24000; /* eTSEC 1, MDIO */
    pLinkParmsAdapter->dwTbiPhyAddr = 16; /* Dummy address assigned to internal TBI PHY */
    pLinkParmsAdapter->oMiiBusMtx = EC_NULL; /* LinkOsCreateLock(eLockType_DEFAULT); */

    /* Get interrupt number from Nth eTSEC device in PowerPC device tree */
    pLinkParmsAdapter->dwRxInterrupt = pLinkParmsAdapter->linkParms.dwInstance - 1;

#if (defined __arm__)
    pLinkParmsAdapter->bNotUseDmaBuffers = EC_TRUE;
#endif

    if (pLinkParmsAdapter->linkParms.dwInstance == 1) /* eTSEC1 */
    {
        pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x10000; /* eTSEC1, Group 0 */
        pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x24000; /* eTSEC1, MDIO */
        pLinkParmsAdapter->dwPhyAddr = 12;
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 2) /* eTSEC2 */
    {
        pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x50000; /* eTSEC3, Group 0 */
        pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x24000; /* eTSEC3, MDIO */
        pLinkParmsAdapter->dwPhyAddr = 3;
    }
    else if (pLinkParmsAdapter->linkParms.dwInstance == 3) /* eTSEC3 */
    {
        pLinkParmsAdapter->dwRegisterBase = dwCcsrbar + 0x90000; /* eTSEC3, Group 0 */
        pLinkParmsAdapter->dwLocalMdioBase = dwCcsrbar + 0x24000; /* eTSEC3, MDIO */
        pLinkParmsAdapter->dwPhyAddr = 4;
    }
#else
#error "Board configuration must be selected"
#endif

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLETSEC */

#if (defined INCLUDE_EMLLFSLFEC)
/***************************************************************************************************/
/**
\brief  Try to create FslFec link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineFslFec(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                   EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_FSLFEC* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-fslfec") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_FSLFEC*)OsMalloc(sizeof(EC_T_LINK_PARMS_FSLFEC));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_FSLFEC));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_FSLFEC, sizeof(EC_T_LINK_PARMS_FSLFEC), EC_LINK_PARMS_IDENT_FSLFEC, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    pLinkParmsAdapter->dwRxBuffers = 96;
    pLinkParmsAdapter->dwTxBuffers = 96;

    /* get reference board */
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if      (0 == OsStricmp((*ptcWord), "mars"))     { pLinkParmsAdapter->eFecType = eFEC_IMX6;  pLinkParmsAdapter->ePhyInterface = ePHY_RGMII;      }
    else if (0 == OsStricmp((*ptcWord), "sabrelite")){ pLinkParmsAdapter->eFecType = eFEC_IMX6;  pLinkParmsAdapter->ePhyInterface = ePHY_RGMII;      }
    else if (0 == OsStricmp((*ptcWord), "sabresd"))  { pLinkParmsAdapter->eFecType = eFEC_IMX6;  pLinkParmsAdapter->ePhyInterface = ePHY_RGMII;      }
    else if (0 == OsStricmp((*ptcWord), "imx28evk"))
            if (pLinkParmsAdapter->linkParms.dwInstance == 1)
                                                     { pLinkParmsAdapter->eFecType = eFEC_IMX28; pLinkParmsAdapter->ePhyInterface = ePHY_RMII;
                                                       pLinkParmsAdapter->dwPhyAddr = 0;}
            else                                     { pLinkParmsAdapter->eFecType = eFEC_IMX28; pLinkParmsAdapter->ePhyInterface = ePHY_RMII;
                                                       pLinkParmsAdapter->dwPhyAddr = 1;}
    else if (0 == OsStricmp((*ptcWord), "topaz"))    { pLinkParmsAdapter->eFecType = eFEC_IMX25; pLinkParmsAdapter->ePhyInterface = ePHY_RMII;       }
    else if (0 == OsStricmp((*ptcWord), "imxceetul2"))
    {
        pLinkParmsAdapter->eFecType = eFEC_IMX6;
        pLinkParmsAdapter->ePhyInterface = ePHY_RMII;
        if (pLinkParmsAdapter->linkParms.dwInstance == 1)
        {
            pLinkParmsAdapter->dwPhyAddr = 1;
        }
        else if (pLinkParmsAdapter->linkParms.dwInstance == 2)
        {
            pLinkParmsAdapter->dwPhyAddr = 2;
        }
        else
        {
            dwRetVal = EC_E_INVALIDPARM;
            goto Exit;
        }
    }
    else if (0 == OsStricmp((*ptcWord), "custom"))
    {
        /* get FEC type */
        (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
        if      (0 == OsStricmp((*ptcWord), "imx25")) pLinkParmsAdapter->eFecType = eFEC_IMX25;
        else if (0 == OsStricmp((*ptcWord), "imx28")) pLinkParmsAdapter->eFecType = eFEC_IMX28;
        else if (0 == OsStricmp((*ptcWord), "imx53")) pLinkParmsAdapter->eFecType = eFEC_IMX53;
        else if (0 == OsStricmp((*ptcWord), "imx6"))  pLinkParmsAdapter->eFecType = eFEC_IMX6;
        else if (0 == OsStricmp((*ptcWord), "vf6"))   pLinkParmsAdapter->eFecType = eFEC_VF6;
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid FecType value\n"));
            dwRetVal = EC_E_INVALIDPARM;
            goto Exit;
        }
        /* get PHY interface */
        (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
        if      (0 == OsStricmp((*ptcWord), "fixed")) pLinkParmsAdapter->ePhyInterface = ePHY_FIXED_LINK;
        else if (0 == OsStricmp((*ptcWord), "mii"))   pLinkParmsAdapter->ePhyInterface = ePHY_MII;
        else if (0 == OsStricmp((*ptcWord), "rmii"))  pLinkParmsAdapter->ePhyInterface = ePHY_RMII;
        else if (0 == OsStricmp((*ptcWord), "gmii"))  pLinkParmsAdapter->ePhyInterface = ePHY_GMII;
        else if (0 == OsStricmp((*ptcWord), "sgmii")) pLinkParmsAdapter->ePhyInterface = ePHY_SGMII;
        else if (0 == OsStricmp((*ptcWord), "rgmii")) pLinkParmsAdapter->ePhyInterface = ePHY_RGMII;
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid PhyInterface value\n"));
            dwRetVal = EC_E_INVALIDPARM;
            goto Exit;
        }
        /* get PHY address */
        if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwPhyAddr))
        {
            pLinkParmsAdapter->dwPhyAddr = 0;
        }
    }
    else
    {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid RefBoard value\n"));
            dwRetVal = EC_E_INVALIDPARM;
            goto Exit;
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLFSLFEC */

#if (defined INCLUDE_EMLLGEM)

EC_T_BOOL ParseGEMType(
    EC_T_CHAR**     ptcWord,
    EC_T_CHAR**     lpCmdLine,
    EC_T_CHAR*      tcStorage,
    EC_T_GEM_TYPE*  peType)
{
    EC_T_BOOL bRes = EC_FALSE;

    if (ParseString(ptcWord, lpCmdLine, tcStorage))
    {
        bRes = EC_TRUE;
        if (0 == OsStricmp((*ptcWord), "zynq7000"))         { *peType = eGemType_Zynq7000; }
        else if (0 == OsStricmp((*ptcWord), "ultrascale"))  { *peType = eGemType_ZynqUltrascale; }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid GEM type %s\n", ptcWord));
            bRes = EC_FALSE;
        }
    }
    return bRes;
}

/***************************************************************************************************/
/**
\brief  Try to create GEM link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineGEM(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_GEM* pLinkParmsAdapter = EC_NULL;
EC_T_DWORD dwUseGmiiToRgmiiConv = 0;

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-gem") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_GEM*)OsMalloc(sizeof(EC_T_LINK_PARMS_GEM));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_GEM));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_GEM, sizeof(EC_T_LINK_PARMS_GEM), EC_LINK_PARMS_IDENT_GEM, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
      || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode) )
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    if (pLinkParmsAdapter->linkParms.dwInstance < 1 || pLinkParmsAdapter->linkParms.dwInstance > 4)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be  within 1 to 4\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* default parameters */
    switch (pLinkParmsAdapter->linkParms.dwInstance)
    {
    case 1:
        pLinkParmsAdapter->dwPhyAddr = 4;
        pLinkParmsAdapter->eRxSource = eGemRxSource_MIO;
        break;
    case 2:
        pLinkParmsAdapter->dwPhyAddr = 1;
        pLinkParmsAdapter->eRxSource = eGemRxSource_EMIO;
        break;
    case 3:
    case 4:
    	pLinkParmsAdapter->dwPhyAddr = 12;
    	pLinkParmsAdapter->eRxSource = eGemRxSource_MIO;
		pLinkParmsAdapter->eGemType = eGemType_ZynqUltrascale;
		break;
    default:
    	break;
    }
    dwUseGmiiToRgmiiConv = EC_FALSE;
    pLinkParmsAdapter->dwConvPhyAddr = 0;

    /* get reference board */
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if (EC_NULL == *ptcWord)
    {
    	/* nothing to do */
    }
    else if (0 == OsStricmp((*ptcWord), "microzed"))
	{
		pLinkParmsAdapter->dwPhyAddr = 4;
		pLinkParmsAdapter->eRxSource = eGemRxSource_MIO;
	}
    else if (0 == OsStricmp((*ptcWord), "zedboard"))
	{
    	pLinkParmsAdapter->dwPhyAddr = 0;
    	pLinkParmsAdapter->eRxSource = eGemRxSource_MIO;
    }
    else if (0 == OsStricmp((*ptcWord), "zc702"))
	{
		pLinkParmsAdapter->dwPhyAddr = 7;
		pLinkParmsAdapter->eRxSource = eGemRxSource_MIO;
	}
	else if ((0 == OsStricmp((*ptcWord), "zcu102")) || (0 == OsStricmp((*ptcWord), "zcu104")))
	{
		pLinkParmsAdapter->dwPhyAddr = 12;
		pLinkParmsAdapter->eRxSource = eGemRxSource_MIO;
		pLinkParmsAdapter->eGemType = eGemType_ZynqUltrascale;
	}
    else if (0 == OsStricmp((*ptcWord), "custom"))
	{
		/* parse optional parameters or use default values */
		if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwPhyAddr)
		 || !ParseDword(ptcWord, lpCmdLine, tcStorage, (EC_T_DWORD*)&pLinkParmsAdapter->eRxSource)
		 || !ParseDword(ptcWord, lpCmdLine, tcStorage, &dwUseGmiiToRgmiiConv)
		 || !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwConvPhyAddr)
		 || !ParseGEMType(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->eGemType))
		{
			if (EC_NULL != pbGetNextWord)
			{
				*pbGetNextWord = EC_FALSE;
			}
		}
	}
    else
    {
    	if (EC_NULL != pbGetNextWord)
    	{
    		*pbGetNextWord = EC_FALSE;
    	}
    }
    pLinkParmsAdapter->bUseGmiiToRgmiiConv = (EC_T_BOOL)dwUseGmiiToRgmiiConv;
    pLinkParmsAdapter->dwRxInterrupt = pLinkParmsAdapter->linkParms.dwInstance - 1;
    pLinkParmsAdapter->bUseDmaBuffers = EC_FALSE;
    pLinkParmsAdapter->bNoPhyAccess = EC_FALSE;   /* Link layer should initialize PHY and read link status (connected/disconnected) */

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLGEM */

#if (defined INCLUDE_EMLLI8254X)
/***************************************************************************************************/
/**
\brief  Try to create I8254x link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineI8254x(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                   EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_I8254X* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-i8254x") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_I8254X*)OsMalloc(sizeof(EC_T_LINK_PARMS_I8254X));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_I8254X));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_I8254X,
        sizeof(EC_T_LINK_PARMS_I8254X),
        EC_LINK_PARMS_IDENT_I8254X, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    else {
        printf("dwInstance = %d, eLinkMode = %d\n", pLinkParmsAdapter->linkParms.dwInstance,
            pLinkParmsAdapter->linkParms.eLinkMode);
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLI8254X */

#if (defined INCLUDE_EMLLI8255X)
/***************************************************************************************************/
/**
\brief  Try to create I8255X link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineI8255x(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                   EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_I8255X* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-i8255x") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_I8255X*)OsMalloc(sizeof(EC_T_LINK_PARMS_I8255X));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_I8255X));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_I8255X, sizeof(EC_T_LINK_PARMS_I8255X), EC_LINK_PARMS_IDENT_I8255X, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLI8255X */

#if (defined INCLUDE_EMLLICSS)

EC_T_VOID cbTtsStartCycle(EC_T_VOID* pvThreadParamDesc);

/***************************************************************************************************/
/**
\brief  Try to create ICSS link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineICSS(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                 EC_T_LINK_PARMS** ppLinkParms
#if defined(INCLUDE_TTS)
                                                 , EC_T_DWORD* pdwTtsBusCycleUsec /* [out] TTS Bus Cycle overrides original one when TTS is used */
                                                 , EC_T_VOID** pvvTtsEvent
#endif
)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_ICSS* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-icss") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_ICSS*)OsMalloc(sizeof(EC_T_LINK_PARMS_ICSS));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_ICSS));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_ICSS, sizeof(EC_T_LINK_PARMS_ICSS), EC_LINK_PARMS_IDENT_ICSS, 1, EcLinkMode_POLLING);

    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
      || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode)
      )
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    if (pLinkParmsAdapter->linkParms.dwInstance < 1 || pLinkParmsAdapter->linkParms.dwInstance > 4)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "ICSS: Port number must be between 1 and 4\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* get master/slave flag */
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if ((0 != (*ptcWord)) && (0 == OsStricmp((*ptcWord), "m")))
    {
        pLinkParmsAdapter->bMaster = EC_TRUE;
    }
    else if ((0 != (*ptcWord)) && (0 == OsStricmp((*ptcWord), "s")))
    {
        pLinkParmsAdapter->bMaster = EC_FALSE;
    }
    else
    {
        EcLogMsg(EC_LOG_LEVEL_WARNING, (pEcLogContext, EC_LOG_LEVEL_WARNING, "ICSS: No master/slave flag specified. Assume master\n"));
        pLinkParmsAdapter->bMaster = EC_TRUE;
        *pbGetNextWord = EC_FALSE;
    }

    /* get reference board */
    if (*pbGetNextWord)
    {
        (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    }
    if ((0 != (*ptcWord)) && (0 == OsStricmp((*ptcWord), "am572x-idk")))
    {
        pLinkParmsAdapter->eBoardType = EcLinkIcssBoard_am572x;
    }
    else if ((0 != (*ptcWord)) && (0 == OsStricmp((*ptcWord), "am571x-idk")))
    {
        pLinkParmsAdapter->eBoardType = EcLinkIcssBoard_am571x;
    }
    else if ((0 != (*ptcWord)) && (0 == OsStricmp((*ptcWord), "am3359-icev2")))
    {
        pLinkParmsAdapter->eBoardType = EcLinkIcssBoard_am3359;
    }
    else if ((0 != (*ptcWord)) && (0 == OsStricmp((*ptcWord), "am572x-emerson")))
    {
        pLinkParmsAdapter->eBoardType = EcLinkIcssBoard_am572x_emerson;
    }
    else
    {
        EcLogMsg(EC_LOG_LEVEL_WARNING, (pEcLogContext, EC_LOG_LEVEL_WARNING, "ICSS: No board name specified. Assume am572x-idk\n"));
        pLinkParmsAdapter->eBoardType = EcLinkIcssBoard_am572x;
        *pbGetNextWord = EC_FALSE;
    }

    /* Disable TTS bz default */
    pLinkParmsAdapter->bTts = EC_FALSE;
    pLinkParmsAdapter->dwTtsCycleTimeUsec  = 0;
    pLinkParmsAdapter->dwTtsSendTimeUsec = 0;
    pLinkParmsAdapter->pfnTtsStartCycle    = EC_NULL;

#if defined(INCLUDE_TTS)
    *pdwTtsBusCycleUsec = 0;
    /* get TTS mode and config time */
    if (*pbGetNextWord)
    {
        (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    }
    if (0 != (*ptcWord))
    {
        if (0 == OsStricmp((*ptcWord), "tts"))
        {
            pLinkParmsAdapter->bTts = EC_TRUE;

            if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwTtsCycleTimeUsec) || pLinkParmsAdapter->dwTtsCycleTimeUsec < 125 )
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "ICSS: Wrong TTS cycle period specified\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }

            if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->dwTtsSendTimeUsec) )
            {
                EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "ICSS: Wrong TTS config time specified\n"));
                dwRetVal = EC_E_INVALIDPARM;
                goto Exit;
            }
            pLinkParmsAdapter->pfnTtsStartCycle = &cbTtsStartCycle;
            pLinkParmsAdapter->pvTtsStartCycleContext = OsCreateEvent();
            *pdwTtsBusCycleUsec = pLinkParmsAdapter->dwTtsCycleTimeUsec;
            *pvvTtsEvent = pLinkParmsAdapter->pvTtsStartCycleContext;
        }
        else
        {
            *pbGetNextWord = EC_FALSE;
        }
    }
#endif

    // todo: change this predefined address for each port.
    if( pLinkParmsAdapter->bMaster == EC_TRUE)
    {
        pLinkParmsAdapter->abyMac[0] = 0x01;
    }
    else
    {
        pLinkParmsAdapter->abyMac[0] = 0x02;
    }
    pLinkParmsAdapter->abyMac[1] = 0xB4;
    pLinkParmsAdapter->abyMac[2] = 0xC3;
    pLinkParmsAdapter->abyMac[3] = 0xDD;
    pLinkParmsAdapter->abyMac[4] = 0xEE;
    pLinkParmsAdapter->abyMac[5] = 0xFF;

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLICSS */

#if (defined INCLUDE_EMLLL9218I)
/***************************************************************************************************/
/**
\brief  Try to create L9218I link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineL9218i(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                   EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_L9218I* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(ptcWord);
    EC_UNREFPARM(lpCmdLine);
    EC_UNREFPARM(tcStorage);
    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-l9218i") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_L9218I*)OsMalloc(sizeof(EC_T_LINK_PARMS_L9218I));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_L9218I));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_L9218I, sizeof(EC_T_LINK_PARMS_L9218I), EC_LINK_PARMS_IDENT_L9218I, 1, EcLinkMode_POLLING);

    /* get mode */
    if (!ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLL9218I */

#if (defined INCLUDE_EMLLNDISUIO)
/***************************************************************************************************/
/**
\brief  Try to create NDISUIO link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineNDISUIO(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                    EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_NDISUIO* pLinkParmsAdapter = EC_NULL;

#define NDISUIO_DEVNAME   TEXT("ECT1:")
#define NDISUIO_DRIVERKEY TEXT("Drivers\\BuiltIn\\ECAT")
HANDLE      hNdisUioDevice  = EC_NULL;
HANDLE      hNdisUioDriver  = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-ndisuio") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* get next word */
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if (((*ptcWord) == EC_NULL) || (OsStrncmp( (*ptcWord), "-", 1) == 0))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_NDISUIO*)OsMalloc(sizeof(EC_T_LINK_PARMS_NDISUIO));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_NDISUIO));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_NDISUIO, sizeof(EC_T_LINK_PARMS_NDISUIO), EC_LINK_PARMS_IDENT_NDISUIO, 1, EcLinkMode_POLLING);

    /* check if NDISUIO driver started */
    EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Check if ECAT driver is started..."));
    hNdisUioDevice = CreateFile( NDISUIO_DEVNAME, GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, 0);
    if ((hNdisUioDevice != EC_NULL) && (hNdisUioDevice != INVALID_HANDLE_VALUE))
    {
        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Ok!\n"));
    }
    else
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Failed!\n"));
        hNdisUioDevice = EC_NULL;
    }
    /* try to load driver if not already loaded */
    if (hNdisUioDevice == EC_NULL)
    {
        EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Try to load ECAT driver..."));
        hNdisUioDriver = ActivateDeviceEx (NDISUIO_DRIVERKEY, 0,  0, 0);
        if ((hNdisUioDriver != EC_NULL) && (hNdisUioDriver != INVALID_HANDLE_VALUE))
        {
            EcLogMsg(EC_LOG_LEVEL_INFO, (pEcLogContext, EC_LOG_LEVEL_INFO, "Ok!\n"));
        }
        else
        {
            EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Failed!\n"));
            hNdisUioDriver = EC_NULL;
            dwRetVal = EC_E_INVALIDSTATE;
            goto Exit;
        }
    }
    /* check if driver is available */
    if ((hNdisUioDevice == EC_NULL) && (hNdisUioDriver == EC_NULL))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "No NDISUIO ECAT driver available!!!\n"));
        dwRetVal = EC_E_INVALIDSTATE;
        goto Exit;
    }
    else if (hNdisUioDevice != EC_NULL)
    {
        /* close handle, it was just for the check */
        CloseHandle(hNdisUioDevice);
        hNdisUioDevice = EC_NULL;
    }
    /* NdisUio uses the network adapter name to select the appropriate network interface */
#ifdef  UNICODE
    _snwprintf((wchar_t*)pLinkParmsAdapter->szNetworkAdapterName, MAX_LEN_NDISUIO_ADAPTER_NAME, L"%S", (*ptcWord));
#else
    _snwprintf(pLinkParmsAdapter->szNetworkAdapterName, MAX_LEN_NDISUIO_ADAPTER_NAME, L"%s", (*ptcWord));
#endif
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLNDISUIO */

#if (defined INCLUDE_EMLLR6040)
/***************************************************************************************************/
/**
\brief  Try to create R6040 link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineR6040(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                  EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_R6040* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-r6040") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_R6040*)OsMalloc(sizeof(EC_T_LINK_PARMS_R6040));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_R6040));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_R6040, sizeof(EC_T_LINK_PARMS_R6040), EC_LINK_PARMS_IDENT_R6040, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLR6040 */

#if (defined INCLUDE_EMLLRIN32M3)
/***************************************************************************************************/
/**
\brief  Try to create RIN32M3 link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineRIN32M3(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                    EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_RIN32M3* pLinkParmsAdapter = EC_NULL;
const size_t nParmsSize = sizeof(EC_T_LINK_PARMS_RIN32M3);

    EC_UNREFPARM(lpCmdLine);
    EC_UNREFPARM(tcStorage);
    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp( (*ptcWord), "-rin32m3") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_RIN32M3*)OsMalloc(nParmsSize);
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, nParmsSize);
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_RIN32M3, nParmsSize, EC_LINK_PARMS_IDENT_RIN32M3, 1, EcLinkMode_POLLING);

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* EC_VERSION_RIN32M3 */

#if (defined INCLUDE_EMLLRTL8139)
/***************************************************************************************************/
/**
\brief  Try to create RTL8139 link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineRTL8139(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                    EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_RTL8139* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-rtl8139") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_RTL8139*)OsMalloc(sizeof(EC_T_LINK_PARMS_RTL8139));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_RTL8139));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_RTL8139, sizeof(EC_T_LINK_PARMS_RTL8139), EC_LINK_PARMS_IDENT_RTL8139, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters: instance, mode */
    if ( !ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
        || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLRTL8139 */

#if (defined INCLUDE_EMLLRTL8169)
/***************************************************************************************************/
/**
\brief  Try to create RTL8169 link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineRTL8169(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                    EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_RTL8169* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-rtl8169") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_RTL8169*)OsMalloc(sizeof(EC_T_LINK_PARMS_RTL8169));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_RTL8169));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_RTL8169, sizeof(EC_T_LINK_PARMS_RTL8169), EC_LINK_PARMS_IDENT_RTL8169, 1, EcLinkMode_POLLING);

    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance)
       || !ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    pLinkParmsAdapter->bNotUseDmaBuffers = EC_FALSE;
#if (defined _ARM_) || (defined __arm__) || (defined __aarch64__)
    /* for arm platform we should not use DMA memory because any unaligned access creates crash */
    pLinkParmsAdapter->bNotUseDmaBuffers = EC_TRUE;
#endif

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLRTL8169 */

#if (defined INCLUDE_EMLLRZT1)
/***************************************************************************************************/
/**
\brief  Try to create RZT1 link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineRZT1(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_RZT1* pLinkParmsAdapter = EC_NULL;
    const size_t nParmsSize = sizeof(EC_T_LINK_PARMS_RZT1);

    EC_UNREFPARM(lpCmdLine);
    EC_UNREFPARM(tcStorage);
    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp( (*ptcWord), "-rzt1") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_RZT1*)OsMalloc(nParmsSize);
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, nParmsSize);
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_RZT1, nParmsSize, EC_LINK_PARMS_IDENT_RZT1, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters */
    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    if (pLinkParmsAdapter->linkParms.dwInstance < 1 || pLinkParmsAdapter->linkParms.dwInstance > 2)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* EC_VERSION_RZT1 */

#if (defined INCLUDE_EMLLSHETH)
/***************************************************************************************************/
/**
\brief  Try to create Super H link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
EC_E_NOTFOUND    if command line was not matching
EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineSHEth(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms)
{
    EC_T_DWORD dwRetVal = EC_E_ERROR;
    EC_T_LINK_PARMS_SHETH* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-sheth") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* get next word */
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if (((*ptcWord) == EC_NULL) || (OsStrncmp((*ptcWord), "-", 1) == 0))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_SHETH*)OsMalloc(sizeof(EC_T_LINK_PARMS_SHETH));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_SHETH));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_SHETH, sizeof(EC_T_LINK_PARMS_SHETH), EC_LINK_PARMS_IDENT_SHETH, 1, EcLinkMode_POLLING);

    /* parse mandatory parameters */
    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.dwInstance))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    if (pLinkParmsAdapter->linkParms.dwInstance < 1 || pLinkParmsAdapter->linkParms.dwInstance > 2)
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Instance number must be 1 or 2\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    /* get reference board */
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if (0 == OsStricmp((*ptcWord), "a800eva"))
    {
        pLinkParmsAdapter->eType = eSHEth_R8A7740;
        pLinkParmsAdapter->dwBaseAddr = 0xE9A00000;
        pLinkParmsAdapter->byPhyAddr = 0;
    }
    else if (0 == OsStricmp((*ptcWord), "rzg1e"))
    {
        pLinkParmsAdapter->eType = eSHEth_R8A77450;
        pLinkParmsAdapter->dwBaseAddr = 0xEE700000;
        pLinkParmsAdapter->byPhyAddr = 1;
    }
    else
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "Invalid RefBoard value\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    {
        EC_T_BYTE abyStationAddress[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xBA, 0xBE };
        memcpy(pLinkParmsAdapter->abyStationAddress, abyStationAddress, sizeof(pLinkParmsAdapter->abyStationAddress));
    }

    pLinkParmsAdapter->bNotUseDmaBuffers = EC_TRUE;

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLSHETH */

#if (defined INCLUDE_EMLLSNARF)
/***************************************************************************************************/
/**
\brief  Try to create Snarf link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineSnarf(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                  EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_SNARF* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-snarf") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* get next word */
    (*ptcWord) = GetNextWord(lpCmdLine, tcStorage);
    if (((*ptcWord) == EC_NULL) || (OsStrncmp( (*ptcWord), "-", 1) == 0))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_SNARF*)OsMalloc(sizeof(EC_T_LINK_PARMS_SNARF));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_SNARF));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_SNARF, sizeof(EC_T_LINK_PARMS_SNARF), EC_LINK_PARMS_IDENT_SNARF, 1, EcLinkMode_POLLING);

    /* get adapter name */
    OsSnprintf(pLinkParmsAdapter->szAdapterName, MAX_LEN_SNARF_ADAPTER_NAME, "%s", (*ptcWord));

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLSNARF */

#if (defined INCLUDE_EMLLSOCKRAW)
/***************************************************************************************************/
/**
\brief  Try to create SockRaw link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineSockRaw(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                    EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_SOCKRAW* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-sockraw") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_SOCKRAW*)OsMalloc(sizeof(EC_T_LINK_PARMS_SOCKRAW));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_SOCKRAW));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_SOCKRAW, sizeof(EC_T_LINK_PARMS_SOCKRAW), EC_LINK_PARMS_IDENT_SOCKRAW, 1, EcLinkMode_POLLING);

    /* get adapter name */
    *ptcWord = GetNextWord(lpCmdLine, tcStorage);
    if ((*ptcWord == EC_NULL) || (OsStrncmp(*ptcWord, "-", 1) == 0))
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    if (OsStrlen((char*)*ptcWord) > MAX_LEN_SOCKRAW_ADAPTER_NAME - 1)
    {
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    OsStrncpy(pLinkParmsAdapter->szAdapterName, (char*)*ptcWord, MAX_LEN_SOCKRAW_ADAPTER_NAME - 1);

#if (defined DISABLE_FORCE_BROADCAST)
    /* Do not overwrite destination in frame with FF:FF:FF:FF:FF:FF, needed for EAP. */
    pLinkParmsAdapter->bDisableForceBroadcast = EC_TRUE;
#endif

    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLSOCKRAW */

#if (defined INCLUDE_EMLLWINPCAP)
/***************************************************************************************************/
/**
\brief  Try to create WinPcap link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineWinPcap(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
                                                    EC_T_LINK_PARMS** ppLinkParms)
{
EC_T_DWORD dwRetVal = EC_E_ERROR;
EC_T_LINK_PARMS_WINPCAP* pLinkParmsAdapter = EC_NULL;

    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-winpcap") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_WINPCAP*)OsMalloc(sizeof(EC_T_LINK_PARMS_WINPCAP));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_WINPCAP));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_WINPCAP, sizeof(EC_T_LINK_PARMS_WINPCAP), EC_LINK_PARMS_IDENT_WINPCAP, 1, EcLinkMode_POLLING);

    /* parse the specified IP address */
    (*ptcWord) = OsStrtok(EC_NULL, ".");
    if (((*ptcWord) == EC_NULL) || (OsStrncmp( (*ptcWord), "-", 1) == 0))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineWinPcap: IP address missing!\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    if (!ParseIpAddress(*ptcWord, pLinkParmsAdapter->abyIpAddress))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineWinPcap: Error parsing IP address!\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* get mode */
    if (!ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineWinPcap: Error parsing LinkMode!\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLWINPCAP */

#if (defined INCLUDE_EMLLUDP)
/***************************************************************************************************/
/**
\brief  Try to create UDP link layer parameters according to current command line parsing

\return EC_E_NOERROR     if link layer parameters was created
        EC_E_NOTFOUND    if command line was not matching
        EC_E_INVALIDPARM if syntax error
*/
static EC_T_DWORD CreateLinkParmsFromCmdLineUdp(EC_T_CHAR** ptcWord, EC_T_CHAR** lpCmdLine, EC_T_CHAR* tcStorage, EC_T_BOOL* pbGetNextWord,
    EC_T_LINK_PARMS** ppLinkParms)
{
    EC_T_DWORD              dwRetVal          = EC_E_ERROR;
    EC_T_LINK_PARMS_UDP*    pLinkParmsAdapter = EC_NULL;
    /*EC_T_DWORD              dwTmp             = 0;*/

    EC_UNREFPARM(lpCmdLine);
    EC_UNREFPARM(tcStorage);
    EC_UNREFPARM(pbGetNextWord);

    /* check for matching adapter */
    if (OsStricmp((*ptcWord), "-udp") != 0)
    {
        dwRetVal = EC_E_NOTFOUND;
        goto Exit;
    }
    /* alloc adapter specific link parms */
    pLinkParmsAdapter = (EC_T_LINK_PARMS_UDP*)OsMalloc(sizeof(EC_T_LINK_PARMS_UDP));
    if (EC_NULL == pLinkParmsAdapter)
    {
        dwRetVal = EC_E_NOMEMORY;
        goto Exit;
    }
    OsMemset(pLinkParmsAdapter, 0, sizeof(EC_T_LINK_PARMS_UDP));
    LinkParmsInit(&pLinkParmsAdapter->linkParms, EC_LINK_PARMS_SIGNATURE_UDP, sizeof(EC_T_LINK_PARMS_UDP), EC_LINK_PARMS_IDENT_UDP, 1, EcLinkMode_POLLING);

    /* parse the specified IP address */
    (*ptcWord) = OsStrtok(EC_NULL, ".");
    if (((*ptcWord) == EC_NULL) || (OsStrncmp( (*ptcWord), "-", 1) == 0))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineUdp: IP address missing!\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    if (!ParseIpAddress(*ptcWord, pLinkParmsAdapter->abyIpAddress))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineUdp: Error parsing IP address!\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    if (!ParseLinkMode(ptcWord, lpCmdLine, tcStorage, &pLinkParmsAdapter->linkParms.eLinkMode))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineUdp: Error parsing link mode!\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }
    /* TODO Remove?*/
    pLinkParmsAdapter->wPort = 0x88a4;

    /*
    if (!ParseDword(ptcWord, lpCmdLine, tcStorage, &dwTmp))
    {
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineUdp: Error parsing port number!\n"));
        dwRetVal = EC_E_INVALIDPARM;
        goto Exit;
    }

    if ((dwTmp < 0xFFFF) && (dwTmp > 0))
    {
        pLinkParmsAdapter->wPort = (EC_T_WORD)dwTmp;
    }
    else
    {
        pLinkParmsAdapter->wPort = 0x88a4;
        EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "CreateLinkParmsFromCmdLineUdp: Wrong port number! Defaulting to %X\n", pLinkParmsAdapter->wPort));
    }
    */
    /* no errors */
    *ppLinkParms = &pLinkParmsAdapter->linkParms;
    dwRetVal = EC_E_NOERROR;

Exit:
    if (EC_E_NOERROR != dwRetVal)
    {
        SafeOsFree(pLinkParmsAdapter);
    }
    return dwRetVal;
}
#endif /* INCLUDE_EMLLUDP */

/***************************************************************************************************/
/**
\brief  Select Link Layer.

This function checks whether parameter portion is a LinkLayer information and processes it
\return EC_TRUE if parameter is LinkLayer Portion, EC_FALSE otherwise.
*/
EC_T_DWORD CreateLinkParmsFromCmdLine
(   EC_T_CHAR**     ptcWord,
    EC_T_CHAR**     lpCmdLine,
    EC_T_CHAR*      tcStorage,
    EC_T_BOOL*      pbGetNextWord,  /* [out]  Shows that next parameter should be read or not */
    EC_T_LINK_PARMS** ppLinkParms
#if defined(INCLUDE_TTS)
    , EC_T_DWORD* pdwTtsBusCycleUsec /* [out] TTS Bus Cycle overrides original one when TTS is used */
    , EC_T_VOID** pvvTtsEvent        /* [out] TTS Cycle event. Should override original one when TTS is used */
#endif
)
{
EC_T_DWORD dwRetVal = EC_E_NOTFOUND;

#if (defined INCLUDE_EMLLALTERATSE)
    dwRetVal = CreateLinkParmsFromCmdLineAlteraTse(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms );
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLANTAIOS)
    dwRetVal = CreateLinkParmsFromCmdLineAntaios(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLCCAT)
    dwRetVal = CreateLinkParmsFromCmdLineCCAT(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLCPSW)
    dwRetVal = CreateLinkParmsFromCmdLineCPSW(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLDUMMY)
    dwRetVal = CreateLinkParmsFromCmdLineDummy(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLDW3504)
    dwRetVal = CreateLinkParmsFromCmdLineDW3504(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLEG20T)
    dwRetVal = CreateLinkParmsFromCmdLineEG20T(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLEMAC)
    dwRetVal = CreateLinkParmsFromCmdLineEMAC(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLETSEC)
    dwRetVal = CreateLinkParmsFromCmdLineETSEC(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLFSLFEC)
    dwRetVal = CreateLinkParmsFromCmdLineFslFec(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLGEM)
    dwRetVal = CreateLinkParmsFromCmdLineGEM(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLI8254X)
    dwRetVal = CreateLinkParmsFromCmdLineI8254x(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLI8255X)
    dwRetVal = CreateLinkParmsFromCmdLineI8255x(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLICSS)
    dwRetVal = CreateLinkParmsFromCmdLineICSS(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms
#if defined(INCLUDE_TTS)
                                              ,pdwTtsBusCycleUsec
                                              ,pvvTtsEvent
#endif
                                              );
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLL9218I)
    dwRetVal = CreateLinkParmsFromCmdLineL9218i(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLNDISUIO)
    dwRetVal = CreateLinkParmsFromCmdLineNDISUIO(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLR6040)
    dwRetVal = CreateLinkParmsFromCmdLineR6040(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLRIN32M3)
    dwRetVal = CreateLinkParmsFromCmdLineRIN32M3(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLRTL8139)
    dwRetVal = CreateLinkParmsFromCmdLineRTL8139(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLRTL8169)
    dwRetVal = CreateLinkParmsFromCmdLineRTL8169(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLRZT1)
    dwRetVal = CreateLinkParmsFromCmdLineRZT1(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLSHETH)
    dwRetVal = CreateLinkParmsFromCmdLineSHEth(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLSNARF)
    dwRetVal = CreateLinkParmsFromCmdLineSnarf(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLSOCKRAW)
    dwRetVal = CreateLinkParmsFromCmdLineSockRaw(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLWINPCAP)
    dwRetVal = CreateLinkParmsFromCmdLineWinPcap(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif
#if (defined INCLUDE_EMLLUDP)
    dwRetVal = CreateLinkParmsFromCmdLineUdp(ptcWord, lpCmdLine, tcStorage, pbGetNextWord, ppLinkParms);
    if (EC_E_NOTFOUND != dwRetVal)
    {
        goto Exit;
    }
#endif

Exit:
    return dwRetVal;
}

EC_T_VOID ShowLinkLayerSyntax1(EC_T_VOID)
{
    const EC_T_CHAR *syntaxLinkLayer = ""
#if (defined INCLUDE_EMLLALTERATSE)
        " [-alteratse Instance Mode]"
#endif
#if (defined INCLUDE_EMLLANTAIOS)
        " [-antaios]"
#endif
#if (defined INCLUDE_EMLLCCAT)
        " [-ccat Instance Mode [Interface]]"
#endif
#if (defined INCLUDE_EMLLCPSW)
        " [-cpsw Instance Mode PortPriority MasterFlag [RefBoard CpswType PhyAddress PhyConnectionMode NotUseDmaBuffers]]"
#endif
#if (defined INCLUDE_EMLLDUMMY)
        " [-dummy Instance Mode]"
#endif
#if (defined INCLUDE_EMLLDW3504)
        " [-dw3504 Instance Mode PhyAddress]"
#endif
#if (defined INCLUDE_EMLLEG20T)
        " [-eg20t Instance Mode]"
#endif
#if (defined INCLUDE_EMLLEMAC)
        " [-emac Instance Mode [RefBoard]]"
#endif
#if (defined INCLUDE_EMLLETSEC)
        " [-fsletsec Instance Mode]"
#endif
#if (defined INCLUDE_EMLLFSLFEC)
        " [-fslfec Instance Mode RefBoard [FecType PhyInterface [PhyAddress]]]"
#endif
#if (defined INCLUDE_EMLLGEM)
        " [-gem Instance Mode [RefBoard [PhyAddress [PhyConnectionMode [UseGmiiToRgmii [GmiiToRgmiiPort [SocType]]]]]]]"
#endif
#if (defined INCLUDE_EMLLI8254X)
        " [-i8254x Instance Mode]"
#endif
#if (defined INCLUDE_EMLLI8255X)
        " [-i8255x Instance Mode]"
#endif
#if (defined INCLUDE_EMLLICSS)
        " [-icss Instance Mode MasterFlag RefBoard [tts tts_config_time]]"
#endif
#if (defined INCLUDE_EMLLL9218I)
        " [-l9218i Mode]"
#endif
#if (defined INCLUDE_EMLLNDISUIO)
        " [-ndisuio Adapter]"
#endif
#if (defined INCLUDE_EMLLR6040)
        " [-r6040 Instance Mode]"
#endif
#if (defined INCLUDE_EMLLRIN32M3)
        " [-rin32m3]"
#endif
#if (defined INCLUDE_EMLLRTL8139)
        " [-rtl8139 Instance Mode]"
#endif
#if (defined INCLUDE_EMLLRTL8169)
        " [-rtl8169 Instance Mode]"
#endif
#if (defined INCLUDE_EMLLRZT1)
        " [-rzt1 Instance]"
#endif
#if (defined INCLUDE_EMLLSHETH)
        " [-sheth Instance Mode RefBoard]"
#endif
#if (defined INCLUDE_EMLLSNARF)
        " [-snarf AdapterName]"
#endif
#if (defined INCLUDE_EMLLSOCKRAW)
        " [-SockRaw device]"
#endif
#if (defined INCLUDE_EMLLWINPCAP)
        " [-winpcap IpAddress Mode]"
#endif
#if (defined INCLUDE_EMLLUDP)
        " [-udp IpAddress]"
#endif
        "\n";
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, syntaxLinkLayer));
}

EC_T_VOID ShowLinkLayerSyntax2(EC_T_VOID)
{
#if (defined INCLUDE_EMLLALTERATSE)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -alteratse        Link layer = Lenze/Intel FPGA TSE\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Port Instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLANTAIOS)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -antaios            Link layer = Antaios link layer device statically loaded\n"));
#endif
#if (defined INCLUDE_EMLLCCAT)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -ccat             Link layer = Beckhoff CCAT\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Interface       PCI | EIM (optional)\n"));
#endif
#if (defined INCLUDE_EMLLCPSW)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -cpsw             Link layer = Texas Instruments Common Platform Switch (CPSW)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (port) 1 P1, 2 P2\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     PortPriority    Low priority (0) or high priority (1)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     MasterFlag      (m) Master (Initialize Switch), (s) Slave\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     RefBoard:       custom | bone | am437x-idk | am572x-idk | 387X_evm (optional)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "      if custom       CpswType: am33XX | am437X | am57X | am387X\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "      if custom       PhyAddress 0 .. 31\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "      if custom       PhyConnection GMII (0) or RGMII (1)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "      if custom       NotUseDmaBuffers FALSE (0) or TRUE (1)\n"));
#endif
#if (defined INCLUDE_EMLLDUMMY)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -dummy            Link layer = Dummy\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLDW3504)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -dw3504           Link layer = Synopsys DesignWare 3504-0 Universal 10/100/1000 Ethernet MAC (DW3504)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance 1 for emac0, 2 for emac1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     RefBoard:       socrates | lces1 | rd55up06 | r12ccpu | rzn1 | custom\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        DW3504Type: cycloneV | lces1 \n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        PhyInterface: fixed | mii | rmii | gmii | sgmii | rgmii\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        PhyAddress: 0 .. 31, default 0\n"));
#endif
#if (defined INCLUDE_EMLLEG20T)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -eg20t            Link layer = EG20T Gigabit Ethernet Controller\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLEMAC)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -emac             Link layer = Xilinx LogiCORE IP XPS EMAC\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (must be 1)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     RefBoard:       MC2002E | custom \n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        RegisterBase: register base address (hex value)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        RegisterLength: register length (hex value)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        NotUseDmaBuffers: FALSE (0) or TRUE (1)\n"));
#endif
#if (defined INCLUDE_EMLLETSEC)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -fsletsec         Link layer = Freescale TSEC / eTSEC V1 / eTSEC V2 (VeTSEC)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLFSLFEC)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -fslfec           Link layer = Freescale FEC\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     RefBoard:       custom | mars | sabrelite | sabresd | imx28evk | topaz | imxceetul2\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        FecType: imx25 | imx28 | imx53 | imx6 | vf6\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        PhyInterface: fixed | mii | rmii | gmii | sgmii | rgmii\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        PhyAddress: 0 .. 31, default 0\n"));
#endif
#if (defined INCLUDE_EMLLGEM)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -gem              Link layer = Xilinx Zynq-7000/Ultrascale (GEM)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance for GEM, GEM0 == 1...\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     RefBoard:       custom | zc702 | zedboard | microzed | zcu102 | zcu104\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        PhyAddress: 0 .. 31\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        PhyConnectionMode: MIO (0) or EMIO (1)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        UseGmiiToRgmii: Use Xilinx GmiiToRgmii converter TRUE (1) or FALSE (0)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        GmiiToRgmiiPort: GmiiToRgmii converter PHY address 0 .. 31\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if custom        GEM type: zynq7000 or ultrascale\n"));
#endif
#if (defined INCLUDE_EMLLI8254X)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -i8254x           Link layer = Intel 8254x\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLI8255X)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -i8255x           Link layer = Intel 8255x\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLICSS)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -icss             Link layer = Texas Instruments PRUICSS\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        ICSS Port (100 Mbit) 1 .. 4\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     MasterFlag      (m) Master (Initialize whole PRUSS), (s) Slave\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     RefBoard        am572x-idk | am571x-idk | am3359\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     tts             OPTIONAL - activate TTS mode\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     if tts tts_conf_time: 50 .. 1000 EtherCAT Master config time (usec)\n"));
#endif
#if (defined INCLUDE_EMLLL9218I)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -l9218i           Link layer = SMSC LAN9218i/LAN9221\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1)\n"));
#endif
#if (defined INCLUDE_EMLLNDISUIO)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -ndisuio          Link layer = NdisUio\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Adapter         Device name (registry), ex. PCI\\RTL81391\n"));
#endif
#if (defined INCLUDE_EMLLR6040)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -r6040            Link layer = R6040\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLRIN32M3)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -rin32m3          Link layer = RIN32M3\n"));
#endif
#if (defined INCLUDE_EMLLRTL8139)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -rtl8139          Link layer = Realtek RTL8139\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLRTL8169)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -rtl8169          Link layer = Realtek RTL8169 / RTL8168 / RTL8111\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
#if (defined INCLUDE_EMLLRZT1)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -rzt1             Link layer = RZT1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance 1 for Port 0 or 2 for Port 1\n"));
#endif
#if (defined INCLUDE_EMLLSHETH)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -sheth            Link layer = Super H Etherner controller\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Instance        Device instance (1=first), ex. 1\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     RefBoard:       a800eva | rzg1e\n"));
#endif
#if (defined INCLUDE_EMLLSNARF)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -snarf            Link layer = SNARF link layer device statically loaded\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "    AdapterName      Adapter name, ex. fei0\n"));
#endif
#if (defined INCLUDE_EMLLSOCKRAW)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -SockRaw          Link layer = raw socket\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     device          network device (e.g. eth1)\n"));
#endif
#if (defined INCLUDE_EMLLUDP)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -udp              Link layer = UDP\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     IpAddress       IP address of network adapter card, ex. 192.168.157.2\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "                     NPF only: 255.255.255.x, x = network adapter number (1,2,...)\n"));
#endif
#if (defined INCLUDE_EMLLWINPCAP)
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "   -winpcap          Link layer = WinPcap/NPF\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     IpAddress       IP address of network adapter card, ex. 192.168.157.2\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "                     NPF only: 255.255.255.x, x = network adapter number (1,2,...)\n"));
    EcLogMsg(EC_LOG_LEVEL_ERROR, (pEcLogContext, EC_LOG_LEVEL_ERROR, "     Mode            Interrupt (0) or Polling (1) mode\n"));
#endif
}

#if (defined INCLUDE_EMLL_STATIC_LIBRARY)
#include "EcLink.h"

EC_PF_LLREGISTER DemoGetLinkLayerRegFunc(EC_T_CHAR* szDriverIdent)
{
EC_PF_LLREGISTER pfLlRegister = EC_NULL;

#if (defined INCLUDE_EMLLANTAIOS)
    if (0 == OsStrcmp(EC_LINK_PARMS_IDENT_ANTAIOS, szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterAntaios;
    } else
#endif
#if (defined INCLUDE_EMLLCCAT)
    if (0 == OsStrcmp(EC_LINK_PARMS_IDENT_CCAT, szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterCCAT;
    } else
#endif
#if (defined INCLUDE_EMLLCPSW)
    if (0 == OsStrcmp(EC_LINK_PARMS_IDENT_CPSW, szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterCPSW;
    } else
#endif
#if (defined INCLUDE_EMLLDW3504)
    if (0 == OsStrcmp(EC_LINK_PARMS_IDENT_DW3504, szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterDW3504;
    } else
#endif
#if (defined INCLUDE_EMLLETSEC)
    if (0 == OsStrcmp("ETSEC", szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterETSEC;
    } else
#endif
#if (defined INCLUDE_EMLLFSLFEC)
    if (0 == OsStrcmp(EC_LINK_PARMS_IDENT_FSLFEC, szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterFslFec;
    } else
#endif
#if (defined INCLUDE_EMLLGEM)
    if (0 == OsStrcmp(EC_LINK_PARMS_IDENT_GEM, szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterGEM;
	} else
#endif
#if (defined INCLUDE_EMLLI8254X)
    if (0 == OsStrcmp("I8254x", szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterI8254x;
    } else
#endif
#if (defined INCLUDE_EMLLI8255X)
    if (0 == OsStrcmp("I8255x", szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterI8255x;
    } else
#endif
#if (defined INCLUDE_EMLLICSS)
    if (0 == OsStrcmp(EC_LINK_PARMS_IDENT_ICSS, szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterICSS;
    } else
#endif
#if (defined INCLUDE_EMLLRTL8139)
    if (0 == OsStrcmp("RTL8139", szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterRTL8139;
    } else
#endif
#if (defined INCLUDE_EMLLRTL8169)
    if (0 == OsStrcmp("RTL8169", szDriverIdent))
    {
        pfLlRegister = (EC_PF_LLREGISTER)emllRegisterRTL8169;
    } else
#endif
    {
        pfLlRegister = EC_NULL;
    }
    return pfLlRegister;
}
#endif

/*-END OF SOURCE FILE--------------------------------------------------------*/
