/*-----------------------------------------------------------------------------
 * CList.h                  header file
 * Copyright                acontis technologies GmbH, Weingarten, Germany
 * Response                 Stefan Zintgraf
 * Description              
 *---------------------------------------------------------------------------*/

#ifndef INC_LIST
#define INC_LIST

#ifdef _MSC_VER
#pragma warning (disable: 4710)
#endif

/*-INCLUDES------------------------------------------------------------------*/

/*-EXTERNALS-----------------------------------------------------------------*/

/*-TYPEDEFS/ENUMS------------------------------------------------------------*/

template<class TYPE>
    class CNode
    {
    public:
        CNode(){};
        ~CNode(){};
        CNode* pNext;
        CNode* pPrev;
        TYPE   data;
    };

template<class TYPE, class ARG_TYPE> 
class CList
{
    
/*-EMBEDDED CLASSES----------------------------------------------------------*/
public:
    struct CNode
    {
        CNode* pNext;
        CNode* pPrev;
        TYPE    data;
    };

protected:
private:
    
    
/*-CONSTRUCTORS/DESTRUCTORS--------------------------------------------------*/
public:
    /* -constructors/destructors/initialization */
    CList( ) 
    {   
        m_pFirstNode = EC_NULL;
        m_pLastNode = EC_NULL;
        m_nNumNodes = 0;
    };
    virtual ~CList();
protected:
private:
    
    
/*-ATTRIBUTES----------------------------------------------------------------*/
public:

protected:
    CNode*  m_pFirstNode;
    CNode*  m_pLastNode;
    EC_T_INT   m_nNumNodes;
private:

/*-METHODS-------------------------------------------------------------------*/
public:
    /* count of elements */
    EC_T_INT GetCount( void ) const    { return m_nNumNodes; }
    EC_T_BOOL IsEmpty( void ) const    { return m_nNumNodes == 0; }

    /* get head or tail (and remove it) - don't call on empty list ! */
    TYPE RemoveTail();

    /* add before head or after tail */
    EC_T_VOID AddHead( ARG_TYPE newElement );
    EC_T_VOID AddTail( ARG_TYPE newElement );

    /* iteration */
    CNode* GetFirstNode() const    { return m_pFirstNode; }

    TYPE GetNext(CNode*& pNode) const; /* return *Position++ */

    /* getting/modifying an element at a given position */
    TYPE& GetAt(CNode* position);

    void RemoveAt(CNode* pOldNode);
    EC_T_BOOL FindAndDelete(ARG_TYPE ElementToFind );
    EC_T_VOID Find( CNode*& pNode, ARG_TYPE ElementToFind );

protected:
private:
};

/*-INLINE METHODS------------------------------------------------------------*/

/********************************************************************************/
/** \brief AddHead
*
* \return
*/
template<class TYPE, class ARG_TYPE>
CList<TYPE, ARG_TYPE>::~CList()
{
	CNode* cur = EC_NULL;
	
	while(m_pFirstNode)
	{
		cur = m_pFirstNode;
		m_pFirstNode = m_pFirstNode->pNext;

		SafeDelete(cur);
	};
	
}

/********************************************************************************/
/** \brief AddHead
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
EC_T_VOID CList<TYPE, ARG_TYPE>::AddHead(ARG_TYPE newElement)
{
    CNode* pNewNode = EC_NEW(CNode);
    if (EC_NULL != pNewNode)
    {
        pNewNode->data = newElement;
        if( m_pFirstNode == EC_NULL )
        {
            m_pFirstNode = m_pLastNode = pNewNode;
            pNewNode->pNext = EC_NULL;
            pNewNode->pPrev = EC_NULL;
        }
        else
        {
            pNewNode->pNext = m_pFirstNode;
            m_pFirstNode->pPrev = pNewNode;
            m_pFirstNode = pNewNode;
            pNewNode->pPrev = EC_NULL;
        }
        m_nNumNodes++;
    }
    return;
}

/********************************************************************************/
/** \brief AddTail
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
EC_T_VOID CList<TYPE, ARG_TYPE>::AddTail(ARG_TYPE newElement)
{
    CNode* pNewNode = EC_NEW(CNode);
    if (EC_NULL != pNewNode)
    {
        pNewNode->data = newElement;
        if( m_pLastNode == EC_NULL )
        {
            m_pFirstNode = m_pLastNode = pNewNode;
            pNewNode->pNext = EC_NULL;
            pNewNode->pPrev = EC_NULL;
        }
        else
        {
            pNewNode->pPrev = m_pLastNode;
            m_pLastNode->pNext = pNewNode;
            m_pLastNode = pNewNode;
            pNewNode->pNext = EC_NULL;
        }
        m_nNumNodes++;
    }
    return;
}

/********************************************************************************/
/** \brief RemoveTail
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
TYPE CList<TYPE, ARG_TYPE>::RemoveTail()
{
    CNode* pOldNode = m_pLastNode;
    if( m_nNumNodes == 1 )
    {
        /* DBG_ASSERT( m_pFirstNode == m_pLastNode ); */
        m_pFirstNode = EC_NULL;
        m_pLastNode = EC_NULL;
    }
    else
    {
        m_pLastNode = pOldNode->pPrev;
        m_pLastNode->pNext = EC_NULL;
    }
    m_nNumNodes--;
    TYPE data = pOldNode->data;
    SafeDelete(pOldNode);
    
    return data; 
}

/********************************************************************************/
/** \brief GetAt
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
TYPE& CList<TYPE, ARG_TYPE>::GetAt(CNode* pCurNode)
{
    return pCurNode->data;
}

/********************************************************************************/
/** \brief RemoveAt
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
void CList<TYPE, ARG_TYPE>::RemoveAt(CNode* pOldNode)
{
    /* remove pOldNode from list */
    if (pOldNode == m_pFirstNode)
    {
        m_pFirstNode = pOldNode->pNext;
    }
    else
    {
        pOldNode->pPrev->pNext = pOldNode->pNext;
    }
    if (pOldNode == m_pLastNode)
    {
        m_pLastNode = pOldNode->pPrev;
    }
    else
    {
        pOldNode->pNext->pPrev = pOldNode->pPrev;
    }
    SafeDelete(pOldNode);
    m_nNumNodes--;
}

/********************************************************************************/
/** \brief FindAndDelete
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
EC_T_BOOL CList<TYPE, ARG_TYPE>::FindAndDelete(ARG_TYPE ElementToFind )
{
EC_T_BOOL bFound = EC_FALSE;

    if( m_pFirstNode != EC_NULL )
    {
        CNode* pNode = m_pFirstNode;
        for( EC_T_INT nListIndex = 0; nListIndex < m_nNumNodes; nListIndex++ )
        {
            if( pNode->data == ElementToFind )
            {
                bFound = EC_TRUE;
                break;
            }
            pNode = (CNode*)pNode->pNext;
        }
        if( bFound )
        {
            RemoveAt( pNode );    /*  remove from list */
        }
    }
    return bFound;
}

/********************************************************************************/
/** \brief Find
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
EC_T_VOID CList<TYPE, ARG_TYPE>::Find(CNode*& pNode, ARG_TYPE ElementToFind )
{
    EC_T_BOOL bFound = EC_FALSE;
    
    pNode = EC_NULL;
    
    if( m_pFirstNode != EC_NULL )
    {
        pNode = m_pFirstNode;
        for( EC_T_INT nListIndex = 0; nListIndex < m_nNumNodes; nListIndex++ )
        {
            if( pNode->data == ElementToFind )
            {
                bFound = EC_TRUE;
                break;
            }
            pNode = (CNode*)pNode->pNext;
        }
        if( !bFound )
        {
            pNode = EC_NULL;
        }
    }

    return;
}

/********************************************************************************/
/** \brief GetNext
*
* \return 
*/
template<class TYPE, class ARG_TYPE>
TYPE CList<TYPE, ARG_TYPE>::GetNext(CNode*& pCurNode) const /* return *Position++ */
{ 
TYPE* pElement;

    CNode* pNode = (CNode*) pCurNode;
    pCurNode = (CNode*) pNode->pNext;
    pElement = &(pNode->data); 
    return *pElement; 
}

#endif /* INC_LIST */


/*-END OF SOURCE FILE--------------------------------------------------------*/
