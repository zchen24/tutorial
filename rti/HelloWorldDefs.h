

/*
WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

This file was generated from HelloWorldDefs.idl using "rtiddsgen".
The rtiddsgen tool is part of the RTI Connext distribution.
For more information, type 'rtiddsgen -help' at a command shell
or consult the RTI Connext manual.
*/

#ifndef HelloWorldDefs_192132248_h
#define HelloWorldDefs_192132248_h

#ifndef NDDS_STANDALONE_TYPE
#ifndef ndds_cpp_h
#include "ndds/ndds_cpp.h"
#endif
#else
#include "ndds_standalone_type.h"
#endif

extern "C" {

    extern const char *HelloWorldIdlTYPENAME;

}

struct HelloWorldIdlSeq;
#ifndef NDDS_STANDALONE_TYPE
class HelloWorldIdlTypeSupport;
class HelloWorldIdlDataWriter;
class HelloWorldIdlDataReader;
#endif

class HelloWorldIdl 
{
  public:
    typedef struct HelloWorldIdlSeq Seq;
    #ifndef NDDS_STANDALONE_TYPE
    typedef HelloWorldIdlTypeSupport TypeSupport;
    typedef HelloWorldIdlDataWriter DataWriter;
    typedef HelloWorldIdlDataReader DataReader;
    #endif

    DDS_Char *   prefix ;
    DDS_Long   sampleId ;
    DDS_OctetSeq  payload ;

};
#if (defined(RTI_WIN32) || defined (RTI_WINCE)) && defined(NDDS_USER_DLL_EXPORT)
/* If the code is building on Windows, start exporting symbols.
*/
#undef NDDSUSERDllExport
#define NDDSUSERDllExport __declspec(dllexport)
#endif

NDDSUSERDllExport DDS_TypeCode* HelloWorldIdl_get_typecode(void); /* Type code */

DDS_SEQUENCE(HelloWorldIdlSeq, HelloWorldIdl);

NDDSUSERDllExport
RTIBool HelloWorldIdl_initialize(
    HelloWorldIdl* self);

NDDSUSERDllExport
RTIBool HelloWorldIdl_initialize_ex(
    HelloWorldIdl* self,RTIBool allocatePointers,RTIBool allocateMemory);

NDDSUSERDllExport
RTIBool HelloWorldIdl_initialize_w_params(
    HelloWorldIdl* self,
    const struct DDS_TypeAllocationParams_t * allocParams);  

NDDSUSERDllExport
void HelloWorldIdl_finalize(
    HelloWorldIdl* self);

NDDSUSERDllExport
void HelloWorldIdl_finalize_ex(
    HelloWorldIdl* self,RTIBool deletePointers);

NDDSUSERDllExport
void HelloWorldIdl_finalize_w_params(
    HelloWorldIdl* self,
    const struct DDS_TypeDeallocationParams_t * deallocParams);

NDDSUSERDllExport
void HelloWorldIdl_finalize_optional_members(
    HelloWorldIdl* self, RTIBool deletePointers);  

NDDSUSERDllExport
RTIBool HelloWorldIdl_copy(
    HelloWorldIdl* dst,
    const HelloWorldIdl* src);

#if (defined(RTI_WIN32) || defined (RTI_WINCE)) && defined(NDDS_USER_DLL_EXPORT)
/* If the code is building on Windows, stop exporting symbols.
*/
#undef NDDSUSERDllExport
#define NDDSUSERDllExport
#endif

#endif /* HelloWorldDefs */

