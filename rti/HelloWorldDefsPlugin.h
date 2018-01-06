

/*
WARNING: THIS FILE IS AUTO-GENERATED. DO NOT MODIFY.

This file was generated from HelloWorldDefs.idl using "rtiddsgen".
The rtiddsgen tool is part of the RTI Connext distribution.
For more information, type 'rtiddsgen -help' at a command shell
or consult the RTI Connext manual.
*/

#ifndef HelloWorldDefsPlugin_192132248_h
#define HelloWorldDefsPlugin_192132248_h

#include "HelloWorldDefs.h"

struct RTICdrStream;

#ifndef pres_typePlugin_h
#include "pres/pres_typePlugin.h"
#endif

#if (defined(RTI_WIN32) || defined (RTI_WINCE)) && defined(NDDS_USER_DLL_EXPORT)
/* If the code is building on Windows, start exporting symbols.
*/
#undef NDDSUSERDllExport
#define NDDSUSERDllExport __declspec(dllexport)
#endif

extern "C" {

    #define HelloWorldIdlPlugin_get_sample PRESTypePluginDefaultEndpointData_getSample 
    #define HelloWorldIdlPlugin_get_buffer PRESTypePluginDefaultEndpointData_getBuffer 
    #define HelloWorldIdlPlugin_return_buffer PRESTypePluginDefaultEndpointData_returnBuffer 

    #define HelloWorldIdlPlugin_create_sample PRESTypePluginDefaultEndpointData_createSample 
    #define HelloWorldIdlPlugin_destroy_sample PRESTypePluginDefaultEndpointData_deleteSample 

    /* --------------------------------------------------------------------------------------
    Support functions:
    * -------------------------------------------------------------------------------------- */

    NDDSUSERDllExport extern HelloWorldIdl*
    HelloWorldIdlPluginSupport_create_data_w_params(
        const struct DDS_TypeAllocationParams_t * alloc_params);

    NDDSUSERDllExport extern HelloWorldIdl*
    HelloWorldIdlPluginSupport_create_data_ex(RTIBool allocate_pointers);

    NDDSUSERDllExport extern HelloWorldIdl*
    HelloWorldIdlPluginSupport_create_data(void);

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPluginSupport_copy_data(
        HelloWorldIdl *out,
        const HelloWorldIdl *in);

    NDDSUSERDllExport extern void 
    HelloWorldIdlPluginSupport_destroy_data_w_params(
        HelloWorldIdl *sample,
        const struct DDS_TypeDeallocationParams_t * dealloc_params);

    NDDSUSERDllExport extern void 
    HelloWorldIdlPluginSupport_destroy_data_ex(
        HelloWorldIdl *sample,RTIBool deallocate_pointers);

    NDDSUSERDllExport extern void 
    HelloWorldIdlPluginSupport_destroy_data(
        HelloWorldIdl *sample);

    NDDSUSERDllExport extern void 
    HelloWorldIdlPluginSupport_print_data(
        const HelloWorldIdl *sample,
        const char *desc,
        unsigned int indent);

    /* ----------------------------------------------------------------------------
    Callback functions:
    * ---------------------------------------------------------------------------- */

    NDDSUSERDllExport extern PRESTypePluginParticipantData 
    HelloWorldIdlPlugin_on_participant_attached(
        void *registration_data, 
        const struct PRESTypePluginParticipantInfo *participant_info,
        RTIBool top_level_registration, 
        void *container_plugin_context,
        RTICdrTypeCode *typeCode);

    NDDSUSERDllExport extern void 
    HelloWorldIdlPlugin_on_participant_detached(
        PRESTypePluginParticipantData participant_data);

    NDDSUSERDllExport extern PRESTypePluginEndpointData 
    HelloWorldIdlPlugin_on_endpoint_attached(
        PRESTypePluginParticipantData participant_data,
        const struct PRESTypePluginEndpointInfo *endpoint_info,
        RTIBool top_level_registration, 
        void *container_plugin_context);

    NDDSUSERDllExport extern void 
    HelloWorldIdlPlugin_on_endpoint_detached(
        PRESTypePluginEndpointData endpoint_data);

    NDDSUSERDllExport extern void    
    HelloWorldIdlPlugin_return_sample(
        PRESTypePluginEndpointData endpoint_data,
        HelloWorldIdl *sample,
        void *handle);    

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPlugin_copy_sample(
        PRESTypePluginEndpointData endpoint_data,
        HelloWorldIdl *out,
        const HelloWorldIdl *in);

    /* ----------------------------------------------------------------------------
    (De)Serialize functions:
    * ------------------------------------------------------------------------- */

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPlugin_serialize(
        PRESTypePluginEndpointData endpoint_data,
        const HelloWorldIdl *sample,
        struct RTICdrStream *stream, 
        RTIBool serialize_encapsulation,
        RTIEncapsulationId encapsulation_id,
        RTIBool serialize_sample, 
        void *endpoint_plugin_qos);

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPlugin_deserialize_sample(
        PRESTypePluginEndpointData endpoint_data,
        HelloWorldIdl *sample, 
        struct RTICdrStream *stream,
        RTIBool deserialize_encapsulation,
        RTIBool deserialize_sample, 
        void *endpoint_plugin_qos);

    NDDSUSERDllExport extern RTIBool
    HelloWorldIdlPlugin_serialize_to_cdr_buffer(
        char * buffer,
        unsigned int * length,
        const HelloWorldIdl *sample); 

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPlugin_deserialize(
        PRESTypePluginEndpointData endpoint_data,
        HelloWorldIdl **sample, 
        RTIBool * drop_sample,
        struct RTICdrStream *stream,
        RTIBool deserialize_encapsulation,
        RTIBool deserialize_sample, 
        void *endpoint_plugin_qos);

    NDDSUSERDllExport extern RTIBool
    HelloWorldIdlPlugin_deserialize_from_cdr_buffer(
        HelloWorldIdl *sample,
        const char * buffer,
        unsigned int length);    
    NDDSUSERDllExport extern DDS_ReturnCode_t
    HelloWorldIdlPlugin_data_to_string(
        const HelloWorldIdl *sample,
        char *str,
        DDS_UnsignedLong *str_size, 
        const struct DDS_PrintFormatProperty *property);    

    NDDSUSERDllExport extern RTIBool
    HelloWorldIdlPlugin_skip(
        PRESTypePluginEndpointData endpoint_data,
        struct RTICdrStream *stream, 
        RTIBool skip_encapsulation,  
        RTIBool skip_sample, 
        void *endpoint_plugin_qos);

    NDDSUSERDllExport extern unsigned int 
    HelloWorldIdlPlugin_get_serialized_sample_max_size_ex(
        PRESTypePluginEndpointData endpoint_data,
        RTIBool * overflow,
        RTIBool include_encapsulation,
        RTIEncapsulationId encapsulation_id,
        unsigned int current_alignment);    

    NDDSUSERDllExport extern unsigned int 
    HelloWorldIdlPlugin_get_serialized_sample_max_size(
        PRESTypePluginEndpointData endpoint_data,
        RTIBool include_encapsulation,
        RTIEncapsulationId encapsulation_id,
        unsigned int current_alignment);

    NDDSUSERDllExport extern unsigned int 
    HelloWorldIdlPlugin_get_serialized_sample_min_size(
        PRESTypePluginEndpointData endpoint_data,
        RTIBool include_encapsulation,
        RTIEncapsulationId encapsulation_id,
        unsigned int current_alignment);

    NDDSUSERDllExport extern unsigned int
    HelloWorldIdlPlugin_get_serialized_sample_size(
        PRESTypePluginEndpointData endpoint_data,
        RTIBool include_encapsulation,
        RTIEncapsulationId encapsulation_id,
        unsigned int current_alignment,
        const HelloWorldIdl * sample);

    /* --------------------------------------------------------------------------------------
    Key Management functions:
    * -------------------------------------------------------------------------------------- */
    NDDSUSERDllExport extern PRESTypePluginKeyKind 
    HelloWorldIdlPlugin_get_key_kind(void);

    NDDSUSERDllExport extern unsigned int 
    HelloWorldIdlPlugin_get_serialized_key_max_size_ex(
        PRESTypePluginEndpointData endpoint_data,
        RTIBool * overflow,
        RTIBool include_encapsulation,
        RTIEncapsulationId encapsulation_id,
        unsigned int current_alignment);

    NDDSUSERDllExport extern unsigned int 
    HelloWorldIdlPlugin_get_serialized_key_max_size(
        PRESTypePluginEndpointData endpoint_data,
        RTIBool include_encapsulation,
        RTIEncapsulationId encapsulation_id,
        unsigned int current_alignment);

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPlugin_serialize_key(
        PRESTypePluginEndpointData endpoint_data,
        const HelloWorldIdl *sample,
        struct RTICdrStream *stream,
        RTIBool serialize_encapsulation,
        RTIEncapsulationId encapsulation_id,
        RTIBool serialize_key,
        void *endpoint_plugin_qos);

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPlugin_deserialize_key_sample(
        PRESTypePluginEndpointData endpoint_data,
        HelloWorldIdl * sample,
        struct RTICdrStream *stream,
        RTIBool deserialize_encapsulation,
        RTIBool deserialize_key,
        void *endpoint_plugin_qos);

    NDDSUSERDllExport extern RTIBool 
    HelloWorldIdlPlugin_deserialize_key(
        PRESTypePluginEndpointData endpoint_data,
        HelloWorldIdl ** sample,
        RTIBool * drop_sample,
        struct RTICdrStream *stream,
        RTIBool deserialize_encapsulation,
        RTIBool deserialize_key,
        void *endpoint_plugin_qos);

    NDDSUSERDllExport extern RTIBool
    HelloWorldIdlPlugin_serialized_sample_to_key(
        PRESTypePluginEndpointData endpoint_data,
        HelloWorldIdl *sample,
        struct RTICdrStream *stream, 
        RTIBool deserialize_encapsulation,  
        RTIBool deserialize_key, 
        void *endpoint_plugin_qos);

    /* Plugin Functions */
    NDDSUSERDllExport extern struct PRESTypePlugin*
    HelloWorldIdlPlugin_new(void);

    NDDSUSERDllExport extern void
    HelloWorldIdlPlugin_delete(struct PRESTypePlugin *);

}

#if (defined(RTI_WIN32) || defined (RTI_WINCE)) && defined(NDDS_USER_DLL_EXPORT)
/* If the code is building on Windows, stop exporting symbols.
*/
#undef NDDSUSERDllExport
#define NDDSUSERDllExport
#endif

#endif /* HelloWorldDefsPlugin_192132248_h */

