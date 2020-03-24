/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*

  Author(s):  Zihan Chen
  Created on: 2014-04-04

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _mtsTestComponentClient_h
#define _mtsTestComponentClient_h

#include <cisstMultiTask/mtsComponent.h>

class CISST_EXPORT mtsTestComponentClient : public mtsComponent
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_ALLOW_DEFAULT);

    // We usually group a required interface in a stucture,
    // so it looks more or less like a class
    struct
    {
        mtsFunctionVoid TestFunction1;
        mtsFunctionVoidReturn TestFunction2;
        mtsFunctionWrite TestFunction3;
        mtsFunctionWriteReturn TestFunction4;
        mtsFunctionRead TestFunction5;
        mtsFunctionQualifiedRead TestFunction6;
    } TestComp;

public:

    mtsTestComponentClient(const std::string & componentName);
    ~mtsTestComponentClient();

    void Configure(void);
    void RunClient(void);

};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTestComponentClient);

#endif  // _mtsTestComponentClient_h
