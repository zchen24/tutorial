/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*

  Author(s):  Zhan Fan Quek   Zihan Chen
  Created on: 2014-04-04

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/


#ifndef _mtsTestComponent_h
#define _mtsTestComponent_h

#include <cisstMultiTask/mtsComponent.h>

class CISST_EXPORT mtsTestComponent : public mtsComponent
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_ALLOW_DEFAULT);

protected:

    // mtsCommandVoid function
    void TestFunction1(void);

    // mtsCommandVoidReturn function
    void TestFunction2(int &result);

    // mtsCommandWrite function
    void TestFunction3(const int &value);

    // mtsCommandWriteReturn function
    void TestFunction4(const int &value, int & result);

    // mtsCommandRead function
    void TestFunction5(int & result) const;

    // mtsCommandQualifiedRead function
    void TestFunction6(const int & qualifier, int & result) const;


public:

    mtsTestComponent(const std::string & componentName, unsigned int numberofActuators);
    ~mtsTestComponent();

    void Configure(void);
    void StartUp(void);
    void CleanUp(void);

};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTestComponent);

#endif
