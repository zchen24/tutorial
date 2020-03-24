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


#include "mtsTestComponent.h"
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>


CMN_IMPLEMENT_SERVICES_DERIVED(mtsTestComponent, mtsComponent);


// ====================== Constructors and Destructors ========================
// ----------------------------------------------------------------------------
mtsTestComponent::mtsTestComponent(const std::string & componentName, unsigned int numberofActuators):
    mtsComponent(componentName)
{
    // provide SetDesiredPositions
    mtsInterfaceProvided * interfaceProvided = AddInterfaceProvided("Controller");
    if (interfaceProvided)
    {
        interfaceProvided->AddCommandVoid(&mtsTestComponent::TestFunction1, this, "TestFunction1");
        interfaceProvided->AddCommandVoidReturn(&mtsTestComponent::TestFunction2, this, "TestFunction2");
        interfaceProvided->AddCommandWrite(&mtsTestComponent::TestFunction3, this, "TestFunction3");
        interfaceProvided->AddCommandWriteReturn(&mtsTestComponent::TestFunction4, this, "TestFunction4");
        interfaceProvided->AddCommandRead(&mtsTestComponent::TestFunction5, this, "TestFunction5");
        interfaceProvided->AddCommandQualifiedRead(&mtsTestComponent::TestFunction6, this, "TestFunction6");
    }
}

// ----------------------------------------------------------------------------
mtsTestComponent::~mtsTestComponent()
{

}

// ----------------------------------------------------------------------------  

// =========================== Essential Functions ============================
void mtsTestComponent::Configure(void)
{
}

// ----------------------------------------------------------------------------
void mtsTestComponent::StartUp(void)
{

}

// ----------------------------------------------------------------------------
void mtsTestComponent::CleanUp(void)
{

}

// ----------------------------------------------------------------------------

// ======================= Provided Interface Functions =======================
// mtsCommandVoid function
void mtsTestComponent::TestFunction1(void)
{
    std::cout << "TestFunction1: mtsCommandVoid" << std::endl;
    return;
}
  
// ----------------------------------------------------------------------------
// mtsCommandVoidReturn function
void mtsTestComponent::TestFunction2(int &result)
{

    result = 10;
    std::cout << "In TestFunction2: mtsCommandVoidReturn, result =" << result << std::endl;

    return;
}

// ----------------------------------------------------------------------------  
// mtsCommandWrite function
void mtsTestComponent::TestFunction3(const int &value)
{
    std::cout << "In TestFunction3: mtsCommandWrite, value = " << value << std::endl;
}

// ----------------------------------------------------------------------------  
// mtsCommandWriteReturn function
void mtsTestComponent::TestFunction4(const int &value, int & result)
{

}
  
// ----------------------------------------------------------------------------
// mtsCommandRead function
void mtsTestComponent::TestFunction5(int & result) const
{
   result = 100;
}

// ----------------------------------------------------------------------------  
// mtsCommandQualifiedRead function
void mtsTestComponent::TestFunction6(const int & qualifier, int & result) const
{
    std::cout << "In TestFunction6: mtsCommandQualifiedRead, qualider = " << qualifier << std::endl;
    result = 1000;
}

// ----------------------------------------------------------------------------
