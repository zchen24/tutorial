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

#include <iostream>

#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>

#include "mtsTestComponentClient.h"

CMN_IMPLEMENT_SERVICES_DERIVED(mtsTestComponentClient, mtsComponent);


// ====================== Constructors and Destructors ========================
// ----------------------------------------------------------------------------
mtsTestComponentClient::mtsTestComponentClient(const std::string & componentName):
    mtsComponent(componentName)
{
    // Required Interface
    mtsInterfaceRequired* required = AddInterfaceRequired("ControllerClient");
    if (required) {
        required->AddFunction("TestFunction1", TestComp.TestFunction1);
        required->AddFunction("TestFunction2", TestComp.TestFunction2);
        required->AddFunction("TestFunction3", TestComp.TestFunction3);
        required->AddFunction("TestFunction4", TestComp.TestFunction4);
        required->AddFunction("TestFunction5", TestComp.TestFunction5);
        required->AddFunction("TestFunction6", TestComp.TestFunction6);
    }


}

// ----------------------------------------------------------------------------
mtsTestComponentClient::~mtsTestComponentClient()
{

}

// ----------------------------------------------------------------------------  

// =========================== Essential Functions ============================
void mtsTestComponentClient::Configure(void)
{
}


void mtsTestComponentClient::RunClient(void)
{
    std::cout << "RunClient" << std::endl;

    std::cout << "Running TestFunction1..." << std::endl;
    TestComp.TestFunction1();
    std::cout << "Finished Running TestFunction1..." << std::endl;


    int resultFunc2 = 10;
    std::cout << "Running TestFunction2..." << std::endl;
    TestComp.TestFunction2(resultFunc2);
    std::cout << "Finished Running TestFunction2..." << std::endl;


    int valueFunc3 = 1;
    std::cout << "Running TestFunction3..." << std::endl;
    TestComp.TestFunction3(valueFunc3);
    std::cout << "Finished Running TestFunction3..." << std::endl;


    int resultFunc4 = 2;
    const int valueFunc4 = 3;
    std::cout << "Running TestFunction4..." << std::endl;
    TestComp.TestFunction4(valueFunc4, resultFunc4);
    std::cout << "Finished Running TestFunction4..." << std::endl;


    int resultFunc5;
    std::cout << "Running TestFunction5..." << std::endl;
    TestComp.TestFunction5(resultFunc5);
    std::cout << "Finished Running TestFunction5 with result = " << resultFunc5 << "..." << std::endl;


    int resultFunc6;
    int qualifierFunc6 = 50;
    std::cout << "Running TestFunction6..." << std::endl;
    TestComp.TestFunction6(qualifierFunc6, resultFunc6);
    std::cout << "Finished Running TestFunction6 with result = " << resultFunc6 << "..." << std::endl;

}
