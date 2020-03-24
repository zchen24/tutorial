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

#include <cisstMultiTask.h>   // mtsManagerLocal

#include "mtsTestComponent.h"
#include "mtsTestComponentClient.h"


int main( int argc, const char* argv[] )
{
    // single process version
    mtsManagerLocal* componentManager = mtsManagerLocal::GetInstance();

    // - create test component
    // - configure if necessary
    // - add to component manager
    mtsTestComponent testComponent("Test", 5);
    componentManager->AddComponent(&testComponent);

    // repeat for client
    mtsTestComponentClient testComponentClient("TestClient");
    componentManager->AddComponent(&testComponentClient);

    // connect interface
    // required --> provided
    componentManager->Connect(testComponentClient.GetName(), "ControllerClient",
                              testComponent.GetName(), "Controller");

    componentManager->CreateAll();
    componentManager->StartAll();

    for (size_t i = 0; i < 5; i++) {
        std::cout << "i = " << i << std::endl;
        testComponentClient.RunClient();
    }

    componentManager->KillAll();
    componentManager->Cleanup();

    // stop all logs
    cmnLogger::Kill();

    return 0;
}
