# (C) Copyright 2020 Johns Hopkins University (JHU), All Rights Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---
cmake_minimum_required (VERSION 2.8)
project (mtsPreemptRT)

# create a list of required cisst libraries
set (REQUIRED_CISST_LIBRARIES
  cisstCommon
  cisstVector
  cisstOSAbstraction
  cisstMultiTask)

# find cisst and make sure the required libraries have been compiled
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})
include (${CISST_USE_FILE})

add_executable(mtsPreemptRT
  mainPreemptRT.cpp
  mtsPreemptRT.h
  mtsPreemptRT.cpp)
cisst_target_link_libraries (mtsPreemptRT ${REQUIRED_CISST_LIBRARIES})
