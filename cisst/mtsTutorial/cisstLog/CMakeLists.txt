# (C) Copyright 2014 Johns Hopkins University (JHU), All Rights Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---
cmake_minimum_required (VERSION 2.8)
project (cisstLog)

# create a list of required cisst libraries
set (REQUIRED_CISST_LIBRARIES
  cisstCommon
  cisstVector
  cisstOSAbstraction)

# find cisst and make sure the required libraries have been compiled
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND_AS_REQUIRED)

  # load cisst configuration
  include (${CISST_USE_FILE})
  include_directories(include)

  add_executable (cisstLog
    main.cpp)
  
  # link against cisst libraries (and dependencies)
  cisst_target_link_libraries (cisstLog ${REQUIRED_CISST_LIBRARIES})

endif (cisst_FOUND_AS_REQUIRED)