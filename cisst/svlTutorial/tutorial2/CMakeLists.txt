# See license at http://www.cisst.org/cisst/license.txt

project(svlTutorial2)
cmake_minimum_required(VERSION 2.6)

set (REQUIRED_CISST_LIBRARIES cisstCommon
                              cisstVector
                              cisstNumerical
                              cisstOSAbstraction
                              cisstMultiTask
                              cisstStereoVision
                              )

# Setup dependency to CISST libraries
find_package(cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if(cisst_FOUND)

  # Inherit project settings from CISST (include directories, library directories, compiler options, etc.)
  include(${CISST_USE_FILE})

  # Create executable file 'svlTutorial1' from source file 'main.cpp'
  add_executable(${PROJECT_NAME}
                 CMyEventHandler.h
                 CMyEventHandler.cpp
                 main.cpp)

  # Specify which CISST libraries need to be linked against
  cisst_target_link_libraries(${PROJECT_NAME} ${REQUIRED_CISST_LIBRARIES})

endif(cisst_FOUND)