# See license at http://www.cisst.org/cisst/license.txt

project(svlTutorial3)
cmake_minimum_required(VERSION 2.6)

set (REQUIRED_CISST_LIBRARIES cisstCommon
                              cisstVector
                              cisstNumerical
                              cisstOSAbstraction
                              cisstMultiTask
                              cisstStereoVision
                              )

find_package(cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})
                            
if(cisst_FOUND)

    include(${CISST_USE_FILE})
    add_executable(${PROJECT_NAME}
                   CMyFilter1.h
                   CMyFilter1.cpp
                   main.cpp
                   )

    cisst_target_link_libraries(${PROJECT_NAME} ${REQUIRED_CISST_LIBRARIES})

endif(cisst_FOUND)