# See license at http://www.cisst.org/cisst/license.txt


project(svlTutorial5)
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

  add_executable(${PROJECT_NAME}
                 CMyEventHandler2.h
                 CMyEventHandler2.cpp
                 main.cpp
                 )

  cisst_target_link_libraries(${PROJECT_NAME} ${REQUIRED_CISST_LIBRARIES})

  # Copy media files to the build directory
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/wall.bmp ${CMAKE_BINARY_DIR}/wall.bmp COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/xray.avi ${CMAKE_BINARY_DIR}/xray.avi COPYONLY)

endif(cisst_FOUND)

