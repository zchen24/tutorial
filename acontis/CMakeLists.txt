cmake_minimum_required(VERSION 3.0)
project(AcontisDemo)

option(BUILD_EXAMPLE_SOEM "Build SOEM-based Example" OFF)
if(BUILD_EXAMPLE_SOEM)
  find_package(soem REQUIRED)
  add_executable(DemoLED
    mainLED.cpp
    EcatDefsLED.h)
  target_link_libraries(DemoLED soem)
endif(BUILD_EXAMPLE_SOEM)


find_package(Acontis REQUIRED)
include_directories(${Acontis_INCLUDE_DIR})
link_directories(${Acontis_LIBRARY_DIR})

set (REQUIRED_CISST_LIBRARIES
     cisstCommon
     cisstCommonXML
     cisstVector
     cisstOSAbstraction
     cisstMultiTask
     cisstNumerical
     )
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})
include (${CISST_USE_FILE})


add_executable(DemoAcontis
  mainAcontis.cpp
  Logging.h
  Logging.cpp
  selectLinkLayer.h
  selectLinkLayer.cpp
  SlaveInfo.h
  SlaveInfo.cpp
  ecatNotification.h
  ecatNotification.cpp
  EcTimer.h
  EcTimer.cpp
  EcList.h)
target_link_libraries(DemoAcontis ${Acontis_LIBRARIES})
cisst_target_link_libraries (DemoAcontis ${REQUIRED_CISST_LIBRARIES})

add_custom_command(
  TARGET DemoAcontis POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/libemllSockRaw.so
  ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
  TARGET DemoAcontis POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/ENI_1xmc48led.xml
  ${CMAKE_CURRENT_BINARY_DIR})
