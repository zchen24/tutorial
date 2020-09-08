cmake_minimum_required(VERSION 3.0)
project(Acontis)

set (Acontis_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SDK/INC ${CMAKE_CURRENT_SOURCE_DIR}/SDK/INC/Linux)
set (Acontis_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64)
set (Acontis_LIBRARIES  AtemRasSrv EcMaster)
set (Acontis_EXTRA_LIBRARIES pthread dl rt)

set(Acontis_VERSION_MAJOR "3")
set(Acontis_VERSION_MINOR "0")
set(Acontis_VERSION_SERVICEPACK "3")
set(Acontis_VERSION_BUILD "100")
set(Acontis_VERSION "${Acontis_VERSION_MAJOR}.${Acontis_VERSION_MINOR}.${Acontis_VERSION_SERVICEPACK}.${Acontis_VERSION_BUILD}")


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Examples/EcMasterDemo
  ${CMAKE_CURRENT_SOURCE_DIR}/SDK/INC/Linux
  ${CMAKE_CURRENT_SOURCE_DIR}/SDK/INC
  ${CMAKE_CURRENT_SOURCE_DIR}/Sources/Common)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/SDK/LIB/Linux/x64)

add_executable(EcMasterDemo 
  ${CMAKE_CURRENT_SOURCE_DIR}/Examples/EcMasterDemo/ATEMDemo.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Examples/EcMasterDemo/ATEMDemoMain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Examples/EcMasterDemo/ecatDemoCommon.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Examples/EcMasterDemo/ecatNotification.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Examples/EcMasterDemo/Logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Examples/EcMasterDemo/selectLinkLayer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sources/Common/EcTimer.cpp
  )
target_link_libraries(EcMasterDemo ${Acontis_LIBRARIES} ${Acontis_EXTRA_LIBRARIES})


# Generate AcontisConfig.cmake
set (CONF_INCLUDE_DIR ${Acontis_INCLUDE_DIR})
set (CONF_LIBRARY_DIR ${Acontis_LIBRARY_DIR})
set (CONF_LIBRARIES   ${Acontis_LIBRARIES} ${Acontis_EXTRA_LIBRARIES})
configure_file (AcontisConfig.cmake.in
  "${Acontis_BINARY_DIR}/AcontisConfig.cmake" @ONLY)