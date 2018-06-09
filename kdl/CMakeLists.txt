cmake_minimum_required(VERSION 2.8.3)
project(kdl_tutorial)

set(CMAKE_CXX_STANDARD 11)

# KDL
find_package(orocos_kdl REQUIRED)
include_directories(${orocos_kdl_INCLUDE_DIRS})
link_directories(${orocos_kdl_LIBRARY_DIRS})

## Cisst
#set(REQUIRED_CISST_LIBRARIES
#  cisstCommon
#  cisstNumerical
#  cisstVector
#  cisstRobot)
#find_package(cisst REQUIRED ${REQUIRED_CISST_LIBRARIES} )
#include( ${CISST_USE_FILE})
#
### Declare a cpp executable
#add_executable(kdlTutorial main.cpp)
#target_link_libraries(kdlTutorial ${orocos_kdl_LIBRARIES})
#cisst_target_link_libraries(kdlTutorial
#  ${REQUIRED_CISST_LIBRARIES})

## wrench
add_executable(kdlWrench mainWrench.cpp)
target_link_libraries(kdlWrench ${orocos_kdl_LIBRARIES})

## trajectory
add_executable(kdlTraj mainTrajectory.cpp)
target_link_libraries(kdlTraj ${orocos_kdl_LIBRARIES})
