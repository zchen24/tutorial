cmake_minimum_required(VERSION 2.8.3)
project(cisstRobotKDL)

# KDL
find_package(orocos_kdl REQUIRED)
include_directories(${orocos_kdl_INCLUDE_DIRS})
link_directories(${orocos_kdl_LIBRARY_DIRS})

# Cisst
# set(REQUIRED_CISST_LIBRARIES
#   cisstCommon
#   cisstNumerical
#   cisstVector
#   cisstRobot)
# find_package(cisst REQUIRED ${REQUIRED_CISST_LIBRARIES} )
# include( ${CISST_USE_FILE})

## Declare a cpp executable
add_executable(kdl main.cpp)
target_link_libraries(kdl ${orocos_kdl_LIBRARIES})
cisst_target_link_libraries(kdl
  ${REQUIRED_CISST_LIBRARIES})

add_executable(kdl_mtm main_mtm.cpp)
target_link_libraries(kdl_mtm ${orocos_kdl_LIBRARIES})
cisst_target_link_libraries(kdl_mtm
  ${REQUIRED_CISST_LIBRARIES})

add_executable(kdl_psm main_psm.cpp)
target_link_libraries(kdl_psm ${orocos_kdl_LIBRARIES})
cisst_target_link_libraries(kdl_psm
  ${REQUIRED_CISST_LIBRARIES})

add_executable(kdl_rrbot main_rrbot.cpp)
target_link_libraries(kdl_rrbot ${orocos_kdl_LIBRARIES})
cisst_target_link_libraries(kdl_rrbot
  ${REQUIRED_CISST_LIBRARIES})
