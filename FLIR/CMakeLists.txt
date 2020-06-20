cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(FLIR)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(Spinnaker REQUIRED)
include_directories(${Spinnaker_INCLUDE_DIRS})


set(PROBLEMS
  00_BasicSetup
  01_Acquisition
  02_Exposure
  09_Trigger)

foreach(problem ${PROBLEMS})
  add_executable(${problem} ${problem}.cpp)
  target_link_libraries(${problem} ${Spinnaker_LIBRARIES})
endforeach(problem)
