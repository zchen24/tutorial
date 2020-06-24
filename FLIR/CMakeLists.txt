cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(FLIR)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(Spinnaker REQUIRED)
include_directories(${Spinnaker_INCLUDE_DIRS})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(PROBLEMS
  00_BasicSetup
  01_Acquisition
  02_Exposure
  04_AcquisitionMaxFrameRate
  09_Trigger
  99_FLIR_CV2)

foreach(problem ${PROBLEMS})
  add_executable(${problem} ${problem}.cpp)
  target_link_libraries(${problem}
    ${Spinnaker_LIBRARIES}
    ${OpenCV_LIBRARIES})
endforeach(problem)
