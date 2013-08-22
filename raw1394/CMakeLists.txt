cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(raw1394_tutorial)

# Set the ouptut path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

add_executable(main1394 src/main1394.cpp)
target_link_libraries(main1394 raw1394)

