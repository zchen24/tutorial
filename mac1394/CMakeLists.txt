cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(mac1394_tutorial)

# Set the output path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

if(APPLE)
    message(STATUS "this is apple")

    # Apple Carbon & IOKit framework
    find_library(CARBON_LIBRARY Carbon)
    find_library(IOKIT_LIBRARY IOKit)

    include_directories(${CARBON_LIBRARY})
    include_directories(${IOKIT_LIBRARY})
    
    add_executable(mac1394 src/mac1394.cpp)
    target_link_libraries(mac1394
      ${CARBON_LIBRARY}
      ${IOKIT_LIBRARY})
endif(APPLE)

