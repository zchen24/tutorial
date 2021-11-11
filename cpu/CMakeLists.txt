cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(cpu_tutorial)
set(CMAKE_CXX_STANDARD 11)

# Set the output path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

add_executable(num_of_cpus num_of_cpus.cpp)

