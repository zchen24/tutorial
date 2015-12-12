cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(boost_tutorial)

# Set the ouptut path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# string example
add_executable(shared_ptr shared_ptr_test.cpp)
