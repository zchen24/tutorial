cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(valgrind_tutorial)

# Set the output path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# valgrind
add_executable(test_valgrind mainValgrind.cpp)

# callgrind
add_executable(test_callgrind mainCallgrind.cpp)

