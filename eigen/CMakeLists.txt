cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(eigen_tutorial)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# Set the ouptut path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# matrix
add_executable(matrix main.cpp)


