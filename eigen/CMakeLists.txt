cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(eigen_tutorial)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED NO_MODULE)
# Uncomment to use the old style ...
# include_directories(${EIGEN3_INCLUDE_DIR})  

# Set the output path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# matrix
add_executable(matrix matrix.cpp)
target_link_libraries(matrix Eigen3::Eigen)

# map from raw memory
add_executable(matrix_map matrix_map.cpp)
target_link_libraries(matrix_map Eigen3::Eigen)

# matrix block access
add_executable(matrix_block matrix_block.cpp)
target_link_libraries(matrix_block Eigen3::Eigen)

