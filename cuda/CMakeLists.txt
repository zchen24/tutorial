cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(CUDAFE LANGUAGES CXX CUDA)   # CUDA for Engineers
set (CMAKE_CXX_STANDARD 11)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(dist_v1)
add_subdirectory(dist_v1_cuda)
add_subdirectory(dist_v2)
add_subdirectory(dist_v2_cuda)
add_subdirectory(dist_v2_cuda_unified)
add_subdirectory(dist_2d)
add_subdirectory(norm)
add_subdirectory(printf)
