cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(CUDAFE LANGUAGES CXX CUDA)   # CUDA for Engineers
set (CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# chapter 0
add_subdirectory(printf)

add_subdirectory(dist_v1)
add_subdirectory(dist_v1_cuda)
add_subdirectory(dist_v2)
add_subdirectory(dist_v2_cuda)
add_subdirectory(dist_v2_cuda_unified)

# chapter 4
add_subdirectory(dist_2d)

# chapter 5
add_subdirectory(sharpen)


# chapter 8: cuda lib
add_subdirectory(norm)
add_subdirectory(dist_1d_thrust)
add_subdirectory(curand_pi)
add_subdirectory(sharpen_npp)
add_subdirectory(tricoloring)
add_subdirectory(linreg)
